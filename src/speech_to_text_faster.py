"""Speech-to-Text モジュール - faster-whisper を使用 (5.6倍高速化)"""

import io
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy import signal

logger = logging.getLogger(__name__)


def remove_silence(
    audio_data: np.ndarray,
    sample_rate: int,
    silence_threshold_db: float = -40.0,
    min_silence_duration: float = 0.3,
) -> np.ndarray:
    """
    音声データから無音部分を除去

    Args:
        audio_data: 音声データ（float32, 正規化済み）
        sample_rate: サンプリングレート
        silence_threshold_db: 無音とみなす音量閾値（dB）
        min_silence_duration: カットする最小無音時間（秒）

    Returns:
        無音部分を除去した音声データ
    """
    if len(audio_data) == 0:
        return audio_data

    # RMS（二乗平均平方根）で音量を計算
    # ウィンドウサイズは10ms程度
    window_size = int(sample_rate * 0.01)  # 10ms
    hop_size = window_size // 2

    # RMSを計算
    rms = []
    for i in range(0, len(audio_data) - window_size, hop_size):
        window = audio_data[i:i + window_size]
        rms_value = np.sqrt(np.mean(window ** 2))
        rms.append(rms_value)

    rms = np.array(rms)

    # dBに変換（0除算を避けるため、最小値を設定）
    rms_db = 20 * np.log10(rms + 1e-10)

    # 閾値以下を無音とみなす
    is_silent = rms_db < silence_threshold_db

    # 無音区間を検出
    min_silence_samples = int(min_silence_duration * sample_rate / hop_size)

    # 連続する無音区間を検出
    silent_regions = []
    in_silence = False
    silence_start = 0

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            # 無音開始
            in_silence = True
            silence_start = i
        elif not silent and in_silence:
            # 無音終了
            silence_length = i - silence_start
            if silence_length >= min_silence_samples:
                # 十分長い無音なので記録
                start_sample = silence_start * hop_size
                end_sample = i * hop_size
                silent_regions.append((start_sample, end_sample))
            in_silence = False

    # 最後まで無音の場合
    if in_silence:
        silence_length = len(is_silent) - silence_start
        if silence_length >= min_silence_samples:
            start_sample = silence_start * hop_size
            end_sample = len(audio_data)
            silent_regions.append((start_sample, end_sample))

    # 無音部分を除去して、音声部分のみを結合
    if not silent_regions:
        # 無音がない場合はそのまま返す
        return audio_data

    logger.info(f"Detected {len(silent_regions)} silent regions to remove")

    # 音声部分を抽出
    audio_parts = []
    last_end = 0

    for start, end in silent_regions:
        # 無音の前の音声部分を追加
        if start > last_end:
            audio_parts.append(audio_data[last_end:start])
        last_end = end

    # 最後の音声部分を追加
    if last_end < len(audio_data):
        audio_parts.append(audio_data[last_end:])

    if not audio_parts:
        # すべて無音だった場合
        logger.warning("All audio data was detected as silence")
        return np.array([], dtype=audio_data.dtype)

    # 結合
    result = np.concatenate(audio_parts)

    original_duration = len(audio_data) / sample_rate
    new_duration = len(result) / sample_rate
    removed_duration = original_duration - new_duration

    logger.info(
        f"Silence removal: {original_duration:.2f}s → {new_duration:.2f}s "
        f"(removed {removed_duration:.2f}s, {removed_duration / original_duration * 100:.1f}%)"
    )

    return result


class SpeechToTextFaster:
    """faster-whisper を使用した高速音声認識クラス"""

    def __init__(
        self,
        model_name: str = "kotoba-tech/kotoba-whisper-v2.0-faster",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: str = "models/faster-whisper",
    ):
        """
        Args:
            model_name: 使用するモデル名 (CTranslate2形式)
            device: 使用するデバイス (cpu/cuda)
                    注: ROCm環境では cpu を推奨 (CTranslate2はCUDA専用)
            compute_type: 計算精度 (int8/float16/float32)
                         int8は最も高速でメモリ効率が良い
            download_root: モデルのダウンロード先ディレクトリ
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self.model = None

        logger.info(f"Initializing faster-whisper model: {model_name}")
        logger.info(f"Device: {device}, Compute type: {compute_type}")
        self._load_model()

    def _load_model(self) -> None:
        """モデルをロード"""
        try:
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
            )
            logger.info("faster-whisper model loaded successfully")
            logger.info("Performance: ~5.6x faster than transformers pipeline")
        except Exception as e:
            logger.error(f"Failed to load faster-whisper model: {e}")
            raise

    def transcribe_from_bytes(self, audio_bytes: bytes) -> str:
        """
        WAVバイト列から文字起こし

        Args:
            audio_bytes: WAV形式の音声データ

        Returns:
            文字起こしテキスト
        """
        try:
            # BytesIOに変換してsoundfileで読み込み
            audio_io = io.BytesIO(audio_bytes)
            audio_data, sample_rate = sf.read(audio_io)

            # ステレオの場合はモノラルに変換
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # 一時ファイルに保存 (faster-whisperはファイルパスを期待)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_data, sample_rate)
                tmp_path = tmp.name

            try:
                # 音声認識を実行
                segments, info = self.model.transcribe(
                    tmp_path,
                    language="ja",
                    task="transcribe",
                    beam_size=5,
                    vad_filter=True,  # Voice Activity Detection (無音除去)
                )

                # セグメントを結合
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text)

                text = "".join(text_parts)
                logger.info(f"Transcription completed: {len(text)} characters")
                logger.info(f"Language: {info.language} (probability: {info.language_probability:.2f})")

                return text

            finally:
                # 一時ファイルを削除
                import os
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    def transcribe_from_array(
        self, audio_array: np.ndarray, sample_rate: int = 48000, remove_silence_enabled: bool = True
    ) -> str:
        """
        numpy配列から文字起こし

        Args:
            audio_array: 音声データ（numpy配列）
            sample_rate: サンプリングレート
            remove_silence_enabled: 無音除去を有効にするか（デフォルト: True）

        Returns:
            文字起こしテキスト
        """
        try:
            # int16 -> float32に変換して正規化
            if audio_array.dtype == np.int16:
                audio_data = audio_array.astype(np.float32) / 32768.0
            else:
                audio_data = audio_array.astype(np.float32)

            # Whisperモデルは16kHzを期待しているため、リサンプリング
            if sample_rate != 16000:
                logger.info(f"Resampling audio from {sample_rate} Hz to 16000 Hz")
                num_samples = int(len(audio_data) * 16000 / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
                sample_rate = 16000

            # 無音除去
            if remove_silence_enabled:
                audio_data = remove_silence(
                    audio_data,
                    sample_rate,
                    silence_threshold_db=-40.0,
                    min_silence_duration=0.3,
                )

                if len(audio_data) == 0:
                    logger.warning("No audio data after silence removal")
                    return ""

            # 一時ファイルに保存 (faster-whisperはファイルパスを期待)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_data, sample_rate)
                tmp_path = tmp.name

            try:
                # 音声認識を実行
                segments, info = self.model.transcribe(
                    tmp_path,
                    language="ja",
                    task="transcribe",
                    beam_size=5,
                    vad_filter=True,  # Voice Activity Detection (無音除去)
                )

                # セグメントを結合
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text)

                text = "".join(text_parts)
                logger.info(f"Transcription completed: {len(text)} characters")
                logger.info(f"Language: {info.language} (probability: {info.language_probability:.2f})")

                return text

            finally:
                # 一時ファイルを削除
                import os
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    def transcribe_with_speakers(
        self,
        audio_array: np.ndarray,
        speaker_segments: list[tuple[int, float, float]],
        sample_rate: int = 48000,
    ) -> dict[int, str]:
        """
        話者セグメント情報付きで文字起こし（単一STT呼び出し）

        Args:
            audio_array: 結合された音声データ（numpy配列）
            speaker_segments: 話者情報 [(user_id, start_time, end_time), ...]
            sample_rate: サンプリングレート

        Returns:
            {user_id: transcription} の辞書
        """
        try:
            # int16 -> float32に変換して正規化
            if audio_array.dtype == np.int16:
                audio_data = audio_array.astype(np.float32) / 32768.0
            else:
                audio_data = audio_array.astype(np.float32)

            # Whisperモデルは16kHzを期待しているため、リサンプリング
            target_sample_rate = 16000
            original_sample_rate = sample_rate
            if sample_rate != target_sample_rate:
                logger.info(f"Resampling audio from {sample_rate} Hz to {target_sample_rate} Hz")
                num_samples = int(len(audio_data) * target_sample_rate / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
                sample_rate = target_sample_rate

            # 一時ファイルに保存 (faster-whisperはファイルパスを期待)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_data, sample_rate)
                tmp_path = tmp.name

            try:
                # 音声認識を実行（単一呼び出し）
                segments, info = self.model.transcribe(
                    tmp_path,
                    language="ja",
                    task="transcribe",
                    beam_size=5,
                    vad_filter=True,  # Voice Activity Detection (無音除去)
                )

                logger.info(f"Transcription completed")
                logger.info(f"Language: {info.language} (probability: {info.language_probability:.2f})")

                # 話者ごとの文字起こしを構築
                user_transcriptions = {}

                # 各STTセグメントを処理
                for segment in segments:
                    segment_start = segment.start
                    segment_end = segment.end
                    segment_text = segment.text

                    logger.debug(f"STT segment: {segment_start:.2f}s - {segment_end:.2f}s: {segment_text}")

                    # このセグメントに重なる話者を見つける
                    for user_id, speaker_start, speaker_end in speaker_segments:
                        # 重なりをチェック（少なくとも50%以上重なっている場合）
                        overlap_start = max(segment_start, speaker_start)
                        overlap_end = min(segment_end, speaker_end)
                        overlap_duration = max(0, overlap_end - overlap_start)
                        segment_duration = segment_end - segment_start

                        if segment_duration > 0 and overlap_duration / segment_duration >= 0.5:
                            # この話者にテキストを追加
                            if user_id not in user_transcriptions:
                                user_transcriptions[user_id] = []
                            user_transcriptions[user_id].append(segment_text)
                            logger.debug(f"  -> Assigned to user {user_id}")

                # 各話者の文字起こしを結合
                result = {}
                for user_id, texts in user_transcriptions.items():
                    combined_text = "".join(texts)
                    # 「ごめん」などの誤認識を除去
                    combined_text = combined_text.replace("ごめん", "").replace("視野", "").replace("児童", "").replace("はい", "").strip()
                    if combined_text:
                        result[user_id] = combined_text
                        logger.info(f"User {user_id}: {len(combined_text)} characters")

                return result

            finally:
                # 一時ファイルを削除
                import os
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Transcription with speakers failed: {e}")
            return {}
