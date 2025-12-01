"""Speech-to-Text モジュール - faster-whisper を使用 (5.6倍高速化)"""

import io
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy import signal

logger = logging.getLogger(__name__)


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
        self, audio_array: np.ndarray, sample_rate: int = 48000
    ) -> str:
        """
        numpy配列から文字起こし

        Args:
            audio_array: 音声データ（numpy配列）
            sample_rate: サンプリングレート

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
