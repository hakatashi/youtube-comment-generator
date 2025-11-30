"""Speech-to-Text モジュール - Kotoba-Whisper を使用"""

import io
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal
from transformers import pipeline

logger = logging.getLogger(__name__)


class SpeechToText:
    """Kotoba-Whisper を使用した音声認識クラス"""

    def __init__(
        self,
        model_name: str = "kotoba-tech/kotoba-whisper-v1.1",
        device: str = "cuda",
    ):
        """
        Args:
            model_name: 使用するモデル名
            device: 使用するデバイス (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device
        self.pipe = None

        logger.info(f"Initializing Speech-to-Text model: {model_name}")
        self._load_model()

    def _load_model(self) -> None:
        """モデルをロード"""
        try:
            # Check if CUDA/ROCm is actually available
            cuda_available = torch.cuda.is_available()
            if self.device == "cuda" and not cuda_available:
                logger.warning("CUDA/ROCm not available, falling back to CPU")
                self.device = "cpu"

            # Set device index or -1 for CPU
            if self.device == "cuda" and cuda_available:
                device_id = 0
                dtype = torch.float16
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                device_id = -1
                dtype = torch.float32
                logger.info("Using CPU")

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=device_id,
                torch_dtype=dtype,
            )
            logger.info("Speech-to-Text model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Speech-to-Text model: {e}")
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

            # 音声認識を実行
            result = self.pipe(
                audio_data,
                generate_kwargs={
                    "language": "japanese",
                    "task": "transcribe",
                },
                return_timestamps=False,
            )

            text = result["text"]
            logger.info(f"Transcription completed: {len(text)} characters")
            return text

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

            # 音声認識を実行
            result = self.pipe(
                audio_data,
                generate_kwargs={
                    "language": "japanese",
                    "task": "transcribe",
                },
                return_timestamps=False,
            )

            text = result["text"]
            logger.info(f"Transcription completed: {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""
