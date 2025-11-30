"""音声バッファ管理モジュール"""

import io
import wave
from collections import deque
from typing import Deque, Optional

import numpy as np


class AudioBuffer:
    """音声データをバッファリングするクラス"""

    def __init__(self, sample_rate: int = 48000, channels: int = 2, buffer_duration: int = 60):
        """
        Args:
            sample_rate: サンプリングレート (Hz)
            channels: チャンネル数
            buffer_duration: バッファ保持時間 (秒)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_duration = buffer_duration

        # バッファサイズ（サンプル数）
        self.max_samples = sample_rate * buffer_duration

        # 音声データバッファ（dequeで古いデータを自動削除）
        self.buffer: Deque[np.ndarray] = deque(maxlen=self.max_samples)

    def add_audio(self, audio_data: bytes) -> None:
        """
        音声データをバッファに追加

        Args:
            audio_data: PCM音声データ（int16形式のバイト列）
        """
        # バイト列をnumpy配列に変換
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # バッファに追加
        for sample in audio_array:
            self.buffer.append(sample)

    def get_audio(self, duration: Optional[int] = None) -> np.ndarray:
        """
        バッファから音声データを取得

        Args:
            duration: 取得する時間（秒）。Noneの場合は全データ

        Returns:
            音声データ（numpy配列）
        """
        if duration is None:
            # 全データを返す
            return np.array(list(self.buffer), dtype=np.int16)

        # 指定時間分のサンプル数
        samples = min(self.sample_rate * duration, len(self.buffer))

        # 最新のsamples分を取得
        if samples == 0:
            return np.array([], dtype=np.int16)

        return np.array(list(self.buffer)[-samples:], dtype=np.int16)

    def get_wav_bytes(self, duration: Optional[int] = None) -> bytes:
        """
        バッファからWAV形式のバイト列を取得

        Args:
            duration: 取得する時間（秒）。Noneの場合は全データ

        Returns:
            WAV形式のバイト列
        """
        audio_data = self.get_audio(duration)

        # WAVファイルとしてバイト列に変換
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return wav_buffer.getvalue()

    def clear(self) -> None:
        """バッファをクリア"""
        self.buffer.clear()

    def get_duration(self) -> float:
        """現在バッファに保存されている音声の長さ（秒）を取得"""
        return len(self.buffer) / self.sample_rate

    def is_ready(self, required_duration: int = 60) -> bool:
        """
        指定時間分のデータが溜まっているかチェック

        Args:
            required_duration: 必要な時間（秒）

        Returns:
            データが溜まっている場合True
        """
        return self.get_duration() >= required_duration
