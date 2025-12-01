"""音声バッファ管理モジュール"""

import io
import threading
import time
import wave
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


class UserAudioBuffer:
    """ユーザーごとの音声データをバッファリングするクラス"""

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

        # 音声データバッファ（タイムスタンプ付き）: (timestamp, sample)
        self.buffer: Deque[Tuple[float, int]] = deque()

        # スレッドセーフ用のロック
        self._lock = threading.Lock()

    def add_audio(self, audio_data: bytes) -> None:
        """
        音声データをバッファに追加

        Args:
            audio_data: PCM音声データ（int16形式のバイト列）
        """
        # 現在時刻を取得
        current_time = time.time()

        # バイト列をnumpy配列に変換
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # ステレオの場合はモノラルに変換
        if self.channels == 2 and len(audio_array) > 0:
            # ステレオデータを2チャンネルに分割してモノラルに変換
            audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)

        # ロックを取得してバッファに追加
        with self._lock:
            # バッファに追加（タイムスタンプ付き）
            for sample in audio_array:
                self.buffer.append((current_time, sample))

            # buffer_duration秒より古いデータを削除
            cutoff_time = current_time - self.buffer_duration
            while self.buffer and self.buffer[0][0] < cutoff_time:
                self.buffer.popleft()

    def get_audio(self, duration: Optional[int] = None) -> np.ndarray:
        """
        バッファから音声データを取得（時刻ベース）

        Args:
            duration: 取得する時間（秒）。Noneの場合は全データ
                     指定された場合、現在時刻から過去duration秒以内のデータを取得

        Returns:
            音声データ（numpy配列）
        """
        with self._lock:
            if not self.buffer:
                return np.array([], dtype=np.int16)

            if duration is None:
                # 全データを返す（サンプル値のみ）
                return np.array([sample for _, sample in self.buffer], dtype=np.int16)

            # 現在時刻から指定時間前までのデータを取得
            current_time = time.time()
            cutoff_time = current_time - duration

            # cutoff_time以降のデータのみを取得
            filtered_data = [sample for timestamp, sample in self.buffer if timestamp >= cutoff_time]

            if not filtered_data:
                return np.array([], dtype=np.int16)

            return np.array(filtered_data, dtype=np.int16)

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
            wav_file.setnchannels(1)  # モノラル
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return wav_buffer.getvalue()

    def clear(self) -> None:
        """バッファをクリア"""
        with self._lock:
            self.buffer.clear()

    def get_duration(self) -> float:
        """現在バッファに保存されている音声の長さ（秒）を取得"""
        with self._lock:
            return len(self.buffer) / self.sample_rate

    def is_ready(self, required_duration: int = 60) -> bool:
        """
        指定時間以内にデータがあるかチェック（時刻ベース）

        Args:
            required_duration: チェックする時間範囲（秒）

        Returns:
            指定時間以内にデータがある場合True
        """
        with self._lock:
            if not self.buffer:
                return False

            # 現在時刻から指定時間前までにデータがあるかチェック
            current_time = time.time()
            cutoff_time = current_time - required_duration

            # cutoff_time以降のデータが存在するかチェック
            return any(timestamp >= cutoff_time for timestamp, _ in self.buffer)


class AudioBuffer:
    """複数ユーザーの音声データを管理するクラス"""

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

        # ユーザーID -> UserAudioBuffer のマップ
        self.user_buffers: Dict[int, UserAudioBuffer] = {}

        # ユーザーID -> ユーザー名 のマップ
        self.user_names: Dict[int, str] = {}

    def add_audio(self, user_id: int, audio_data: bytes) -> None:
        """
        ユーザーの音声データをバッファに追加

        Args:
            user_id: ユーザーID
            audio_data: PCM音声データ（int16形式のバイト列）
        """
        # ユーザーのバッファが存在しない場合は作成
        if user_id not in self.user_buffers:
            self.user_buffers[user_id] = UserAudioBuffer(
                sample_rate=self.sample_rate,
                channels=self.channels,
                buffer_duration=self.buffer_duration,
            )

        # バッファに追加
        self.user_buffers[user_id].add_audio(audio_data)

    def get_user_ids(self) -> List[int]:
        """
        現在バッファに音声が保存されているユーザーIDのリストを取得

        Returns:
            ユーザーIDのリスト
        """
        return list(self.user_buffers.keys())

    def get_audio(self, user_id: int, duration: Optional[int] = None) -> np.ndarray:
        """
        特定ユーザーのバッファから音声データを取得

        Args:
            user_id: ユーザーID
            duration: 取得する時間（秒）。Noneの場合は全データ

        Returns:
            音声データ（numpy配列）
        """
        if user_id not in self.user_buffers:
            return np.array([], dtype=np.int16)

        return self.user_buffers[user_id].get_audio(duration)

    def get_wav_bytes(self, user_id: int, duration: Optional[int] = None) -> bytes:
        """
        特定ユーザーのバッファからWAV形式のバイト列を取得

        Args:
            user_id: ユーザーID
            duration: 取得する時間（秒）。Noneの場合は全データ

        Returns:
            WAV形式のバイト列
        """
        if user_id not in self.user_buffers:
            # 空のWAVファイルを返す
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
            return wav_buffer.getvalue()

        return self.user_buffers[user_id].get_wav_bytes(duration)

    def get_duration(self, user_id: int) -> float:
        """
        特定ユーザーの現在バッファに保存されている音声の長さ（秒）を取得

        Args:
            user_id: ユーザーID

        Returns:
            音声の長さ（秒）
        """
        if user_id not in self.user_buffers:
            return 0.0

        return self.user_buffers[user_id].get_duration()

    def is_ready(self, user_id: int, required_duration: int = 60) -> bool:
        """
        特定ユーザーの指定時間分のデータが溜まっているかチェック

        Args:
            user_id: ユーザーID
            required_duration: 必要な時間（秒）

        Returns:
            データが溜まっている場合True
        """
        if user_id not in self.user_buffers:
            return False

        return self.user_buffers[user_id].is_ready(required_duration)

    def clear(self, user_id: Optional[int] = None) -> None:
        """
        バッファをクリア

        Args:
            user_id: ユーザーID。Noneの場合は全ユーザーのバッファをクリア
        """
        if user_id is None:
            self.user_buffers.clear()
        elif user_id in self.user_buffers:
            self.user_buffers[user_id].clear()

    def set_user_name(self, user_id: int, user_name: str) -> None:
        """
        ユーザー名を設定

        Args:
            user_id: ユーザーID
            user_name: ユーザー名
        """
        self.user_names[user_id] = user_name

    def get_user_name(self, user_id: int) -> str:
        """
        ユーザー名を取得

        Args:
            user_id: ユーザーID

        Returns:
            ユーザー名（未登録の場合は "Unknown"）
        """
        return self.user_names.get(user_id, "Unknown")
