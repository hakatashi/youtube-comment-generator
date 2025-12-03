"""音声バッファ管理モジュール"""

import io
import threading
import time
import wave
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


class AudioChunk:
    """音声チャンクを表すクラス"""
    def __init__(self, chunk_id: int, timestamp: float, samples: np.ndarray):
        self.chunk_id = chunk_id
        self.timestamp = timestamp
        self.samples = samples


class UserAudioBuffer:
    """ユーザーごとの音声データをバッファリングするクラス"""

    # チャンク結合の閾値（秒）: この時間以内の隙間は同一チャンクとみなす
    CHUNK_MERGE_EPSILON = 0.1

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

        # 音声チャンクバッファ（チャンク単位で管理）
        self.chunks: Deque[AudioChunk] = deque()
        self._next_chunk_id = 0

        # スレッドセーフ用のロック
        self._lock = threading.Lock()

    def add_audio(self, audio_data: bytes) -> None:
        """
        音声データをバッファに追加（チャンク単位、連続チャンクを自動マージ）

        連続するチャンク（タイムスタンプの差が CHUNK_MERGE_EPSILON 以内）は
        同一チャンクとして結合されます。

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
            # 最後のチャンクと時間差をチェック
            should_merge = False
            if self.chunks:
                last_chunk = self.chunks[-1]
                time_diff = current_time - last_chunk.timestamp

                # 時間差が閾値以内なら同一チャンクとして扱う
                if time_diff <= self.CHUNK_MERGE_EPSILON:
                    should_merge = True

            if should_merge:
                # 既存のチャンクに音声を追加（マージ）
                last_chunk = self.chunks[-1]
                last_chunk.samples = np.concatenate([last_chunk.samples, audio_array])
                # タイムスタンプは最初のチャンクのものを保持
            else:
                # 新しいチャンクとして追加
                chunk = AudioChunk(
                    chunk_id=self._next_chunk_id,
                    timestamp=current_time,
                    samples=audio_array
                )
                self.chunks.append(chunk)
                self._next_chunk_id += 1

            # buffer_duration秒より古いチャンクを削除
            cutoff_time = current_time - self.buffer_duration
            while self.chunks and self.chunks[0].timestamp < cutoff_time:
                self.chunks.popleft()

    def get_audio(self, duration: Optional[int] = None) -> np.ndarray:
        """
        バッファから音声データを取得（時刻ベース、チャンク単位）

        Args:
            duration: 取得する時間（秒）。Noneの場合は全データ
                     指定された場合、現在時刻から過去duration秒以内のデータを取得

        Returns:
            音声データ（numpy配列）
        """
        with self._lock:
            if not self.chunks:
                return np.array([], dtype=np.int16)

            if duration is None:
                # 全チャンクのデータを結合して返す
                all_samples = []
                for chunk in self.chunks:
                    all_samples.append(chunk.samples)
                return np.concatenate(all_samples) if all_samples else np.array([], dtype=np.int16)

            # 現在時刻から指定時間前までのデータを取得
            current_time = time.time()
            cutoff_time = current_time - duration

            # cutoff_time以降のチャンクのみを取得
            filtered_samples = []
            for chunk in self.chunks:
                if chunk.timestamp >= cutoff_time:
                    filtered_samples.append(chunk.samples)

            if not filtered_samples:
                return np.array([], dtype=np.int16)

            return np.concatenate(filtered_samples)

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
            self.chunks.clear()

    def get_duration(self) -> float:
        """現在バッファに保存されている音声の長さ（秒）を取得"""
        with self._lock:
            total_samples = sum(len(chunk.samples) for chunk in self.chunks)
            return total_samples / self.sample_rate

    def is_ready(self, required_duration: int = 60) -> bool:
        """
        指定時間以内にデータがあるかチェック（時刻ベース）

        Args:
            required_duration: チェックする時間範囲（秒）

        Returns:
            指定時間以内にデータがある場合True
        """
        with self._lock:
            if not self.chunks:
                return False

            # 現在時刻から指定時間前までにデータがあるかチェック
            current_time = time.time()
            cutoff_time = current_time - required_duration

            # cutoff_time以降のチャンクが存在するかチェック
            return any(chunk.timestamp >= cutoff_time for chunk in self.chunks)


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

    def get_merged_audio(
        self, user_ids: Optional[List[int]] = None, duration: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Tuple[int, float, float]]]:
        """
        複数ユーザーの音声をチャンク単位で時系列順に結合して取得

        各ユーザーのチャンク（連続する音声データ）を分割せずに結合します。

        Args:
            user_ids: 対象ユーザーIDのリスト。Noneの場合は全ユーザー
            duration: 取得する時間（秒）。Noneの場合は全データ

        Returns:
            タプル (結合された音声データ, 話者情報リスト)
            話者情報リスト: [(user_id, start_time, end_time), ...]
                start_time/end_time: 秒単位での相対時間
        """
        if user_ids is None:
            user_ids = self.get_user_ids()

        if not user_ids:
            return np.array([], dtype=np.int16), []

        # 現在時刻と時間範囲を決定
        current_time = time.time()
        cutoff_time = current_time - duration if duration else None

        # 全ユーザーの音声チャンクをタイムスタンプ付きで収集
        all_chunks: List[Tuple[float, int, AudioChunk]] = []  # (timestamp, user_id, chunk)

        for user_id in user_ids:
            if user_id not in self.user_buffers:
                continue

            user_buffer = self.user_buffers[user_id]
            with user_buffer._lock:
                for chunk in user_buffer.chunks:
                    if cutoff_time is None or chunk.timestamp >= cutoff_time:
                        all_chunks.append((chunk.timestamp, user_id, chunk))

        if not all_chunks:
            return np.array([], dtype=np.int16), []

        # タイムスタンプでソート（チャンク単位）
        all_chunks.sort(key=lambda x: x[0])

        # 最も古いタイムスタンプを基準時刻として設定
        base_timestamp = all_chunks[0][0]

        # 音声データと話者情報を構築（チャンク単位）
        merged_audio_parts = []
        speaker_segments = []

        current_position = 0.0  # 秒単位での現在位置

        for timestamp, user_id, chunk in all_chunks:
            # チャンクの音声データを追加
            merged_audio_parts.append(chunk.samples)

            # チャンクの長さを計算（秒）
            chunk_duration = len(chunk.samples) / self.sample_rate

            # 話者セグメント情報を記録
            segment_start = current_position
            segment_end = current_position + chunk_duration
            speaker_segments.append((user_id, segment_start, segment_end))

            # 位置を更新
            current_position = segment_end

        # すべてのチャンクを結合
        merged_audio = np.concatenate(merged_audio_parts) if merged_audio_parts else np.array([], dtype=np.int16)

        return merged_audio, speaker_segments

    def get_merged_wav_bytes(
        self, user_ids: Optional[List[int]] = None, duration: Optional[int] = None
    ) -> bytes:
        """
        複数ユーザーの音声を結合してWAV形式のバイト列を取得

        Args:
            user_ids: 対象ユーザーIDのリスト。Noneの場合は全ユーザー
            duration: 取得する時間（秒）。Noneの場合は全データ

        Returns:
            WAV形式のバイト列
        """
        audio_data, _ = self.get_merged_audio(user_ids, duration)

        # WAVファイルとしてバイト列に変換
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # モノラル
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return wav_buffer.getvalue()
