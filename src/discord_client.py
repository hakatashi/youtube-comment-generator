"""Discord クライアントモジュール"""

import asyncio
import logging
from typing import Optional

import discord
from discord import VoiceClient
from discord.ext import commands

from .audio_buffer import AudioBuffer

logger = logging.getLogger(__name__)


class AudioBufferSink(discord.sinks.Sink):
    """音声データをAudioBufferに送るカスタムSink"""

    def __init__(self, audio_buffer: AudioBuffer, ignored_user_ids: list[int]):
        super().__init__()
        self.audio_buffer = audio_buffer
        self.ignored_user_ids = set(ignored_user_ids)
        self._bytes_written = {}  # user_id -> bytes のマップ

    def write(self, data, user):
        """音声データを受信してバッファに追加"""
        if user is not None:
            # userはdiscord.Memberオブジェクト
            user_id = int(user.id) if hasattr(user, 'id') else int(user)

            # 無視リストに含まれるユーザーの音声は無視
            if user_id in self.ignored_user_ids:
                logger.debug(f"Ignoring audio from user {user_id}")
                return

            # データが空の場合はスキップ（デコードエラー時）
            if not data or len(data) == 0:
                logger.debug(f"Skipping empty audio data from user {user_id}")
                return

            logger.debug(f"Received {len(data)} bytes from user {user_id}")

            try:
                self.audio_buffer.add_audio(user_id, data)

                # ユーザー名を保存（初回のみ）
                if hasattr(user, 'name') and user_id not in self.audio_buffer.user_names:
                    self.audio_buffer.set_user_name(user_id, user.name)
            except Exception as e:
                logger.warning(f"Failed to add audio data for user {user_id}: {e}")
                return

            # ユーザーごとのバイト数をカウント
            if user_id not in self._bytes_written:
                self._bytes_written[user_id] = 0
            self._bytes_written[user_id] += len(data)

            # 定期的にログを出力（デバッグ用）
            if self._bytes_written[user_id] % (48000 * 2 * 10) < len(data):  # 約10秒ごと
                logger.info(f"User {user_id}: Total audio received: {self._bytes_written[user_id] / (48000 * 2):.1f}s")

    def cleanup(self):
        """クリーンアップ処理"""
        for user_id, bytes_count in self._bytes_written.items():
            logger.info(f"User {user_id}: Sink cleanup - total bytes: {bytes_count}")


class CommentBot(commands.Bot):
    """YouTube風コメント生成Bot"""

    def __init__(
        self,
        guild_id: int,
        voice_channel_id: int,
        text_channel_id: int,
        audio_buffer: AudioBuffer,
        ignored_user_ids: list[int],
    ):
        """
        Args:
            guild_id: サーバーID
            voice_channel_id: ボイスチャンネルID
            text_channel_id: テキストチャンネルID
            audio_buffer: 音声バッファ
            ignored_user_ids: 無視するユーザーIDのリスト
        """
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True
        intents.guild_messages = True
        intents.members = True  # 音声受信に必要

        super().__init__(command_prefix="!", intents=intents)

        self.guild_id = guild_id
        self.voice_channel_id = voice_channel_id
        self.text_channel_id = text_channel_id
        self.audio_buffer = audio_buffer
        self.ignored_user_ids = ignored_user_ids
        self.voice_client: Optional[VoiceClient] = None
        self.is_ready = False

    async def on_ready(self) -> None:
        """Bot起動時の処理"""
        logger.info(f"Logged in as {self.user}")

        # ギルドとボイスチャンネルを取得
        guild = self.get_guild(self.guild_id)
        if guild is None:
            logger.error(f"Guild {self.guild_id} not found")
            return

        voice_channel = guild.get_channel(self.voice_channel_id)
        if voice_channel is None:
            logger.error(f"Voice channel {self.voice_channel_id} not found")
            return

        if not isinstance(voice_channel, discord.VoiceChannel):
            logger.error(f"Channel {self.voice_channel_id} is not a voice channel")
            return

        # ボイスチャンネルに接続
        try:
            logger.info(f"Connecting to voice channel: {voice_channel.name}")

            # reconnect=False を指定して接続の再試行を無効化
            # timeout を長めに設定
            self.voice_client = await voice_channel.connect(
                timeout=60.0,
                reconnect=False,
            )

            # 接続後、少し待機してから録音開始
            await asyncio.sleep(1.0)

            # カスタムSinkを作成して音声受信を開始
            sink = AudioBufferSink(self.audio_buffer, self.ignored_user_ids)
            self.voice_client.start_recording(
                sink,
                self._on_recording_finished,
            )

            logger.info("Started recording")
            self.is_ready = True

        except Exception as e:
            logger.error(f"Failed to connect to voice channel: {e}")

    async def _on_recording_finished(self, sink: discord.sinks.Sink) -> None:
        """
        録音終了時のコールバック（通常は呼ばれない）

        Args:
            sink: 音声シンク
        """
        logger.info("Recording finished")

    async def post_comments(self, comments: list[str]) -> None:
        """
        テキストチャンネルにコメントを投稿

        Args:
            comments: 投稿するコメントのリスト
        """
        if not comments:
            logger.warning("No comments to post")
            return

        try:
            # ギルドとテキストチャンネルを取得
            guild = self.get_guild(self.guild_id)
            if guild is None:
                logger.error(f"Guild {self.guild_id} not found")
                return

            text_channel = guild.get_channel(self.text_channel_id)
            if text_channel is None:
                logger.error(f"Text channel {self.text_channel_id} not found")
                return

            if not isinstance(text_channel, discord.TextChannel):
                logger.error(f"Channel {self.text_channel_id} is not a text channel")
                return

            # コメントを1つのメッセージとして投稿
            message_content = "\n".join(f"{i}. {comment}" for i, comment in enumerate(comments, 1))

            # Discordの文字数制限（2000文字）を考慮
            if len(message_content) > 2000:
                # 長すぎる場合は分割して送信
                current_message = ""
                for i, comment in enumerate(comments, 1):
                    line = f"{i}. {comment}\n"
                    if len(current_message) + len(line) > 2000:
                        await text_channel.send(current_message)
                        current_message = line
                    else:
                        current_message += line

                if current_message:
                    await text_channel.send(current_message)
            else:
                await text_channel.send(message_content)

            logger.info(f"Posted {len(comments)} comments to text channel")

        except Exception as e:
            logger.error(f"Failed to post comments to text channel: {e}")

    async def disconnect_voice(self) -> None:
        """ボイスチャンネルから切断"""
        if self.voice_client and self.voice_client.is_connected():
            await self.voice_client.disconnect()
            logger.info("Disconnected from voice channel")

    async def close(self) -> None:
        """Bot終了"""
        await self.disconnect_voice()
        await super().close()


async def start_discord_client(
    token: str,
    guild_id: int,
    voice_channel_id: int,
    text_channel_id: int,
    audio_buffer: AudioBuffer,
    ignored_user_ids: list[int],
) -> CommentBot:
    """
    Discordクライアントを起動

    Args:
        token: Discordボットトークン
        guild_id: サーバーID
        voice_channel_id: ボイスチャンネルID
        text_channel_id: テキストチャンネルID
        audio_buffer: 音声バッファ
        ignored_user_ids: 無視するユーザーIDのリスト

    Returns:
        起動したBotインスタンス
    """
    bot = CommentBot(
        guild_id=guild_id,
        voice_channel_id=voice_channel_id,
        text_channel_id=text_channel_id,
        audio_buffer=audio_buffer,
        ignored_user_ids=ignored_user_ids,
    )

    # 非同期でBotを起動
    asyncio.create_task(bot.start(token))

    # Botの準備が整うまで待機
    while not bot.is_ready:
        await asyncio.sleep(0.1)

    logger.info("Discord client is ready")
    return bot
