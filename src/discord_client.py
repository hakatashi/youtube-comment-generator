"""Discord クライアントモジュール"""

import asyncio
import logging
from typing import Optional

import discord
from discord import VoiceClient
from discord.ext import commands

from .audio_buffer import AudioBuffer

logger = logging.getLogger(__name__)


class VoiceReceiver(discord.VoiceClient):
    """音声受信用のカスタムVoiceClient"""

    def __init__(self, client: discord.Client, channel: discord.VoiceChannel):
        super().__init__(client, channel)
        self.audio_buffer: Optional[AudioBuffer] = None

    def set_audio_buffer(self, audio_buffer: AudioBuffer) -> None:
        """音声バッファを設定"""
        self.audio_buffer = audio_buffer


class CommentBot(commands.Bot):
    """YouTube風コメント生成Bot"""

    def __init__(
        self,
        guild_id: int,
        voice_channel_id: int,
        audio_buffer: AudioBuffer,
    ):
        """
        Args:
            guild_id: サーバーID
            voice_channel_id: ボイスチャンネルID
            audio_buffer: 音声バッファ
        """
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True

        super().__init__(command_prefix="!", intents=intents)

        self.guild_id = guild_id
        self.voice_channel_id = voice_channel_id
        self.audio_buffer = audio_buffer
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
            self.voice_client = await voice_channel.connect()

            # 音声受信を開始
            self.voice_client.start_recording(
                discord.sinks.WaveSink(),
                self._on_recording_finished,
                self._on_audio_packet,
            )

            logger.info("Started recording")
            self.is_ready = True

        except Exception as e:
            logger.error(f"Failed to connect to voice channel: {e}")

    def _on_audio_packet(self, user: discord.User, data: bytes) -> None:
        """
        音声パケット受信時のコールバック

        Args:
            user: 音声を送信したユーザー
            data: PCM音声データ
        """
        # 音声データをバッファに追加
        if self.audio_buffer is not None:
            self.audio_buffer.add_audio(data)

    async def _on_recording_finished(self, sink: discord.sinks.Sink) -> None:
        """
        録音終了時のコールバック（通常は呼ばれない）

        Args:
            sink: 音声シンク
        """
        logger.info("Recording finished")

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
    audio_buffer: AudioBuffer,
) -> CommentBot:
    """
    Discordクライアントを起動

    Args:
        token: Discordボットトークン
        guild_id: サーバーID
        voice_channel_id: ボイスチャンネルID
        audio_buffer: 音声バッファ

    Returns:
        起動したBotインスタンス
    """
    bot = CommentBot(
        guild_id=guild_id,
        voice_channel_id=voice_channel_id,
        audio_buffer=audio_buffer,
    )

    # 非同期でBotを起動
    asyncio.create_task(bot.start(token))

    # Botの準備が整うまで待機
    while not bot.is_ready:
        await asyncio.sleep(0.1)

    logger.info("Discord client is ready")
    return bot
