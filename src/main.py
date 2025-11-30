"""メインアプリケーション"""

import asyncio
import logging
import sys
from pathlib import Path

from .audio_buffer import AudioBuffer
from .comment_generator import CommentGenerator
from .config import AppConfig
from .discord_client import start_discord_client
from .speech_to_text import SpeechToText

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


class CommentGeneratorApp:
    """YouTube風コメント生成アプリケーション"""

    def __init__(self, config: AppConfig):
        """
        Args:
            config: アプリケーション設定
        """
        self.config = config
        self.audio_buffer: AudioBuffer
        self.stt: SpeechToText
        self.comment_gen: CommentGenerator
        self.discord_bot = None

    async def initialize(self) -> None:
        """初期化処理"""
        logger.info("Initializing application...")

        # 音声バッファの初期化
        self.audio_buffer = AudioBuffer(
            sample_rate=self.config.audio.sample_rate,
            channels=self.config.audio.channels,
            buffer_duration=self.config.audio.buffer_duration,
        )
        logger.info("Audio buffer initialized")

        # Speech-to-Text モデルの初期化
        logger.info("Loading Speech-to-Text model...")
        self.stt = SpeechToText(
            model_name=self.config.model.whisper_model,
            device=self.config.model.device,
        )
        logger.info("Speech-to-Text model loaded")

        # LLMの初期化
        logger.info("Loading LLM for comment generation...")
        model_path = (
            self.config.model.models_dir / self.config.model.qwen_model_file
        )

        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            logger.error(
                f"Please download the model and place it at {model_path}"
            )
            sys.exit(1)

        self.comment_gen = CommentGenerator(model_path=model_path)
        logger.info("LLM loaded")

        # Discordクライアントの起動
        logger.info("Starting Discord client...")
        self.discord_bot = await start_discord_client(
            token=self.config.discord.token,
            guild_id=self.config.discord.guild_id,
            voice_channel_id=self.config.discord.voice_channel_id,
            audio_buffer=self.audio_buffer,
        )
        logger.info("Discord client started")

        logger.info("Initialization complete!")

    async def process_audio_and_generate_comments(self) -> None:
        """音声処理とコメント生成を実行"""
        logger.info("Processing audio and generating comments...")

        # バッファから60秒分の音声を取得
        audio_data = self.audio_buffer.get_audio(duration=60)

        if len(audio_data) == 0:
            logger.warning("No audio data in buffer")
            return

        logger.info(f"Retrieved {len(audio_data)} audio samples")

        # Speech-to-Text で文字起こし
        logger.info("Transcribing audio...")
        transcription = self.stt.transcribe_from_array(
            audio_data, sample_rate=self.config.audio.sample_rate
        )

        if not transcription.strip():
            logger.warning("Transcription is empty")
            return

        logger.info(f"Transcription: {transcription}")

        # コメント生成
        logger.info("Generating comments...")
        comments = self.comment_gen.generate_comments(
            transcription, num_comments=10
        )

        # コメントを出力
        print("\n" + "=" * 60)
        print("生成されたコメント:")
        print("=" * 60)
        for i, comment in enumerate(comments, 1):
            print(f"{i:2d}. {comment}")
        print("=" * 60 + "\n")

    async def run(self) -> None:
        """アプリケーションのメインループ"""
        await self.initialize()

        logger.info(
            f"Starting main loop (interval: {self.config.audio.interval}s)..."
        )

        try:
            while True:
                # 指定間隔で処理を実行
                await asyncio.sleep(self.config.audio.interval)

                # バッファに十分なデータがあるかチェック
                if not self.audio_buffer.is_ready(required_duration=60):
                    logger.warning(
                        f"Insufficient audio data "
                        f"({self.audio_buffer.get_duration():.1f}s / 60s)"
                    )
                    continue

                # 音声処理とコメント生成
                await self.process_audio_and_generate_comments()

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            if self.discord_bot:
                await self.discord_bot.close()
            logger.info("Application stopped")


async def main() -> None:
    """エントリーポイント"""
    # 設定を読み込み
    config = AppConfig.from_env()

    # アプリケーションを起動
    app = CommentGeneratorApp(config)
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
