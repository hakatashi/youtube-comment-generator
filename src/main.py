"""メインアプリケーション"""

import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from .audio_buffer import AudioBuffer
from .comment_generator import CommentGenerator
from .config import AppConfig
from .discord_client import start_discord_client
from .firestore_client import FirestoreClient
from .model_downloader import ModelDownloader
from .speech_to_text_faster import SpeechToTextFaster

# .envファイルを読み込み
load_dotenv()

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

# Discord Opusのログレベルを上げて警告を抑制
logging.getLogger("discord.opus").setLevel(logging.ERROR)
logging.getLogger("discord.voice_client").setLevel(logging.WARNING)

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
        self.stt: SpeechToTextFaster
        self.comment_gen: CommentGenerator
        self.discord_bot = None
        self.firestore_client: FirestoreClient = None
        # CPU/GPU集約的な処理用のスレッドプール
        self.executor = ThreadPoolExecutor(max_workers=2)

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

        # Firestoreクライアントの初期化
        try:
            logger.info("Initializing Firestore client...")
            self.firestore_client = FirestoreClient()
            logger.info("Firestore client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Firestore client: {e}")
            logger.warning("Comments will be saved to file instead")

        # LLMの初期化
        logger.info("Loading LLM for comment generation...")

        # モデルファイルが存在することを確認（なければダウンロード）
        model_path = ModelDownloader.ensure_qwen_model(
            repo_id=self.config.model.qwen_model,
            filename=self.config.model.qwen_model_file,
            models_dir=self.config.model.models_dir,
        )

        self.comment_gen = CommentGenerator(
            model_path=model_path,
            firestore_client=self.firestore_client,
        )
        logger.info("LLM loaded")

        # Speech-to-Text モデルの初期化 (faster-whisper: 5.6倍高速)
        logger.info("Loading Speech-to-Text model (faster-whisper)...")
        self.stt = SpeechToTextFaster(
            model_name="kotoba-tech/kotoba-whisper-v2.0-faster",
            device="cpu",  # ROCm環境ではCPUを使用 (CTranslate2はCUDA専用)
            compute_type="int8",  # int8量子化で高速化
            download_root="models/faster-whisper",
        )
        logger.info("Speech-to-Text model loaded (5.6x faster than before)")

        # Discordクライアントの起動
        logger.info("Starting Discord client...")
        if self.config.discord.ignored_user_ids:
            logger.info(f"Ignoring audio from users: {self.config.discord.ignored_user_ids}")
        self.discord_bot = await start_discord_client(
            token=self.config.discord.token,
            guild_id=self.config.discord.guild_id,
            voice_channel_id=self.config.discord.voice_channel_id,
            text_channel_id=self.config.discord.text_channel_id,
            audio_buffer=self.audio_buffer,
            ignored_user_ids=self.config.discord.ignored_user_ids,
        )
        logger.info("Discord client started")

        logger.info("Initialization complete!")

    async def process_user_audio(self, user_id: int) -> tuple[str, str]:
        """
        特定ユーザーの音声を処理して文字起こしを返す

        Args:
            user_id: ユーザーID

        Returns:
            (ユーザー名, 文字起こしテキスト) のタプル
        """
        logger.info(f"Processing audio for user {user_id}...")

        # バッファから60秒分の音声を取得
        audio_data = self.audio_buffer.get_audio(user_id, duration=60)

        if len(audio_data) == 0:
            logger.warning(f"User {user_id}: No audio data in buffer")
            return ("", "")

        logger.info(f"User {user_id}: Retrieved {len(audio_data)} audio samples")

        # デバッグ用にWAVファイルを保存
        from datetime import datetime
        from pathlib import Path

        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)

        debug_wav_path = debug_dir / f"debug_audio_user{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        wav_bytes = self.audio_buffer.get_wav_bytes(user_id, duration=60)
        with open(debug_wav_path, 'wb') as f:
            f.write(wav_bytes)
        logger.info(f"User {user_id}: Saved debug audio to {debug_wav_path}")

        # Speech-to-Text で文字起こし（別スレッドで実行）
        logger.info(f"User {user_id}: Transcribing audio...")
        loop = asyncio.get_event_loop()
        transcription = await loop.run_in_executor(
            self.executor,
            self.stt.transcribe_from_array,
            audio_data,
            self.config.audio.sample_rate
        )

        # 「ごめん」を除去（無音部分の誤認識対策）
        transcription = transcription.replace("ごめん", "").replace("視野", "").replace("児童", "").replace("はい", "").strip()

        if not transcription.strip():
            logger.warning(f"User {user_id}: Transcription is empty")
            return ("", "")

        user_name = self.audio_buffer.get_user_name(user_id)
        logger.info(f"User {user_id} ({user_name}): Transcription: {transcription}")

        return (user_name, transcription)

    async def process_audio_and_generate_comments(self) -> None:
        """全ユーザーの音声処理とコメント生成を実行"""
        logger.info("Processing audio and generating comments...")

        # バッファに音声が保存されているすべてのユーザーを取得
        user_ids = self.audio_buffer.get_user_ids()

        if not user_ids:
            logger.warning("No users have audio data in buffer")
            return

        logger.info(f"Found {len(user_ids)} user(s) with audio data: {user_ids}")

        # 各ユーザーの音声を文字起こし
        user_transcriptions = []
        for user_id in user_ids:
            # 十分なデータがあるユーザーのみ処理
            if not self.audio_buffer.is_ready(user_id, required_duration=60):
                duration = self.audio_buffer.get_duration(user_id)
                logger.warning(
                    f"User {user_id}: Insufficient audio data ({duration:.1f}s / 60s)"
                )
                continue

            user_name, transcription = await self.process_user_audio(user_id)
            if transcription:
                user_transcriptions.append(f"{user_name}: {transcription}")

        # すべてのユーザーの発言を結合
        if not user_transcriptions:
            logger.warning("No transcriptions available for comment generation")
            return

        combined_transcript = "\n".join(user_transcriptions)
        logger.info(f"Combined transcript:\n{combined_transcript}")

        # コメント生成（別スレッドで実行）
        logger.info("Generating comments from all users' speech...")
        loop = asyncio.get_event_loop()
        comments = await loop.run_in_executor(
            self.executor,
            self.comment_gen.generate_comments,
            combined_transcript,
            100,  # num_comments
            user_transcriptions  # user_transcriptions for Firestore
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

                # 音声処理とコメント生成（全ユーザー）
                try:
                    await self.process_audio_and_generate_comments()
                except Exception as e:
                    logger.error(f"Error during audio processing: {e}", exc_info=True)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except asyncio.CancelledError:
            logger.info("Task cancelled, shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
        finally:
            if self.discord_bot:
                try:
                    await self.discord_bot.close()
                except Exception as e:
                    logger.error(f"Error during Discord bot shutdown: {e}")

            # エグゼキューターをシャットダウン
            logger.info("Shutting down thread pool executor...")
            self.executor.shutdown(wait=True)

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
