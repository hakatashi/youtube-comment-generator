"""メインアプリケーション"""

import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from .audio_buffer import AudioBuffer
from .comment_generator import CommentGenerator
from .vlm_comment_generator import VLMCommentGenerator
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
        if self.config.model.use_vlm:
            logger.info("Loading VLM for comment generation...")

            # VLMモデルファイルが存在することを確認（なければダウンロード）
            vlm_model_path = ModelDownloader.ensure_qwen_model(
                repo_id=self.config.model.vlm_model,
                filename=self.config.model.vlm_model_file,
                models_dir=self.config.model.models_dir,
            )

            vlm_mmproj_path = ModelDownloader.ensure_qwen_model(
                repo_id=self.config.model.vlm_model,
                filename=self.config.model.vlm_mmproj_file,
                models_dir=self.config.model.models_dir,
            )

            self.comment_gen = VLMCommentGenerator(
                model_path=vlm_model_path,
                mmproj_path=vlm_mmproj_path,
                llama_server_path=self.config.model.llama_server_path,
                firestore_client=self.firestore_client,
                storage_bucket_name=self.config.model.storage_bucket_name,
            )
            # VLMサーバーを起動
            if not self.comment_gen.start_server():
                raise RuntimeError("Failed to start VLM server")
            logger.info("VLM loaded and server started")
        else:
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

    async def process_audio_and_generate_comments(self) -> None:
        """全ユーザーの音声処理とコメント生成を実行（最適化版: 単一STT呼び出し）"""
        import time

        start_time = time.time()
        logger.info("Processing audio and generating comments...")

        # バッファに音声が保存されているすべてのユーザーを取得
        user_ids = self.audio_buffer.get_user_ids()

        if not user_ids:
            logger.warning("No users have audio data in buffer")
            return

        logger.info(f"Found {len(user_ids)} user(s) with audio data: {user_ids}")

        # 十分なデータがあるユーザーのみをフィルタリング
        ready_user_ids = []
        for user_id in user_ids:
            if self.audio_buffer.is_ready(user_id, required_duration=60):
                ready_user_ids.append(user_id)
            else:
                duration = self.audio_buffer.get_duration(user_id)
                logger.warning(
                    f"User {user_id}: Insufficient audio data ({duration:.1f}s / 60s)"
                )

        if not ready_user_ids:
            logger.warning("No users have sufficient audio data")
            return

        logger.info(f"Processing {len(ready_user_ids)} user(s) with sufficient audio data")

        # 複数ユーザーの音声を結合して取得（話者情報は不要）
        merged_audio, _ = self.audio_buffer.get_merged_audio(
            user_ids=ready_user_ids,
            duration=60
        )

        if len(merged_audio) == 0:
            logger.warning("No audio data in merged buffer")
            return

        logger.info(f"Merged audio: {len(merged_audio)} samples")

        # 無音除去を適用（デバッグ出力とSTT処理で共用）
        from .speech_to_text_faster import remove_silence
        import numpy as np
        from scipy import signal

        # int16 -> float32に変換して正規化
        audio_float = merged_audio.astype(np.float32) / 32768.0

        # リサンプリング（16kHz）
        target_sample_rate = 16000
        if self.config.audio.sample_rate != target_sample_rate:
            logger.info(f"Resampling audio from {self.config.audio.sample_rate} Hz to {target_sample_rate} Hz")
            num_samples = int(len(audio_float) * target_sample_rate / self.config.audio.sample_rate)
            audio_float = signal.resample(audio_float, num_samples)
            audio_sample_rate = target_sample_rate
        else:
            audio_sample_rate = self.config.audio.sample_rate

        # 無音除去
        audio_processed = remove_silence(
            audio_float,
            audio_sample_rate,
            silence_threshold_db=-45.0,
            min_silence_duration=0.5,
        )

        if len(audio_processed) == 0:
            logger.warning("No audio data after silence removal")
            return

        # デバッグ用にWAVファイルを保存（無音除去後）
        from datetime import datetime
        from pathlib import Path
        import soundfile as sf

        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)

        debug_wav_path = debug_dir / f"debug_audio_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        sf.write(debug_wav_path, audio_processed, audio_sample_rate)
        logger.info(f"Saved merged debug audio (after silence removal) to {debug_wav_path}")

        # 音声の長さを計算（秒）
        audio_duration = len(audio_processed) / audio_sample_rate

        # Speech-to-Text で文字起こし（無音除去は既に適用済みなので無効化）
        logger.info("Transcribing merged audio...")

        # 処理済み音声をint16に戻す
        audio_int16 = (audio_processed * 32768.0).astype(np.int16)

        loop = asyncio.get_event_loop()
        stt_start_time = time.time()
        transcription = await loop.run_in_executor(
            self.executor,
            self.stt.transcribe_from_array,
            audio_int16,
            audio_sample_rate,
            False  # remove_silence_enabled=False（既に除去済み）
        )
        stt_duration = time.time() - stt_start_time

        # 文字起こし結果のチェック
        if not transcription or not transcription.strip():
            logger.warning("No transcription available for comment generation")
            return

        logger.info(f"Transcription: {transcription}")
        logger.info(f"STT processing time: {stt_duration:.2f}s")

        # コメント生成（別スレッドで実行）
        logger.info("Generating comments from transcription...")
        comment_gen_start_time = time.time()
        comments = await loop.run_in_executor(
            self.executor,
            self.comment_gen.generate_comments,
            transcription,
            100,  # num_comments
            None,  # user_transcriptions は不要
            # 追加メタデータ
            audio_duration,
            stt_duration,
            ready_user_ids,
        )
        comment_gen_duration = time.time() - comment_gen_start_time

        # 合計時間
        total_duration = time.time() - start_time

        logger.info(f"Comment generation time: {comment_gen_duration:.2f}s")
        logger.info(f"Total processing time: {total_duration:.2f}s")
        logger.info(f"Audio duration: {audio_duration:.2f}s")
        logger.info(f"Processed {len(ready_user_ids)} user(s): {ready_user_ids}")

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

            # VLMサーバーを停止
            if isinstance(self.comment_gen, VLMCommentGenerator):
                try:
                    self.comment_gen.stop_server()
                except Exception as e:
                    logger.error(f"Error during VLM server shutdown: {e}")

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
