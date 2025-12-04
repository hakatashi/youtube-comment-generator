"""設定管理モジュール"""

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class DiscordConfig:
    """Discord接続設定"""
    token: str
    guild_id: int
    voice_channel_id: int
    text_channel_id: int
    ignored_user_ids: List[int]


@dataclass
class ModelConfig:
    """モデル設定"""
    whisper_model: str = "kotoba-tech/kotoba-whisper-v1.1"
    qwen_model: str = "Qwen/Qwen3-32B-GGUF"
    qwen_model_file: str = "Qwen3-32B-Q5_K_M.gguf"
    device: str = "cuda"  # ROCm uses CUDA API
    models_dir: Path = Path("./models")
    # VLM settings
    use_vlm: bool = False  # VLMを使用するかどうか
    vlm_model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF"
    vlm_model_file: str = "Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf"
    vlm_mmproj_file: str = "mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf"
    llama_server_path: Path = Path.home() / "Documents/GitHub/llama.cpp/build/bin/llama-server"
    storage_bucket_name: str = "vtuber-comment-generator.firebasestorage.app"
    vlm_n_ctx: int = 16384  # VLMのコンテキスト長（デフォルト: 16384）


@dataclass
class AudioConfig:
    """音声処理設定"""
    sample_rate: int = 48000  # Discord voice sample rate
    channels: int = 2  # Stereo
    buffer_duration: int = 60  # seconds
    interval: int = 30  # seconds


@dataclass
class AppConfig:
    """アプリケーション全体の設定"""
    discord: DiscordConfig
    model: ModelConfig
    audio: AudioConfig

    @classmethod
    def from_env(cls) -> "AppConfig":
        """環境変数から設定を読み込み"""
        import os
        import sys

        # 必須の環境変数をチェック
        required_vars = ["DISCORD_TOKEN", "DISCORD_GUILD_ID", "DISCORD_VOICE_CHANNEL_ID", "DISCORD_TEXT_CHANNEL_ID"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
            print("Please create a .env file or set these environment variables.")
            print("See .env.example for reference.")
            sys.exit(1)

        # 無視するユーザーIDのリストを読み込み
        ignored_user_ids = []
        ignored_ids_str = os.getenv("DISCORD_IGNORED_USER_IDS", "")
        if ignored_ids_str.strip():
            try:
                ignored_user_ids = [int(uid.strip()) for uid in ignored_ids_str.split(",") if uid.strip()]
            except ValueError:
                print("Warning: Invalid DISCORD_IGNORED_USER_IDS format. Expected comma-separated integers.")

        discord_config = DiscordConfig(
            token=os.getenv("DISCORD_TOKEN"),
            guild_id=int(os.getenv("DISCORD_GUILD_ID")),
            voice_channel_id=int(os.getenv("DISCORD_VOICE_CHANNEL_ID")),
            text_channel_id=int(os.getenv("DISCORD_TEXT_CHANNEL_ID")),
            ignored_user_ids=ignored_user_ids,
        )

        model_config = ModelConfig(
            whisper_model=os.getenv("WHISPER_MODEL", "kotoba-tech/kotoba-whisper-v1.1"),
            qwen_model=os.getenv("QWEN_MODEL", "Qwen/Qwen3-32B-GGUF"),
            qwen_model_file=os.getenv("QWEN_MODEL_FILE", "Qwen3-32B-Q5_K_M.gguf"),
            device=os.getenv("DEVICE", "cuda"),
            models_dir=Path(os.getenv("MODELS_DIR", "./models")),
            # VLM settings
            use_vlm=os.getenv("USE_VLM", "false").lower() == "true",
            vlm_model=os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF"),
            vlm_model_file=os.getenv("VLM_MODEL_FILE", "Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf"),
            vlm_mmproj_file=os.getenv("VLM_MMPROJ_FILE", "mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf"),
            llama_server_path=Path(os.getenv("LLAMA_SERVER_PATH", str(Path.home() / "Documents/GitHub/llama.cpp/build/bin/llama-server"))),
            storage_bucket_name=os.getenv("STORAGE_BUCKET_NAME", "vtuber-comment-generator.firebasestorage.app"),
            vlm_n_ctx=int(os.getenv("VLM_N_CTX", "16384")),
        )

        audio_config = AudioConfig(
            sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "48000")),
            channels=int(os.getenv("AUDIO_CHANNELS", "2")),
            buffer_duration=int(os.getenv("AUDIO_BUFFER_DURATION", "60")),
            interval=int(os.getenv("AUDIO_INTERVAL", "30")),
        )

        return cls(
            discord=discord_config,
            model=model_config,
            audio=audio_config,
        )
