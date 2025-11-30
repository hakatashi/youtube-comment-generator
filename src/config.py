"""設定管理モジュール"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DiscordConfig:
    """Discord接続設定"""
    token: str
    guild_id: int
    voice_channel_id: int


@dataclass
class ModelConfig:
    """モデル設定"""
    whisper_model: str = "kotoba-tech/kotoba-whisper-v1.1"
    qwen_model: str = "Qwen/Qwen2.5-32B-Instruct-GGUF"
    qwen_model_file: str = "qwen2.5-32b-instruct-q5_k_m.gguf"
    device: str = "cuda"  # ROCm uses CUDA API
    models_dir: Path = Path("./models")


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
        required_vars = ["DISCORD_TOKEN", "DISCORD_GUILD_ID", "DISCORD_VOICE_CHANNEL_ID"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
            print("Please create a .env file or set these environment variables.")
            print("See .env.example for reference.")
            sys.exit(1)

        discord_config = DiscordConfig(
            token=os.getenv("DISCORD_TOKEN"),
            guild_id=int(os.getenv("DISCORD_GUILD_ID")),
            voice_channel_id=int(os.getenv("DISCORD_VOICE_CHANNEL_ID")),
        )

        model_config = ModelConfig(
            whisper_model=os.getenv("WHISPER_MODEL", "kotoba-tech/kotoba-whisper-v1.1"),
            qwen_model=os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-32B-Instruct-GGUF"),
            qwen_model_file=os.getenv("QWEN_MODEL_FILE", "qwen2.5-32b-instruct-q5_k_m.gguf"),
            device=os.getenv("DEVICE", "cuda"),
            models_dir=Path(os.getenv("MODELS_DIR", "./models")),
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
