"""モデルダウンロードモジュール"""

import logging
from pathlib import Path

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Hugging Face Hub からモデルをダウンロードするクラス"""

    @staticmethod
    def download_gguf_model(
        repo_id: str,
        filename: str,
        local_dir: Path,
    ) -> Path:
        """
        GGUF形式のモデルファイルをダウンロード

        Args:
            repo_id: Hugging Face リポジトリID (例: "Qwen/Qwen2.5-32B-Instruct-GGUF")
            filename: ダウンロードするファイル名 (例: "qwen2.5-32b-instruct-q5_k_m.gguf")
            local_dir: ダウンロード先ディレクトリ

        Returns:
            ダウンロードされたファイルのパス
        """
        # ディレクトリが存在しない場合は作成
        local_dir.mkdir(parents=True, exist_ok=True)

        # ファイルが既に存在するかチェック
        local_file = local_dir / filename
        if local_file.exists():
            logger.info(f"Model file already exists: {local_file}")
            return local_file

        logger.info(f"Downloading model from {repo_id}/{filename}...")
        logger.info("This may take a while depending on your internet connection.")

        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )

            logger.info(f"Model downloaded successfully: {downloaded_path}")
            return Path(downloaded_path)

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

    @staticmethod
    def ensure_qwen_model(
        repo_id: str,
        filename: str,
        models_dir: Path,
    ) -> Path:
        """
        Qwen モデルが存在することを確認し、なければダウンロード

        Args:
            repo_id: Hugging Face リポジトリID
            filename: モデルファイル名
            models_dir: モデル保存ディレクトリ

        Returns:
            モデルファイルのパス
        """
        model_path = models_dir / filename

        if model_path.exists():
            logger.info(f"Qwen model found: {model_path}")
            return model_path

        logger.info("Qwen model not found. Starting download...")
        return ModelDownloader.download_gguf_model(
            repo_id=repo_id,
            filename=filename,
            local_dir=models_dir,
        )
