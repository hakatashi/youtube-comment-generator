"""コメント生成モジュール - Qwen3-32B を使用"""

import json
import logging
import re
import unicodedata
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, List

from jinja2 import Environment, FileSystemLoader
from llama_cpp import Llama

logger = logging.getLogger(__name__)

# TYPE_CHECKING用のインポート
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .firestore_client import FirestoreClient


class CommentGenerator:
    """Qwen3-32B を使用したコメント生成クラス"""

    def __init__(
        self,
        model_path: Path,
        n_gpu_layers: int = -1,  # -1 = すべてのレイヤーをGPUに
        n_ctx: int = 4096,
        verbose: bool = False,
        comments_file: str = "comments.txt",
        firestore_client: "FirestoreClient" = None,
    ):
        """
        Args:
            model_path: GGUFモデルファイルのパス
            n_gpu_layers: GPUに載せるレイヤー数（-1で全レイヤー）
            n_ctx: コンテキスト長
            verbose: 詳細ログを出力するか
            comments_file: コメント出力ファイルのパス
            firestore_client: Firestoreクライアント（Noneの場合はファイルに保存）
        """
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.llm = None
        self.comments_file = Path(comments_file)
        self.firestore_client = firestore_client

        # 直近30件のコメント履歴を保持
        self.comment_history: Deque[str] = deque(maxlen=30)

        # Jinja2環境のセットアップ
        template_dir = Path(__file__).parent / "templates"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        self.prompt_template = self.jinja_env.get_template("comment_prompt.jinja")

        logger.info(f"Initializing LLM: {model_path}")
        self._load_model()

    def _load_model(self) -> None:
        """モデルをロード"""
        try:
            logger.info(f"Loading LLM model with GPU layers: {self.n_gpu_layers}")
            self.llm = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=True,
            )
            logger.info("LLM loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise

    def generate_comments(
        self,
        transcription: str,
        num_comments: int = 100,
        user_transcriptions: list[str] = None,
        audio_duration: float = None,
        stt_duration: float = None,
        user_ids: list[int] = None,
    ) -> List[str]:
        """
        文字起こしテキストからYouTube風コメントを生成

        Args:
            transcription: 文字起こしテキスト
            num_comments: 生成するコメント数
            user_transcriptions: ユーザーごとの文字起こしリスト（Firestore保存用）
            audio_duration: 処理された音声の長さ（秒）
            stt_duration: 文字起こしにかかった時間（秒）
            user_ids: 文字起こしに使用したユーザーのID一覧

        Returns:
            生成されたコメントのリスト
        """
        if not transcription.strip():
            logger.warning("Empty transcription provided")
            return []

        prompt = self._create_prompt(transcription, num_comments)

        try:
            import time
            comment_gen_start = time.time()

            logger.info(f"Generating {num_comments} comments...")
            logger.info(f"Prompt: {prompt}")

            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "あなたは日本のVTuber配信のコメント欄で視聴者がよく書き込むようなコメントを生成するアシスタントです。 /no_think",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=128,
                temperature=1.1,
                top_p=0.95,
                top_k=100,
                repeat_penalty=1.1,
            )

            # レスポンスからコメントを抽出
            generated_text = response["choices"][0]["message"]["content"]
            output_tokens = response["usage"]["completion_tokens"]
            logger.info(f"LLM generated text: {generated_text}")
            logger.info(f"Output Tokens: {output_tokens}")

            comments = self._extract_comments(generated_text)

            logger.info(f"Generated {len(comments)} comments")

            # コメント生成にかかった時間（LLMのみ）
            comment_gen_duration = time.time() - comment_gen_start

            # コメント履歴に追加
            for comment in comments[:num_comments]:
                self.comment_history.append(comment)

            # Firestoreまたはファイルに保存
            if self.firestore_client:
                self._save_comments_to_firestore(
                    comments[:num_comments],
                    prompt,
                    transcription,
                    user_transcriptions or [],
                    audio_duration,
                    stt_duration,
                    comment_gen_duration,
                    user_ids or [],
                )
            else:
                self._save_comments_to_file(comments[:num_comments])

            return comments[:num_comments]

        except Exception as e:
            logger.error(f"Comment generation failed: {e}")
            return []

    def _create_prompt(self, transcription: str, num_comments: int) -> str:
        """
        プロンプトを作成

        Args:
            transcription: 文字起こしテキスト
            num_comments: 生成するコメント数

        Returns:
            プロンプト文字列
        """
        return self.prompt_template.render(
            transcription=transcription,
            num_comments=num_comments,
            comment_history=list(self.comment_history)[-10:] if self.comment_history else None,
        )

    def _has_at_least_one_letter(self, s: str) -> bool:
        """文字列にUnicodeのLetterカテゴリの文字が少なくとも1つ含まれているかチェック"""
        return any(unicodedata.category(c).startswith("L") for c in s)

    def _extract_comments(self, generated_text: str) -> List[str]:
        """
        生成されたテキストからコメントを抽出

        Args:
            generated_text: LLMが生成したテキスト

        Returns:
            コメントのリスト
        """
        # Qwen3の<think>ブロックを除去
        import re
        generated_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL)

        # 改行で分割
        lines = generated_text.strip().split("\n")

        comments = []
        for line in lines:
            line = line.strip()

            # 空行はスキップ
            if not line:
                continue

            # 番号付きリスト（1. や - など）を削除
            line = re.sub(r"^[\-\*\+]+\s*", "", line)
            line = re.sub(r"^\d+\.\s*", "", line)
            line = re.sub(r"^[・•]\s*", "", line)
            if re.match(r'^[「『].*[」』]$', line):
                line = line[1:-1].strip()

            # コメントとして有効な文字列のみ追加
            if line and len(line) > 0 and len(line) <= 30 and self._has_at_least_one_letter(line):
                comments.append(line)

        return comments

    def _save_comments_to_file(self, comments: List[str]) -> None:
        """
        生成されたコメントをタイムスタンプ付きでファイルに保存

        Args:
            comments: 保存するコメントのリスト
        """
        if not comments:
            return

        try:
            timestamp = datetime.now(timezone.utc).isoformat()

            # ファイルに追記
            with open(self.comments_file, "a", encoding="utf-8") as f:
                f.write(f"# {timestamp}\n")
                for comment in comments:
                    f.write(f"{comment}\n")
                f.write("\n")  # セクション区切り

            logger.info(f"Saved {len(comments)} comments to {self.comments_file}")

        except Exception as e:
            logger.error(f"Failed to save comments to file: {e}")

    def _save_comments_to_firestore(
        self,
        comments: List[str],
        prompt: str,
        transcription: str,
        user_transcriptions: list[str],
        audio_duration: float,
        stt_duration: float,
        comment_gen_duration: float,
        user_ids: list[int],
    ) -> None:
        """
        生成されたコメントをFirestoreに保存

        Args:
            comments: 保存するコメントのリスト
            prompt: 使用したプロンプト
            transcription: 結合された文字起こしテキスト
            user_transcriptions: ユーザーごとの文字起こしリスト
            audio_duration: 処理された音声の長さ（秒）
            stt_duration: 文字起こしにかかった時間（秒）
            comment_gen_duration: コメント生成にかかった時間（秒）
            user_ids: 文字起こしに使用したユーザーのID一覧
        """
        if not comments:
            return

        try:
            self.firestore_client.save_comments_batch(
                comments=comments,
                prompt=prompt,
                transcription=transcription,
                user_transcriptions=user_transcriptions,
                audio_duration=audio_duration,
                stt_duration=stt_duration,
                comment_gen_duration=comment_gen_duration,
                user_ids=user_ids,
            )
            logger.info(f"Saved {len(comments)} comments to Firestore")

        except Exception as e:
            logger.error(f"Failed to save comments to Firestore: {e}")

if __name__ == "__main__":
    # テストコード
    import os
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # モデルパス（適宜変更してください）
    model_path = Path("models/Qwen3-32B-Q5_K_M.gguf")

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        exit(1)

    # コメントジェネレーターの初期化
    comment_gen = CommentGenerator(model_path=model_path)

    # テスト用の文字起こしテキスト
    test_transcription = """
今日はみんなでマイクラをやっていきたいと思います！
えっと、まずは木を集めましょうか。
おっ、村を発見しました！すごい！
ダイヤ見つけた！やったー！
"""

    logger.info("Generating comments...")
    comments = comment_gen.generate_comments(test_transcription, num_comments=100)

    logger.info("Generated comments:")
    for i, comment in enumerate(comments, 1):
        logger.info(f"{i}. {comment}")