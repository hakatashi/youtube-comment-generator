"""コメント生成モジュール - Qwen2.5-32B を使用"""

import json
import logging
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List

from llama_cpp import Llama

logger = logging.getLogger(__name__)


class CommentGenerator:
    """Qwen2.5-32B を使用したコメント生成クラス"""

    def __init__(
        self,
        model_path: Path,
        n_gpu_layers: int = -1,  # -1 = すべてのレイヤーをGPUに
        n_ctx: int = 4096,
        verbose: bool = False,
        comments_file: str = "comments.txt",
    ):
        """
        Args:
            model_path: GGUFモデルファイルのパス
            n_gpu_layers: GPUに載せるレイヤー数（-1で全レイヤー）
            n_ctx: コンテキスト長
            verbose: 詳細ログを出力するか
            comments_file: コメント出力ファイルのパス
        """
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.llm = None
        self.comments_file = Path(comments_file)

        # 直近30件のコメント履歴を保持
        self.comment_history: Deque[str] = deque(maxlen=30)

        logger.info(f"Initializing LLM: {model_path}")
        self._load_model()

    def _load_model(self) -> None:
        """モデルをロード"""
        try:
            self.llm = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=False,
            )
            logger.info("LLM loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise

    def generate_comments(self, transcription: str, num_comments: int = 10) -> List[str]:
        """
        文字起こしテキストからYouTube風コメントを生成

        Args:
            transcription: 文字起こしテキスト
            num_comments: 生成するコメント数

        Returns:
            生成されたコメントのリスト
        """
        if not transcription.strip():
            logger.warning("Empty transcription provided")
            return []

        prompt = self._create_prompt(transcription, num_comments)

        try:
            logger.info(f"Generating {num_comments} comments...")

            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "あなたは日本のVTuber配信のコメント欄で視聴者がよく書き込むようなコメントを生成するアシスタントです。",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
                temperature=0.9,
                top_p=0.95,
            )

            # レスポンスからコメントを抽出
            generated_text = response["choices"][0]["message"]["content"]
            comments = self._extract_comments(generated_text)

            logger.info(f"Generated {len(comments)} comments")

            # コメント履歴に追加
            for comment in comments[:num_comments]:
                self.comment_history.append(comment)

            # ファイルに保存
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
        prompt = f"""以下のテキストは、とあるVTuberのライブ配信の直近60秒の配信内容の文字起こしです。

この文字起こしを参考にして、この場面で投稿されていそうなYouTubeのコメントを{num_comments}件考えてください。

コメントの内容は以下のようなバリエーションを含めてください：
- 配信内容に直接関連するコメント（「草」「すごい」「かわいい」など）
- リアクション系（「www」「！？」「えぇ...」など）
- 応援系（「がんばれ！」「いいぞ！」など）
- 初見・挨拶系（「初見です」「こんにちは」など）
- 質問や相槌（「なるほど」「それな」など）

コメントは1行ずつ、番号なしで出力してください。各コメントは短く、自然な口語表現で書いてください。"""

        # 直近のコメント履歴がある場合は追加
        if self.comment_history:
            history_text = "\n".join(f"- {comment}" for comment in self.comment_history)
            prompt += f"""

**重要**: 以下は直近で生成されたコメントです。これらと類似した内容や表現を避け、バリエーションに富んだ新しいコメントを生成してください：

{history_text}"""

        prompt += f"""

## 直近60秒の配信内容の文字起こし

{transcription}

## YouTubeコメント（{num_comments}件）"""

        return prompt

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
            line = re.sub(r"^[\d\-\*\+]+[\.\)]\s*", "", line)
            line = re.sub(r"^[・•]\s*", "", line)

            # 引用符を削除
            line = line.strip('"\'「」『』')

            # コメントとして有効な文字列のみ追加
            if line and len(line) > 0:
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
            timestamp = datetime.now().isoformat()

            # ファイルに追記
            with open(self.comments_file, "a", encoding="utf-8") as f:
                f.write(f"# {timestamp}\n")
                for comment in comments:
                    f.write(f"{comment}\n")
                f.write("\n")  # セクション区切り

            logger.info(f"Saved {len(comments)} comments to {self.comments_file}")

        except Exception as e:
            logger.error(f"Failed to save comments to file: {e}")
