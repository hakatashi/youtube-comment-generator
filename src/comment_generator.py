"""コメント生成モジュール - Qwen2.5-32B を使用"""

import json
import logging
import re
from pathlib import Path
from typing import List

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
    ):
        """
        Args:
            model_path: GGUFモデルファイルのパス
            n_gpu_layers: GPUに載せるレイヤー数（-1で全レイヤー）
            n_ctx: コンテキスト長
            verbose: 詳細ログを出力するか
        """
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.llm = None

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

コメントは1行ずつ、番号なしで出力してください。各コメントは短く、自然な口語表現で書いてください。

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
