"""VLMコメント生成モジュール - Qwen3-VL-30B を使用"""

import base64
import json
import logging
import os
import re
import subprocess
import tempfile
import time
import unicodedata
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Deque, List, Optional

import requests
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

# TYPE_CHECKING用のインポート
if TYPE_CHECKING:
    from .firestore_client import FirestoreClient


class VLMCommentGenerator:
    """Qwen3-VL-30B を使用したVLMコメント生成クラス"""

    def __init__(
        self,
        model_path: Path,
        mmproj_path: Path,
        llama_server_path: Path,
        server_host: str = "127.0.0.1",
        server_port: int = 8080,
        n_ctx: int = 8192,
        comments_file: str = "comments.txt",
        firestore_client: "FirestoreClient" = None,
        storage_bucket_name: str = None,
    ):
        """
        Args:
            model_path: GGUFモデルファイルのパス
            mmproj_path: mmprojファイルのパス（視覚エンコーダー）
            llama_server_path: llama-serverバイナリのパス
            server_host: llama-serverのホスト
            server_port: llama-serverのポート
            n_ctx: コンテキスト長
            comments_file: コメント出力ファイルのパス
            firestore_client: Firestoreクライアント（Noneの場合はファイルに保存）
            storage_bucket_name: Firebase Storage bucket名
        """
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.llama_server_path = llama_server_path
        self.server_host = server_host
        self.server_port = server_port
        self.server_url = f"http://{server_host}:{server_port}"
        self.n_ctx = n_ctx
        self.comments_file = Path(comments_file)
        self.firestore_client = firestore_client
        self.storage_bucket_name = storage_bucket_name

        # llama-serverプロセス
        self.server_process: Optional[subprocess.Popen] = None

        # 直近30件のコメント履歴を保持
        self.comment_history: Deque[str] = deque(maxlen=30)

        # Jinja2環境のセットアップ
        template_dir = Path(__file__).parent / "templates"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        self.prompt_template = self.jinja_env.get_template("vlm_comment_prompt.jinja")

        logger.info(f"Initializing VLM: model={model_path}, mmproj={mmproj_path}")

    def start_server(self) -> bool:
        """llama-serverを起動"""
        if self.server_process is not None:
            logger.warning("Server is already running")
            return True

        if not self.llama_server_path.exists():
            logger.error(f"llama-server not found at: {self.llama_server_path}")
            return False

        cmd = [
            str(self.llama_server_path),
            "-m", str(self.model_path),
            "--mmproj", str(self.mmproj_path),
            "--host", self.server_host,
            "--port", str(self.server_port),
            "-c", str(self.n_ctx),
            "-ngl", "99",                    # GPU layers for ROCm acceleration
            "--cont-batching",               # enable continuous batching
            "--slots",                       # enable slots monitoring endpoint
            "--metrics",                     # enable prometheus metrics
            "--threads-http", "4",           # HTTP processing threads
            "--timeout", "600",              # server timeout in seconds
            "--temp", "1.1",                 # default temperature
            "--top-p", "0.95",               # default top-p
            "--top-k", "100",                # default top-k
            "--repeat-penalty", "1.1",       # repeat penalty
        ]

        logger.info(f"Starting llama-server...")
        logger.info(f"Server will be available at: {self.server_url}")

        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace',
            )

            # サーバー出力をストリーム表示するスレッドを開始
            output_thread = Thread(target=self._stream_server_output, args=(self.server_process,), daemon=True)
            output_thread.start()

            # サーバーが起動するまで待機
            if not self._wait_for_server():
                logger.error("Server startup failed")
                self.stop_server()
                return False

            logger.info("llama-server started successfully")
            return True

        except Exception as e:
            logger.error(f"Error starting server: {e}")
            return False

    def stop_server(self):
        """llama-serverを停止"""
        if self.server_process is not None:
            logger.info("Stopping llama-server...")
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
            logger.info("llama-server stopped")

    def _stream_server_output(self, process: subprocess.Popen):
        """サーバー出力をストリーム"""
        while True:
            try:
                if process.poll() is not None:
                    break

                stdout_line = process.stdout.readline()
                if stdout_line:
                    logger.debug(f"[llama-server] {stdout_line.strip()}")

                stderr_line = process.stderr.readline()
                if stderr_line:
                    logger.debug(f"[llama-server ERR] {stderr_line.strip()}")

            except (UnicodeDecodeError, Exception):
                continue

    def _wait_for_server(self, timeout: int = 120) -> bool:
        """サーバーが起動するまで待機"""
        logger.info("Waiting for llama-server to start...")
        start_time = time.time()

        time.sleep(5)

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("llama-server is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass

            time.sleep(2)

        logger.error("Server failed to start within timeout period")
        return False

    def _get_latest_screenshots_from_storage(self, count: int = 3, max_age_minutes: int = 10) -> Optional[list[tuple[str, str]]]:
        """
        Firebase Storageの/screenshots/ディレクトリから最新の画像を複数取得

        Args:
            count: 取得する画像の数（デフォルト: 3）
            max_age_minutes: 画像の最大経過時間（分）、この時間より古い画像は除外（デフォルト: 10）

        Returns:
            [(ローカルファイルパス, Storageのパス), ...] のリスト、失敗時はNone
        """
        if not self.storage_bucket_name:
            logger.error("Storage bucket name not configured")
            return None

        try:
            from firebase_admin import storage
            from datetime import timedelta

            bucket = storage.bucket(self.storage_bucket_name)

            # /screenshots/ ディレクトリ内のすべてのファイルをリスト
            blobs = list(bucket.list_blobs(prefix='screenshots/'))

            if not blobs:
                logger.warning("No screenshots found in Storage /screenshots/ directory")
                return None

            # 現在時刻（UTC）
            now_utc = datetime.now(timezone.utc)
            cutoff_time = now_utc - timedelta(minutes=max_age_minutes)

            # ファイル名からタイムスタンプをパースして、max_age_minutes以内のものだけをフィルタ
            valid_blobs = []
            for blob in blobs:
                # ファイル名が discord_screenshot_20250101090000.png のような形式を想定
                match = re.search(r'discord_screenshot_(\d{14})\.png', blob.name)
                if match:
                    timestamp_str = match.group(1)
                    try:
                        # タイムスタンプをパース（UTC）
                        timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
                        # 10分以内のものだけを追加
                        if timestamp >= cutoff_time:
                            valid_blobs.append((blob, timestamp))
                    except ValueError:
                        logger.warning(f"Failed to parse timestamp from filename: {blob.name}")
                        continue

            if not valid_blobs:
                logger.warning(f"No screenshots found within the last {max_age_minutes} minutes")
                return None

            # タイムスタンプでソートして最新のものを取得
            valid_blobs.sort(key=lambda x: x[1], reverse=True)

            # 最新count件を取得（存在する数が少ない場合は全て）
            latest_blobs = [blob for blob, _ in valid_blobs[:min(count, len(valid_blobs))]]

            logger.info(f"Found {len(blobs)} screenshot(s) in Storage, {len(valid_blobs)} within last {max_age_minutes} minutes")
            logger.info(f"Using latest {len(latest_blobs)} screenshot(s): {[b.name for b in latest_blobs]}")

            # 一時ファイルにダウンロード
            results = []
            for blob in latest_blobs:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    blob.download_to_filename(tmp_file.name)
                    logger.debug(f"Downloaded image from Storage to: {tmp_file.name}")
                    results.append((tmp_file.name, blob.name))

            return results

        except Exception as e:
            logger.error(f"Failed to get latest screenshots from Storage: {e}")
            return None

    def _encode_image_to_base64(self, image_path: str) -> str:
        """画像をbase64エンコード"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _chat_with_images(self, image_paths: list[str], prompt: str, system_prompt: str = None) -> Optional[str]:
        """llama-server APIを使用して画像付き/テキストのみチャット

        Args:
            image_paths: 画像ファイルパスのリスト（空リストの場合はテキストのみ）
            prompt: プロンプトテキスト
            system_prompt: システムプロンプト
        """
        # 画像がある場合のみエンコード
        image_base64_list = []
        if image_paths:
            # 画像の存在チェック
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    logger.error(f"Image not found: {image_path}")
                    return None

            # 画像をbase64エンコード
            try:
                image_base64_list = [self._encode_image_to_base64(path) for path in image_paths]
            except Exception as e:
                logger.error(f"Error encoding images: {e}")
                return None

        # メッセージを準備
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # ユーザーメッセージにテキストを追加
        if image_base64_list:
            # 画像がある場合: content配列形式
            user_content = [{"type": "text", "text": prompt}]
            for image_base64 in image_base64_list:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                })
            messages.append({"role": "user", "content": user_content})
        else:
            # 画像がない場合: テキストのみ
            messages.append({"role": "user", "content": prompt})

        # リクエストデータを準備
        request_data = {
            "messages": messages,
            "max_tokens": 128,
            "temperature": 1.1,
            "top_p": 0.95,
            "top_k": 100,
            "stream": False,  # ストリーミングを無効化
        }

        try:
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=600,
            )

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    return content.strip()
                else:
                    logger.error(f"Unexpected response format: {result}")
                    return None
            else:
                logger.error(f"Error: Server returned status {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error communicating with server: {e}", exc_info=True)
            return None

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
        文字起こしテキストと画像からYouTube風コメントを生成

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

        # Firebase Storageから最新の画像を複数取得
        results = self._get_latest_screenshots_from_storage(count=3)

        # 画像がない場合は空のリストを使用（テキストのみモード）
        if not results:
            logger.warning("No recent screenshots found, generating comments from transcription only")
            image_paths = []
            storage_paths = []
        else:
            image_paths = [result[0] for result in results]
            storage_paths = [result[1] for result in results]

        try:
            import time as time_module
            comment_gen_start = time_module.time()

            # プロンプトを作成
            prompt = self._create_prompt(transcription, num_comments)

            logger.info(f"Generating {num_comments} comments with VLM...")
            if storage_paths:
                logger.info(f"Using {len(storage_paths)} screenshot(s): {storage_paths}")
            else:
                logger.info("No screenshots available, using transcription only")
            logger.debug(f"Prompt: {prompt}")

            system_prompt = "あなたは日本のVTuber配信のコメント欄で視聴者がよく書き込むようなコメントを生成するアシスタントです。"

            # VLM/LLMで生成
            generated_text = self._chat_with_images(
                image_paths=image_paths,
                prompt=prompt,
                system_prompt=system_prompt
            )

            if not generated_text:
                logger.error("Failed to generate comments from VLM")
                return []

            logger.info(f"VLM generated text: {generated_text}")

            # コメントを抽出
            comments = self._extract_comments(generated_text)
            logger.info(f"Generated {len(comments)} comments")

            # コメント生成にかかった時間（VLMのみ）
            comment_gen_duration = time_module.time() - comment_gen_start

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
                    storage_paths,  # 使用した画像パスを配列として渡す
                )
            else:
                self._save_comments_to_file(comments[:num_comments])

            return comments[:num_comments]

        except Exception as e:
            logger.error(f"Comment generation failed: {e}", exc_info=True)
            return []

        finally:
            # 一時画像ファイルを削除
            for image_path in image_paths:
                if image_path and os.path.exists(image_path):
                    try:
                        os.unlink(image_path)
                        logger.debug(f"Cleaned up temporary image file: {image_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up temporary file: {e}")

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
            generated_text: VLMが生成したテキスト

        Returns:
            コメントのリスト
        """
        # Qwen3の<think>ブロックを除去
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
        image_paths: list[str],
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
            image_paths: 使用した画像のStorageパスのリスト
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
                image_paths=image_paths,
            )
            logger.info(f"Saved {len(comments)} comments to Firestore with {len(image_paths)} image reference(s)")

        except Exception as e:
            logger.error(f"Failed to save comments to Firestore: {e}")

    def __enter__(self):
        """コンテキストマネージャー: 入口"""
        if not self.start_server():
            raise RuntimeError("Failed to start llama-server")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー: 出口"""
        self.stop_server()
        return False
