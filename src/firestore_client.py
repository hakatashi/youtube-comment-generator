"""Firestore クライアントモジュール"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import firebase_admin
from firebase_admin import credentials, firestore

logger = logging.getLogger(__name__)


class FirestoreClient:
    """Firestoreクライアント"""

    def __init__(self, credential_path: Optional[str] = None):
        """
        Args:
            credential_path: Firebase Admin SDKの認証鍵のパス。
                           Noneの場合はデフォルトの場所を探す。
        """
        if not firebase_admin._apps:
            if credential_path is None:
                # デフォルトの場所を探す
                credential_path = self._find_credential_file()

            if credential_path and Path(credential_path).exists():
                cred = credentials.Certificate(credential_path)
                firebase_admin.initialize_app(cred)
                logger.info(f"Initialized Firebase Admin SDK with credential: {credential_path}")
            else:
                raise FileNotFoundError(
                    f"Firebase Admin SDK credential file not found: {credential_path}"
                )

        self.db = firestore.client()

    def _find_credential_file(self) -> Optional[str]:
        """
        Firebase Admin SDKの認証鍵ファイルを探す

        Returns:
            認証鍵ファイルのパス、見つからない場合はNone
        """
        # カレントディレクトリで *-firebase-adminsdk-*.json を探す
        for path in Path(".").glob("*-firebase-adminsdk-*.json"):
            return str(path)
        return None

    def save_comments_batch(
        self,
        comments: list[str],
        prompt: str,
        transcription: str,
        user_transcriptions: list[str],
        timestamp: Optional[datetime] = None,
        audio_duration: Optional[float] = None,
        stt_duration: Optional[float] = None,
        comment_gen_duration: Optional[float] = None,
        user_ids: Optional[list[int]] = None,
    ) -> str:
        """
        複数のコメントを一括でFirestoreに保存
        バッチメタデータを /batches/{batchId} に保存し、
        各コメントをサブコレクション /batches/{batchId}/comments/{commentId} に保存

        Args:
            comments: 生成されたコメントのリスト
            prompt: 使用したプロンプト
            transcription: 結合された文字起こしテキスト
            user_transcriptions: ユーザーごとの文字起こしリスト
            timestamp: タイムスタンプ（Noneの場合は現在時刻）
            audio_duration: 処理された音声の長さ（秒）
            stt_duration: 文字起こしにかかった時間（秒）
            comment_gen_duration: コメント生成にかかった時間（秒）
            user_ids: 文字起こしに使用したユーザーのID一覧

        Returns:
            バッチのドキュメントID
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # バッチメタデータを作成
        batch_ref = self.db.collection("batches").document()
        batch_data = {
            "created_at": timestamp,
            "count": len(comments),
            "prompt": prompt,
            "transcription": transcription,
            "user_transcriptions": user_transcriptions,
        }

        # 追加のメタデータがあれば含める
        if audio_duration is not None:
            batch_data["audio_duration"] = audio_duration
        if stt_duration is not None:
            batch_data["stt_duration"] = stt_duration
        if comment_gen_duration is not None:
            batch_data["comment_gen_duration"] = comment_gen_duration
        if user_ids is not None:
            batch_data["user_ids"] = user_ids

        # 合計時間を計算
        if stt_duration is not None and comment_gen_duration is not None:
            batch_data["total_duration"] = stt_duration + comment_gen_duration

        # バッチ操作を開始
        batch = self.db.batch()
        batch.set(batch_ref, batch_data)

        # 各コメントをサブコレクションに保存（生成順序を保持）
        for index, comment in enumerate(comments):
            comment_ref = batch_ref.collection("comments").document()
            comment_data = {
                "comment": comment,
                "created_at": timestamp,
                "index": index,  # 生成順序を保持
            }
            batch.set(comment_ref, comment_data)

        batch.commit()
        logger.info(f"Saved batch {batch_ref.id} with {len(comments)} comments to Firestore")
        return batch_ref.id

    def get_recent_batches(self, limit: int = 10) -> list[dict]:
        """
        最新のバッチを取得

        Args:
            limit: 取得するバッチ件数

        Returns:
            バッチのリスト
        """
        docs = (
            self.db.collection("batches")
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )

        batches = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            batches.append(data)

        return batches
