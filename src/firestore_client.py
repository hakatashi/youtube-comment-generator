"""Firestore クライアントモジュール"""

import logging
from datetime import datetime
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

    def save_comment(
        self,
        comment: str,
        prompt: str,
        transcription: str,
        user_transcriptions: list[str],
        timestamp: Optional[datetime] = None,
    ) -> str:
        """
        コメントをFirestoreに保存

        Args:
            comment: 生成されたコメント
            prompt: 使用したプロンプト
            transcription: 結合された文字起こしテキスト
            user_transcriptions: ユーザーごとの文字起こしリスト
            timestamp: タイムスタンプ（Noneの場合は現在時刻）

        Returns:
            保存されたドキュメントのID
        """
        if timestamp is None:
            timestamp = datetime.now()

        doc_data = {
            "comment": comment,
            "prompt": prompt,
            "transcription": transcription,
            "user_transcriptions": user_transcriptions,
            "created_at": timestamp,
        }

        doc_ref = self.db.collection("comments").add(doc_data)
        doc_id = doc_ref[1].id
        logger.info(f"Saved comment to Firestore: {doc_id}")
        return doc_id

    def save_comments_batch(
        self,
        comments: list[str],
        prompt: str,
        transcription: str,
        user_transcriptions: list[str],
        timestamp: Optional[datetime] = None,
    ) -> list[str]:
        """
        複数のコメントを一括でFirestoreに保存

        Args:
            comments: 生成されたコメントのリスト
            prompt: 使用したプロンプト
            transcription: 結合された文字起こしテキスト
            user_transcriptions: ユーザーごとの文字起こしリスト
            timestamp: タイムスタンプ（Noneの場合は現在時刻）

        Returns:
            保存されたドキュメントのIDリスト
        """
        if timestamp is None:
            timestamp = datetime.now()

        doc_ids = []
        batch = self.db.batch()

        for comment in comments:
            doc_ref = self.db.collection("comments").document()
            doc_data = {
                "comment": comment,
                "prompt": prompt,
                "transcription": transcription,
                "user_transcriptions": user_transcriptions,
                "created_at": timestamp,
            }
            batch.set(doc_ref, doc_data)
            doc_ids.append(doc_ref.id)

        batch.commit()
        logger.info(f"Saved {len(comments)} comments to Firestore")
        return doc_ids

    def get_recent_comments(self, limit: int = 50) -> list[dict]:
        """
        最新のコメントを取得

        Args:
            limit: 取得する件数

        Returns:
            コメントのリスト
        """
        docs = (
            self.db.collection("comments")
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )

        comments = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            comments.append(data)

        return comments
