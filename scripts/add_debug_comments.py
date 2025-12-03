#!/usr/bin/env python3
"""Firestoreのbatchesコレクションにデバッグ用データを追加するスクリプト"""

import sys
from datetime import datetime, timedelta, timezone

from src.firestore_client import FirestoreClient


def main():
    """デバッグ用コメントバッチをFirestoreに追加"""
    print("Initializing Firestore client...")
    client = FirestoreClient()

    # デバッグ用のサンプルデータ
    debug_comments = [
        "草",
        "待ってました！今日も配信ありがとう！",
        "そのアイテム強いですよね",
        "www かわいい",
        "その後ろ危ないですよ！",
        "歌上手すぎる...",
        "初見です！登録しました！",
        "そのネタ好きwww",
        "おつかれさまでした～！",
        "次の配信も楽しみにしてます！",
    ]

    prompt = "配信の内容に基づいて、視聴者のコメントを生成してください。"
    transcription = "今日は新しいゲームを始めます！楽しみですね～ みなさんこんにちは！今日もよろしくお願いします。"
    user_transcriptions = [
        "今日は新しいゲームを始めます！",
        "楽しみですね～",
        "みなさんこんにちは！",
        "今日もよろしくお願いします。",
    ]

    timestamp = datetime.now(timezone.utc)

    # デバッグ用のメタデータ
    audio_duration = 15.5  # 15.5秒の音声
    stt_duration = 1.8  # STTに1.8秒
    comment_gen_duration = 3.2  # コメント生成に3.2秒
    user_ids = [123456789, 987654321]  # 2人のユーザー

    print(f"\nAdding batch with {len(debug_comments)} debug comments to Firestore...")
    batch_id = client.save_comments_batch(
        comments=debug_comments,
        prompt=prompt,
        transcription=transcription,
        user_transcriptions=user_transcriptions,
        timestamp=timestamp,
        audio_duration=audio_duration,
        stt_duration=stt_duration,
        comment_gen_duration=comment_gen_duration,
        user_ids=user_ids,
    )

    print(f"\n✓ Successfully added batch with {len(debug_comments)} comments!")
    print(f"  Batch ID: {batch_id}")
    print(f"  Comments:")
    for i, comment in enumerate(debug_comments):
        print(f"    [{i+1}/{len(debug_comments)}] \"{comment}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
