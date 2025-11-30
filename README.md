# YouTube Comment Generator

Discord のボイスチャンネルから音声をリアルタイムで取得し、ローカルVLMを使用してYouTube風のコメントを自動生成する常駐アプリケーションです。

## 機能

- Discordの特定のボイスチャンネルに接続し、音声をトラックごとに分離
- 30秒ごとに以下の処理を実行:
  1. 直近60秒分の音声をSpeech-to-Textモデル（Kotoba-Whisper）でテキスト化
  2. テキストを元にLLM（Qwen2.5-32B）でYouTube風コメントを10件生成
  3. 生成したコメントをコンソールに出力

## システム要件

- Python 3.10 以上
- ROCm 6.1.5 以上（AMD GPU使用時）
- Radeon Instinct MI50 または同等のGPU（VRAM 32GB推奨）
- Poetry（Pythonパッケージ管理）
- ffmpeg（音声処理用）

## 使用モデル

- **Speech-to-Text**: Kotoba-Whisper v1.1 (約3GB VRAM)
- **LLM**: Qwen2.5-32B-Instruct (Q5_K_M量子化, 約24GB VRAM)

合計で約27GBのVRAMを使用します。

## セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/your-username/youtube-comment-generator.git
cd youtube-comment-generator
```

### 2. 依存関係のインストール

```bash
# Poetry のインストール（未インストールの場合）
curl -sSL https://install.python-poetry.org | python3 -

# プロジェクトの依存関係をインストール
poetry install

# llama-cpp-python を ROCm 対応でインストール（重要！）
poetry run pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DGGML_HIP=on -DCMAKE_PREFIX_PATH=/opt/rocm-6.1.5 -DROCM_PATH=/opt/rocm-6.1.5" \
  poetry run pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir --verbose
```

### 3. Discord設定

デフォルトでは `src/config.py` にハードコードされた設定が使用されますが、環境変数で上書き可能です。

環境変数を使用する場合は `.env` ファイルを作成してください:

```bash
# .env ファイル（オプション）
DISCORD_TOKEN=your_discord_bot_token
DISCORD_GUILD_ID=your_guild_id
DISCORD_VOICE_CHANNEL_ID=your_voice_channel_id
```

## 実行

```bash
# Poetry環境で実行
poetry run python -m src.main

# または Poetry shell に入ってから実行
poetry shell
python -m src.main
```

**注意**: 初回実行時は以下のモデルが自動的にダウンロードされます：
- **Qwen2.5-32B-Instruct** (約20GB) - Hugging Face Hub からダウンロード
- **Kotoba-Whisper v1.1** (約3GB) - Hugging Face Hub からダウンロード

ダウンロードには時間がかかる場合があります。`models/` ディレクトリに保存されます。

## 設定オプション

環境変数で以下の設定をカスタマイズできます:

| 環境変数 | デフォルト値 | 説明 |
|---------|------------|------|
| `DISCORD_TOKEN` | (ハードコード値) | Discordボットトークン |
| `DISCORD_GUILD_ID` | **必須** | サーバーID |
| `DISCORD_VOICE_CHANNEL_ID` | **必須** | ボイスチャンネルID |
| `WHISPER_MODEL` | kotoba-tech/kotoba-whisper-v1.1 | STTモデル名 |
| `QWEN_MODEL` | Qwen/Qwen2.5-32B-Instruct-GGUF | LLMモデル名 |
| `QWEN_MODEL_FILE` | qwen2.5-32b-instruct-q5_k_m.gguf | GGUFファイル名 |
| `DEVICE` | cuda | 使用デバイス |
| `MODELS_DIR` | ./models | モデル保存ディレクトリ |
| `AUDIO_SAMPLE_RATE` | 48000 | サンプリングレート (Hz) |
| `AUDIO_BUFFER_DURATION` | 60 | バッファ保持時間 (秒) |
| `AUDIO_INTERVAL` | 30 | 処理実行間隔 (秒) |

## プロジェクト構造

```
youtube-comment-generator/
├── src/
│   ├── __init__.py
│   ├── main.py              # エントリーポイント
│   ├── config.py            # 設定管理
│   ├── model_downloader.py  # モデル自動ダウンロード
│   ├── discord_client.py    # Discord接続・音声受信
│   ├── audio_buffer.py      # 音声バッファ管理
│   ├── speech_to_text.py    # Speech-to-Text (Kotoba-Whisper)
│   └── comment_generator.py # コメント生成 (Qwen2.5-32B)
├── models/                  # モデルファイル（.gitignore）
├── pyproject.toml           # Poetry設定
└── README.md
```

## トラブルシューティング

### ROCm関連のエラー

```bash
# ROCm のバージョン確認
rocm-smi

# llama-cpp-python を再インストール
poetry run pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DGGML_HIP=on -DCMAKE_PREFIX_PATH=/opt/rocm-6.1.5 -DROCM_PATH=/opt/rocm-6.1.5" \
  poetry run pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir --verbose
```

### VRAM不足エラー

使用するモデルサイズを小さくしてください:

- Qwen2.5-14B-Instruct (約14GB)
- Qwen2.5-7B-Instruct (約7GB)

### Discord接続エラー

- ボットトークンが正しいか確認
- ボットが対象サーバーに招待されているか確認
- ボットに適切な権限（Voice Channelへの接続権限）があるか確認

### 音声が取得できない

- py-cord の音声受信機能は実験的機能です
- ffmpeg がインストールされているか確認: `ffmpeg -version`

## ライセンス

MIT License

## 注意事項

- このアプリケーションはDiscordの音声を録音します。プライバシーに配慮して使用してください
- 大規模言語モデルの生成内容は必ずしも適切とは限りません
- ローカルで動作するため、インターネット接続は不要です（モデルダウンロード後）