# YouTube Comment Generator

Discord のボイスチャンネルから音声をリアルタイムで取得し、ローカルLLMを使用してYouTube風のコメントを自動生成する常駐アプリケーションです。

## 機能

- Discordの特定のボイスチャンネルに接続し、音声をユーザーごとに分離して記録
- 30秒ごとに以下の処理を各ユーザーごとに実行:
  1. 直近60秒分の音声をSpeech-to-Textモデル（Kotoba-Whisper）でテキスト化
  2. テキストを元にLLM（Qwen3-32B）でYouTube風コメントを10件生成
  3. 生成したコメントをコンソールに出力
- BOTユーザーなど、特定のユーザーの音声を無視する機能

## システム要件

- Python 3.10 以上
- ROCm 6.1.5 以上（AMD GPU使用時）
- Radeon Instinct MI50 または同等のGPU（VRAM 32GB推奨）
- Poetry（Pythonパッケージ管理）
- ffmpeg（音声処理用）

## 使用モデル

- **Speech-to-Text**: Kotoba-Whisper v1.1 (約3GB VRAM)
- **LLM**: Qwen3-32B (Q5_K_M量子化, 約24GB VRAM)

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

# PyTorch を ROCm 対応でインストール（重要！）
poetry run pip install torch --index-url https://download.pytorch.org/whl/rocm6.1

# llama-cpp-python を ROCm 対応でインストール（重要！）
CMAKE_ARGS="-DGGML_HIP=on -DCMAKE_PREFIX_PATH=/opt/rocm-6.1.5 -DROCM_PATH=/opt/rocm-6.1.5" \
  poetry run pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir --verbose
```

### 3. 環境変数の設定

`.env.example` をコピーして `.env` ファイルを作成し、Discord接続情報を設定してください:

```bash
# .env.example をコピー
cp .env.example .env

# .env ファイルを編集
# DISCORD_TOKEN, DISCORD_GUILD_ID, DISCORD_VOICE_CHANNEL_ID を設定
```

`.env` ファイルの例:

```bash
# Discord接続設定（必須）
DISCORD_TOKEN=your_discord_bot_token_here
DISCORD_GUILD_ID=your_guild_id_here
DISCORD_VOICE_CHANNEL_ID=your_voice_channel_id_here

# 無視するユーザーIDのリスト（カンマ区切り、オプション）
DISCORD_IGNORED_USER_IDS=123456789012345678,123456789012345679

# モデル設定（オプション）
# WHISPER_MODEL=kotoba-tech/kotoba-whisper-v1.1
# QWEN_MODEL=Qwen/Qwen3-32B-GGUF
# QWEN_MODEL_FILE=Qwen3-32B-Q5_K_M.gguf
```

**注意**: `.env` ファイルには秘密情報が含まれるため、Gitにコミットしないでください（既に `.gitignore` に含まれています）。

## 実行

```bash
# Poetry環境で実行
poetry run python -m src.main

# または Poetry shell に入ってから実行
poetry shell
python -m src.main
```

**注意**: 初回実行時は以下のモデルが自動的にダウンロードされます：
- **Qwen3-32B** (約20GB) - Hugging Face Hub からダウンロード
- **Kotoba-Whisper v1.1** (約3GB) - Hugging Face Hub からダウンロード

ダウンロードには時間がかかる場合があります。`models/` ディレクトリに保存されます。

## 設定オプション

環境変数で以下の設定をカスタマイズできます:

| 環境変数 | デフォルト値 | 説明 |
|---------|------------|------|
| `DISCORD_TOKEN` | **必須** | Discordボットトークン |
| `DISCORD_GUILD_ID` | **必須** | サーバーID |
| `DISCORD_VOICE_CHANNEL_ID` | **必須** | ボイスチャンネルID |
| `DISCORD_IGNORED_USER_IDS` | (空) | 無視するユーザーIDのリスト（カンマ区切り） |
| `WHISPER_MODEL` | kotoba-tech/kotoba-whisper-v1.1 | STTモデル名 |
| `QWEN_MODEL` | Qwen/Qwen3-32B-GGUF | LLMモデル名 |
| `QWEN_MODEL_FILE` | Qwen3-32B-Q5_K_M.gguf | GGUFファイル名 |
| `DEVICE` | cuda | 使用デバイス |
| `MODELS_DIR` | ./models | モデル保存ディレクトリ |
| `AUDIO_SAMPLE_RATE` | 48000 | サンプリングレート (Hz) |
| `AUDIO_CHANNELS` | 2 | チャンネル数（ステレオ） |
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
│   ├── audio_buffer.py      # ユーザーごとの音声バッファ管理
│   ├── speech_to_text.py    # Speech-to-Text (Kotoba-Whisper)
│   └── comment_generator.py # コメント生成 (Qwen3-32B)
├── models/                  # モデルファイル（.gitignore）
├── pyproject.toml           # Poetry設定
├── .env                     # 環境変数（.gitignore）
├── .env.example             # 環境変数テンプレート
└── README.md
```

## 技術詳細

### 音声処理

- Discordから受信する音声はステレオ（48kHz、2チャンネル）
- バッファ追加時にモノラルに変換して保存
- ユーザーごとに独立したバッファで管理
- 60秒分の音声を保持し、30秒ごとに処理

### モデル

- **Kotoba-Whisper**: 日本語音声認識に特化したWhisperモデル
- **Qwen3-32B**: Alibabaが開発した大規模言語モデル（GGUF形式で量子化）
- Qwen3の`<think>`ブロックは自動的に除去されます

## トラブルシューティング

### ROCm関連のエラー

```bash
# ROCm のバージョン確認
rocm-smi

# PyTorch を ROCm 対応で再インストール
poetry run pip uninstall torch -y
poetry run pip install torch --index-url https://download.pytorch.org/whl/rocm6.1

# llama-cpp-python を再インストール
poetry run pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DGGML_HIP=on -DCMAKE_PREFIX_PATH=/opt/rocm-6.1.5 -DROCM_PATH=/opt/rocm-6.1.5" \
  poetry run pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir --verbose
```

### PyTorchがGPUを認識しない

```bash
# PyTorchの状態確認
poetry run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ROCm対応PyTorchをインストール
poetry run pip install torch --index-url https://download.pytorch.org/whl/rocm6.1
```

### VRAM不足エラー

使用するモデルサイズを小さくしてください:

- Qwen3-14B (約14GB)
- Qwen3-7B (約7GB)

または、より小さいWhisperモデルを使用:

```bash
WHISPER_MODEL=openai/whisper-small
```

### Discord接続エラー

- ボットトークンが正しいか確認
- ボットが対象サーバーに招待されているか確認
- ボットに適切な権限（Voice Channelへの接続権限）があるか確認

### 音声が取得できない

- py-cord の音声受信機能は実験的機能です
- ffmpeg がインストールされているか確認: `ffmpeg -version`
- デバッグ用WAVファイル（`debug_audio_user*.wav`）を確認

### サンプリングレートが間違っている

音声が遅く低く聞こえる場合、ステレオ/モノラル変換が正しく行われていない可能性があります。最新版ではこの問題は修正されています。

## ライセンス

MIT License

## 注意事項

- このアプリケーションはDiscordの音声を録音します。プライバシーに配慮して使用してください
- 大規模言語モデルの生成内容は必ずしも適切とは限りません
- ローカルで動作するため、インターネット接続は不要です（モデルダウンロード後）
- `DISCORD_IGNORED_USER_IDS` を設定することで、BOTなど特定のユーザーの音声を無視できます
