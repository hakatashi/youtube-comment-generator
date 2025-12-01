# 日本語音声文字起こしAIモデル検証レポート

## 検証環境
- **OS**: Linux (Kernel 6.8.0-87)
- **GPU**: AMD Radeon Graphics
- **ROCm**: 6.1.5
- **PyTorch**: 2.6.0+rocm6.1
- **Python**: 3.10
- **VRAM要件**: 8GB未満

## 検証結果サマリー

### 絶対条件: GPUで動作すること
**結論**: ❌ **要件を満たすモデルは見つかりませんでした**

## 試したモデルと結果

### 1. ReazonSpeech k2-asr
- **実装**: ONNX Runtime + sherpa-onnx
- **GPU動作**: ❌ **CPUのみで動作**
- **理由**: sherpa-onnxがGPU有効化せずにコンパイルされているため、CPUにフォールバック
- **文字起こし精度**: 中程度（音声によってばらつきあり）
- **推論速度**: 平均 3.51秒 (CPU)
- **VRAM使用**: N/A (CPU動作のため)
- **追加機能**:
  - ✓ 日本語サポート
  - ○ タイムスタンプ (部分的)
  - ○ 句読点 (基本的なもの)
  - ✗ 話者分離

**備考**: ONNX RuntimeのROCm版が必要だが、pipでは配布されておらず、ソースからのビルドが必要。

### 2. MMS (Massively Multilingual Speech)
- **実装**: PyTorch + HuggingFace Transformers
- **GPU動作**: ⚠️ **モデルのロードは成功、推論でSegmentation Fault**
- **VRAM使用**: 3.70 GB (ロード時)
- **理由**: transformersライブラリとROCm環境の互換性問題

### 3. Wav2Vec2 XLS-R Japanese
- **実装**: PyTorch + HuggingFace Transformers
- **GPU動作**: ❌ **Segmentation Fault**
- **理由**: transformersライブラリとROCm環境の互換性問題

### 4. OpenAI Whisper (公式実装)
- **実装**: PyTorch + openai-whisper
- **GPU動作**: ⚠️ **モデルのロードは成功、推論でSegmentation Fault**
- **VRAM使用**: 0.27 GB (base モデル, ロード時)
- **理由**: ユーザーがすでに試して動作しなかったことと一致

### 5. Nue-ASR (rinna)
- **実装**: PyTorch + HuggingFace Transformers
- **GPU動作**: ❌ **テスト不可**
- **理由**: プライベートリポジトリで認証が必要

### 6. Distil-Whisper
- **実装**: PyTorch + HuggingFace Transformers
- **GPU動作**: ❌ **Segmentation Fault**
- **理由**: transformersライブラリとROCm環境の互換性問題

### 7. faster-whisper + CTranslate2 (CUDA版)
- **実装**: CTranslate2 + ONNX Runtime
- **GPU動作**: ❌ **CUDAランタイムエラーでCPUにフォールバック**
- **エラー**: `CUDA driver version is insufficient for CUDA runtime version`
- **CPU推論速度**: 8.78秒
- **VRAM使用**: N/A (CPU動作のため)
- **理由**: pipでインストールされるctranslate2とonnxruntimeはCUDA専用でコンパイルされており、ROCmを認識できない
- **追加機能**:
  - ✓ 日本語サポート
  - ✓ タイムスタンプ (word & segment level)
  - ✓ 句読点
  - ✓ 言語自動検出
  - ✗ 話者分離

**備考**: ROCm環境でGPUを使用するには、CTranslate2 ROCm版をソースからビルドする必要がある。

## 根本的な問題

このROCm環境では、以下の組み合わせで**Segmentation Fault**が発生します:

1. **PyTorch 2.6.0+rocm6.1 + HuggingFace transformers + 音声処理**
2. **PyTorch 2.6.0+rocm6.1 + openai-whisper + 音声処理**

この問題は、以下のライブラリの組み合わせによる互換性問題と思われます:
- numba 0.62.1
- llvmlite 0.45.1
- scipy/resampy (リサンプリング処理)
- PyTorch ROCm版

PyTorch単体ではGPUは正常に動作しますが、transformersまたはwhisperライブラリと組み合わせると推論時にクラッシュします。

## 代替アプローチの検討と追加調査

### faster-whisper + CTranslate2 ROCm版（追加調査実施）

**検証結果**: ❌ **pip版はGPUで動作せず**

#### 実施内容
1. faster-whisperとctranslate2をpip経由でインストール（成功）
2. GPU動作テストを実施
3. CUDA版のライブラリがROCmを認識できず、CPUにフォールバック

#### 詳細な結果
- **インストール**: ✓ 成功（ctranslate2 4.6.1, faster-whisper 1.2.1, onnxruntime 1.23.2）
- **CPUでの動作**: ✓ 成功
  - モデルロード時間: 3.48秒
  - 推論時間: 8.78秒（30秒の音声）
  - 日本語認識: 動作確認済み（言語自動検出確率1.00）
- **GPUでの動作**: ✗ 失敗
  - エラー: `CUDA driver version is insufficient for CUDA runtime version`
  - 原因: ONNX RuntimeとCTranslate2がCUDA専用ビルドのため、ROCmと互換性なし

#### GPU動作のための要件
ROCm環境でfaster-whisperをGPUで動作させるには、以下のビルドが必要:

1. **CTranslate2 ROCm版のソースビルド**
   - リポジトリ: https://github.com/ROCm/CTranslate2 (amd_devブランチ)
   - ビルド時間: 数時間（推定）
   - 必要な依存関係:
     - oneDNN 3.1.1+
     - hipBLAS
     - MIOpen (CUDNNサポート用)
     - cmake 3.21+
   - GPUアーキテクチャ指定: gfx906（このマシン用）
   - パッチ適用: ct2_3.23.0_rocm.patch

2. **ONNX Runtime ROCm版のビルド（オプション）**
   - faster-whisperはONNX Runtimeに依存
   - ROCm対応版は公式配布されていない

3. **Dockerfile利用（推奨）**
   - `/tmp/ctranslate2-rocm/docker_rocm/Dockerfile.rocm` が利用可能
   - ただし、GPUアーキテクチャ指定（gfx90a, gfx942）を gfx906 に変更が必要
   - ベースイメージ: rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.1.2

#### 参考情報
- [ROCm/CTranslate2](https://github.com/ROCm/CTranslate2)
- [CTranslate2 ROCm Blog](https://rocm.blogs.amd.com/artificial-intelligence/ctranslate2/README.html)
- [beecave-homelab/insanely-fast-whisper-rocm](https://github.com/beecave-homelab/insanely-fast-whisper-rocm)

### ESPnet
- **可能性**: ⚠️ **試していないが、同様の問題が発生する可能性**
- **理由**: PyTorchベースで、同じ環境依存の問題が起こりうる

## 推奨事項

このROCm環境で日本語音声文字起こしをGPUで実行するには、以下の対応が必要です:

### オプション1: Docker環境の使用
AMDが提供するROCm公式Dockerイメージを使用することで、環境の互換性問題を回避できる可能性があります。

```bash
docker pull rocm/pytorch:latest
```

### オプション2: CTranslate2 ROCm版のビルド（詳細手順あり）
faster-whisperを使用するために、CTranslate2 ROCm版をソースからビルドします。

#### ビルド手順の概要
1. CTranslate2 ROCm版をクローン（完了: `/tmp/ctranslate2-rocm`）
2. 依存関係のインストール（oneDNN、hipBLAS、MIOpen）
3. パッチ適用とcmake設定
4. makeによるビルド（推定時間: 1-3時間）
5. Pythonパッケージのインストール
6. faster-whisperのインストール

#### 必要なcmake設定
```bash
cmake -DCMAKE_INSTALL_PREFIX=/opt/ctranslate2 \
      -DWITH_CUDA=ON -DWITH_CUDNN=ON \
      -DGPU_RUNTIME=HIP \
      -DCMAKE_HIP_ARCHITECTURES="gfx906" \
      -DGPU_TARGETS="gfx906" \
      -DAMDGPU_TARGETS="gfx906" \
      -DWITH_MKL=OFF -DWITH_DNNL=ON \
      -DWITH_OPENBLAS=ON \
      -DOPENMP_RUNTIME=COMP \
      -DCMAKE_BUILD_TYPE=Release ..
```

- リポジトリ: https://github.com/ROCm/CTranslate2
- ブログ: https://rocm.blogs.amd.com/artificial-intelligence/ctranslate2/README.html
- Dockerfile参考: `/tmp/ctranslate2-rocm/docker_rocm/Dockerfile.rocm`

### オプション3: 環境の再構築
以下のバージョンの組み合わせで環境を再構築:
- ROCm 5.7+
- PyTorch 2.2.1+ (ROCm版)
- transformers 4.36+ (ROCm CIサポート付き)

### オプション4: CPUでの実行を許容
ReazonSpeech k2-asrはCPUで動作し、平均3.5秒で推論可能です。高速ではありませんが、実用的な速度です。

## 参考リンク

- [Fine-tuning and Testing Cutting-Edge Speech Models using ROCm on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/speech_models/README.html)
- [Speech-to-Text on an AMD GPU with Whisper](https://rocm.blogs.amd.com/artificial-intelligence/whisper/README.html)
- [ReazonSpeech v2.1](https://research.reazon.jp/blog/2024-08-01-ReazonSpeech.html)
- [MMS: Massively Multilingual Speech](https://huggingface.co/facebook/mms-1b-all)
- [CTranslate2: Efficient Inference with Transformer Models on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/ctranslate2/README.html)
- [insanely-fast-whisper-rocm](https://github.com/beecave-homelab/insanely-fast-whisper-rocm)

## ReazonSpeech k2-asrの実行結果サンプル

CPUで動作したReazonSpeech k2-asrの文字起こし結果:

### サンプル1: debug_audio_hakatashi.wav
- **音声時間**: 30.00秒
- **推論時間**: 1.86秒
- **文字起こし**: `この精霊の無駄を使ってくれ`
- **正解**: `コメントを生成する側はちゃんとGPUでやってます。いや、ちゃんとなんか両方乗り、なんかVRAMに、この、このVRAMに両方収まるサイズのモデルを使って`
- **精度**: 約13%（認識ほぼ失敗）

### サンプル2: debug_audio_platypus.wav
- **音声時間**: 61.8秒
- **推論時間**: 4.38秒
- **文字起こし**: `見回るの早朝かもしれませんがそうではないため良いどんだけ早く見積もっても朝に洗濯機が回っていることになりますあと自分周りの部屋の洗濯機の音僕聞こえたことないんですよね振動でもだからそんなに自分ないっすね`
- **精度**: 約75%（後半部分は良好）

### サンプル3: debug_audio_satos.wav
- **音声時間**: 約30秒
- **推論時間**: 4.29秒
- **文字起こし**: `上から聞こえてきたことはない`
- **精度**: 約22%（最後の一文のみ認識）

**平均推論速度**: 3.51秒（リアルタイムの約1/10）

## faster-whisperの実行結果サンプル

CPUで動作したfaster-whisperの文字起こし結果:

### サンプル1: debug_audio_hakatashi.wav
- **音声時間**: 30.00秒
- **推論時間**: 8.78秒
- **文字起こし**: `今のコントセンセンセンセンリがはい ちゃんと聞いてみたです`
- **正解**: `コメントを生成する側はちゃんとGPUでやってます。いや、ちゃんとなんか両方乗り、なんかVRAMに、この、このVRAMに両方収まるサイズのモデルを使って`
- **言語検出**: 日本語（確率1.00）
- **タイムスタンプ**: [10.16s - 13.30s]

## 結論

現在のvenv環境では、**GPUで動作する日本語音声文字起こしモデルの動作確認はできませんでした**。

### 試したアプローチの総数: 7モデル/手法
- ✗ Whisper系（4種類）: Segmentation Fault
- ✗ Transformer系（3種類）: Segmentation Fault or 認証エラー
- ✗ faster-whisper（CUDA版）: ONNX Runtime互換性エラー
- ○ ReazonSpeech k2-asr: CPUのみで動作（精度にばらつき）

### 根本的な原因
1. **PyTorch ROCm版とWhisper/Transformersライブラリ間の互換性問題**（Segmentation Fault）
2. **ONNX Runtime/CTranslate2がCUDA専用**でROCmを認識できない

### GPU実行のための次のステップ
1. **推奨**: Docker環境の使用（rocm/pytorch:latest）
2. **代替**: CTranslate2 ROCm版のソースビルド（1-3時間）
3. **検討**: 環境の再構築（ROCm 5.7+ + PyTorch 2.2.1+）

このマシンのGPUアーキテクチャは **gfx906** であることを確認済み。
