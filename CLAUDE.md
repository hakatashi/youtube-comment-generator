# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Discord voice channel audio recorder that generates YouTube-style comments in real-time using local AI models. It captures audio from Discord voice channels, transcribes it using faster-whisper (Kotoba-Whisper), and generates comments using Qwen3-32B LLM running on AMD ROCm GPU.

## Development Commands

### Running the Application

```bash
# Run with Poetry (recommended)
poetry run python -m src.main

# Or activate Poetry shell first
poetry shell
python -m src.main
```

### Web Viewer Development

```bash
# Install dependencies
cd viewer
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

### Firebase Deployment

```bash
# Deploy Firestore rules
firebase deploy --only firestore:rules

# Deploy hosting (viewer)
firebase deploy --only hosting

# Deploy all
firebase deploy
```

### Model Installation

On first run, models are automatically downloaded:
- **Qwen3-32B** (~20GB) → `models/` directory
- **Kotoba-Whisper v2.0** (~3GB) → `models/faster-whisper/` directory

### ROCm-Specific Setup

```bash
# Install PyTorch with ROCm support
poetry run pip install torch --index-url https://download.pytorch.org/whl/rocm6.1

# Install llama-cpp-python with ROCm support
CMAKE_ARGS="-DGGML_HIP=on -DCMAKE_PREFIX_PATH=/opt/rocm-6.1.5 -DROCM_PATH=/opt/rocm-6.1.5" \
  poetry run pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir --verbose
```

## Architecture

### Audio Processing Pipeline

The application processes audio in a three-stage pipeline:

1. **Audio Buffer (src/audio_buffer.py)**
   - Manages per-user audio buffers using a chunk-based system
   - Automatically merges audio chunks within 0.5s intervals (defined by `CHUNK_MERGE_EPSILON`)
   - Converts stereo Discord audio (48kHz, 2ch) to mono on ingestion
   - Maintains 60 seconds of rolling audio per user, dropping older chunks
   - Provides time-based retrieval and multi-user merging in chronological order

2. **Speech-to-Text (src/speech_to_text_faster.py)**
   - Uses faster-whisper (CTranslate2-based) for 5.6x speedup over PyTorch
   - Runs on CPU because ROCm + PyTorch + transformers causes segfaults on MI50 GPUs
   - Resamples audio to 16kHz before transcription
   - Removes silence using energy threshold (-45dB default, 0.5s minimum silence)
   - Processes merged audio from all users in a single STT call (optimization)

3. **Comment Generation (src/comment_generator.py)**
   - Uses Qwen3-32B (Q5_K_M quantized, ~24GB VRAM) via llama-cpp-python
   - Runs on GPU with ROCm backend (uses CUDA API compatibility)
   - Maintains 30-comment rolling history for context
   - Uses Jinja2 templates (`src/templates/comment_prompt.jinja`) for prompts
   - Automatically strips Qwen3's `<think>` blocks from output
   - Validates comments: 1-30 chars, must contain Unicode letters

### Main Loop (src/main.py)

- Runs every 30 seconds (configurable via `AUDIO_INTERVAL`)
- Checks which users have ≥60s of audio data
- Merges all ready users' audio chronologically by chunks
- Applies silence removal (shared between debug output and STT)
- Saves debug WAV files to `debug/` directory
- Runs STT and comment generation in thread pool executor
- Sends comments to Discord text channel and saves to Firestore

### Firestore Schema

**Batch metadata** at `/batches/{batchId}`:
- `created_at`: Timestamp
- `count`: Number of comments generated
- `prompt`: Full prompt sent to LLM
- `transcription`: Merged transcription text
- `user_transcriptions`: Array of per-user transcriptions

**Individual comments** at `/batches/{batchId}/comments/{commentId}`:
- `comment`: Generated comment text
- `created_at`: Timestamp

This structure supports both batch queries and collection group queries across all comments.

### Web Viewer (viewer/)

Built with Solid.js and Firebase SDK:
- Main page (`/`) displays all comments in real-time using Firestore listeners
- Admin page (`/admin`) shows batch-level metadata with expandable prompts
- Firebase config in `viewer/src/lib/firebase.ts`
- Type definitions in `viewer/src/lib/schema.d.ts`

## Key Implementation Details

### Chunk-Based Audio Merging

The audio buffer uses a chunk system where each `add_audio()` call creates a chunk. Chunks within 0.5s are automatically merged. When getting merged audio from multiple users, chunks are sorted chronologically and concatenated without splitting, preserving the temporal structure of the conversation.

### ROCm GPU Compatibility

- **LLM (Qwen3)**: Runs on GPU via llama-cpp-python with ROCm backend
- **STT (faster-whisper)**: Runs on CPU because CTranslate2 only supports CUDA GPUs, not ROCm
- **Why CPU STT works**: STT runs once per 30s interval, while LLM is the main computational bottleneck

The original PyTorch + transformers implementation caused segfaults on MI50 GPUs with ROCm 6.1.5 + PyTorch 2.6.0. Switching to faster-whisper (CTranslate2) solved this while providing a 5.6x speedup.

### Thread Pool Executor

CPU/GPU-intensive operations (STT, LLM) run in a 2-worker thread pool to prevent blocking the async event loop. This allows Discord audio reception to continue smoothly while processing.

### Configuration System

All config lives in `src/config.py` with environment variable overrides. Required variables:
- `DISCORD_TOKEN`, `DISCORD_GUILD_ID`, `DISCORD_VOICE_CHANNEL_ID`, `DISCORD_TEXT_CHANNEL_ID`

Optional variables allow customizing models, intervals, sample rates, etc.

### User Filtering

Set `DISCORD_IGNORED_USER_IDS` (comma-separated integers) to exclude specific users (e.g., bots) from audio processing. Filtering happens in `AudioBufferSink.write()` before buffering.

## Important Constraints

- **Do not modify ROCm installation paths**: The build expects ROCm at `/opt/rocm-6.1.5`
- **Firebase credentials**: Must have `*-firebase-adminsdk-*.json` in project root (auto-detected)
- **Model files**: Store in `models/` directory; never commit to git
- **Debug audio files**: Saved to `debug/` directory; gitignored
- **Environment secrets**: Keep `.env` file out of version control

## Common Tasks

### Adding New Comment Generation Parameters

1. Update the Jinja2 template in `src/templates/comment_prompt.jinja`
2. Modify `CommentGenerator._create_prompt()` to pass new variables
3. Update `CommentGenerator.generate_comments()` call in `main.py` if needed

### Changing STT Model

Set `WHISPER_MODEL` environment variable to a different faster-whisper compatible model. Must be a CTranslate2 format model or convertible via `ct2-transformers-converter`.

### Changing LLM Model

1. Set `QWEN_MODEL` (HuggingFace repo) and `QWEN_MODEL_FILE` (GGUF filename)
2. Ensure model fits in VRAM (32GB on MI50)
3. Adjust `n_gpu_layers` in `CommentGenerator.__init__()` if needed

### Modifying Audio Buffer Duration

Change `AUDIO_BUFFER_DURATION` (default: 60s) and/or `AUDIO_INTERVAL` (default: 30s). Buffer duration should be ≥ interval to avoid data gaps.

### Debugging Audio Issues

1. Check `debug/debug_audio_merged_*.wav` files to verify audio quality
2. Ensure ffmpeg is installed: `ffmpeg -version`
3. Monitor Discord bot logs for audio reception stats (every 10s)
4. Verify stereo→mono conversion and silence removal thresholds

## Testing

No formal test suite currently exists. Manual testing workflow:

1. Run application with test Discord channel
2. Speak in voice channel and wait 30s
3. Check console output for transcription and generated comments
4. Verify comments appear in Discord text channel
5. Check Firestore console for batch/comment documents
6. Test viewer at `http://localhost:3000` (dev mode)

## Dependencies

- **Python**: 3.10-3.13 (3.14 not supported due to dependencies)
- **PyTorch**: Must use ROCm 6.1 wheel, not pip default
- **llama-cpp-python**: Must be built with ROCm support (see setup commands)
- **Firebase Admin SDK**: Requires service account JSON file
- **Node.js**: ≥22 for viewer
