# Athena

A fully offline, privacy-first voice assistant that runs entirely on local hardware. Athena combines a large mixture-of-experts language model (Qwen3.5), neural text-to-speech (Orpheus 3B), real-time speech recognition (Whisper), and a SNAC neural audio codec into a four-process pipeline — all in C++ with zero Python dependencies.

Athena speaks with natural emotion (laughs, sighs, gasps), maintains long conversational context, and runs on a single consumer GPU + system RAM. No cloud, no telemetry, no API keys.


## Architecture

```
                                     ┌──────────────────────┐
                                     │ llama-server         │
                                     │ (Orpheus 3B GGUF)    │
                                     │ GPU inference        │
                                     │ -np 2 (2 slots)      │
                                     └──────┬───────────────┘
                                            │ HTTP /completion
                                            │ (all-ahead parallel
                                            │  + 200ms stagger)
┌────────────────────────────────┐          │
│ orpheus-speak (daemon)         │◄─────────┘
│                                │
│  text → character-based chunks │
│  → concurrent HTTP submission  │
│  → SNAC token parse            │
│  → SNAC ONNX decode (GPU/CPU) │
│  → double-buffered WAV         │
│  → pipelined playback          │
│    + PulseAudio drain guard    │
└────────────┬───────────────────┘
             │ file watch (speak_tts.txt)
             │
┌────────────┴───────────────────┐        ┌──────────────────┐
│ speak-daemon.sh                │◄───────│ talk-llama       │
│ (sanitize markdown/URLs,       │  write │ (Whisper STT     │
│  atomic trigger file write)    │        │  + Qwen3.5 LLM   │
└────────────────────────────────┘        │  + MoE CPU offload│
                                          │  + voice I/O)     │
                                          └──────────────────┘
```

Four processes run simultaneously:

1. **llama-server** — runs Orpheus 3B on GPU with 2 parallel slots. Accepts text prompts via HTTP, outputs SNAC audio tokens via continuous batching.

2. **orpheus-speak** (daemon) — watches a trigger file. Splits long text into character-limited chunks (≤300 chars), submits all chunks as concurrent HTTP requests with retry-on-503, decodes SNAC tokens via ONNX Runtime, and plays audio through double-buffered WAV files with pipelined playback and PulseAudio drain guard.

3. **talk-llama** — Whisper speech-to-text + Qwen3.5 chat LLM. Writes assistant responses to a speak file. Supports MoE CPU offloading — expert FFN tensors live in system RAM while attention and shared experts run on GPU.

4. **speak-daemon.sh** — bridges talk-llama and orpheus-speak. Sanitizes markdown, URLs, and chat template tokens, then atomically writes to the trigger file.


## Prerequisites

- CMake ≥ 3.18
- C++17 compiler (gcc ≥ 8, clang ≥ 10)
- libcurl development headers
- SDL2 development headers (for microphone capture)
- ONNX Runtime (C/C++ GPU release)
- CUDA toolkit (12.x or 13.x)
- cuDNN 9.x (see installation below)

```bash
sudo apt install build-essential cmake git libcurl4-openssl-dev libsdl2-dev moreutils
```

`moreutils` provides the `ts` timestamp utility used for instrumented logging.


## Install cuDNN 9

ONNX Runtime GPU requires cuDNN 9.x. The CUDA toolkit does **not** include cuDNN.

1. Download the **Full** variant (not JIT) from
   [developer.nvidia.com/cudnn-downloads](https://developer.nvidia.com/cudnn-downloads).
   Select: Linux → x86_64 → Debian → deb (local) → **FULL**.

2. Install:

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.20.0/local_installers/cudnn-local-repo-debian12-9.20.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-debian12-9.20.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-debian12-9.20.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn9-cuda-12   # or cudnn9-cuda-13 for CUDA 13.x
```

3. Ensure `/etc/ld.so.conf` includes the ONNX Runtime library path, then run `sudo ldconfig`.

Use **Full**, not JIT — JIT compiles kernels at runtime, adding latency.


## Download Models

All models can be downloaded via `wget` or `curl`.

### Orpheus 3B GGUF (TTS)

| Variant | Size | Link |
|---------|------|------|
| UD-Q4_K_XL (recommended) | 1.98 GB | [Download](https://huggingface.co/unsloth/orpheus-3b-0.1-ft-GGUF/resolve/main/orpheus-3b-0.1-ft-UD-Q4_K_XL.gguf) |
| Q4_K_M | 1.94 GB | [Download](https://huggingface.co/unsloth/orpheus-3b-0.1-ft-GGUF/resolve/main/orpheus-3b-0.1-ft-Q4_K_M.gguf) |
| Q8_0 (best quality) | 4.03 GB | [Download](https://huggingface.co/unsloth/orpheus-3b-0.1-ft-GGUF/resolve/main/orpheus-3b-0.1-ft-Q8_0.gguf) |

```bash
mkdir -p ~/models/Orpheus
wget -O ~/models/Orpheus/orpheus-3b-0.1-ft-UD-Q4_K_XL.gguf \
    https://huggingface.co/unsloth/orpheus-3b-0.1-ft-GGUF/resolve/main/orpheus-3b-0.1-ft-UD-Q4_K_XL.gguf
```

[Browse all Orpheus quants](https://huggingface.co/unsloth/orpheus-3b-0.1-ft-GGUF/tree/main)

### SNAC 24kHz Decoder (ONNX)

```bash
wget -O ~/Orpheus/snac24_dynamic_fp16.onnx \
    https://huggingface.co/onnx-community/snac_24khz-ONNX/resolve/main/onnx/decoder_model_fp16.onnx
```

[Browse model page](https://huggingface.co/onnx-community/snac_24khz-ONNX)

### ONNX Runtime (C/C++ GPU)

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-x64-gpu_cuda13-1.24.4.tgz
tar xzf onnxruntime-linux-x64-gpu_cuda13-1.24.4.tgz
```

[Browse releases](https://github.com/microsoft/onnxruntime/releases)

### Qwen3.5-122B-A10B (chat LLM — T15g Gen 2)

Split GGUF shards — download all parts to the same directory.

| Variant | Size | Shards | RAM |
|---------|------|--------|-----|
| [UD-Q6_K_XL](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF/tree/main) (recommended) | ~105 GB | 4 | 128+ GB |
| [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF/tree/main) (faster on DDR4) | ~58 GB | 2 | 96+ GB |

```bash
mkdir -p ~/models/QWEN-3.5-122B
for i in $(seq -w 1 4); do
    wget -O ~/models/QWEN-3.5-122B/Qwen3.5-122B-A10B-UD-Q6_K_XL-0000${i}-of-00004.gguf \
        "https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF/resolve/main/Qwen3.5-122B-A10B-UD-Q6_K_XL-0000${i}-of-00004.gguf"
done
```

### Qwen3.5-397B-A17B (chat LLM — P16 Gen 3)

| Variant | Size | Shards | RAM |
|---------|------|--------|-----|
| [UD-Q3_K_XL](https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF/tree/main) | ~165 GB | 6 | 192 GB |

```bash
mkdir -p ~/models/QWEN-3.5-397B
for i in $(seq -w 1 6); do
    wget -O ~/models/QWEN-3.5-397B/Qwen3.5-397B-A17B-UD-Q3_K_XL-0000${i}-of-00006.gguf \
        "https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF/resolve/main/Qwen3.5-397B-A17B-UD-Q3_K_XL-0000${i}-of-00006.gguf"
done
```

### Whisper (speech-to-text)

```bash
mkdir -p ~/models/Whisper
wget -O ~/models/Whisper/ggml-small.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
```

[Browse Whisper models](https://huggingface.co/ggerganov/whisper.cpp/tree/main)


## Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DGGML_SCHED_MAX_COPIES=1 \
    -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Change `-DCMAKE_CUDA_ARCHITECTURES=86` to `120` for Blackwell GPUs.

[llama.cpp repository](https://github.com/ggml-org/llama.cpp)


## Build whisper.cpp with custom talk-llama

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cp ~/Orpheus/talk-llama.cpp examples/talk-llama/talk-llama.cpp
cmake -B build -DWHISPER_SDL2=ON -DGGML_CUDA=ON -DGGML_SCHED_MAX_COPIES=1 \
    -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

[whisper.cpp repository](https://github.com/ggml-org/whisper.cpp)


## Build orpheus-speak

```bash
cd ~/Orpheus
cmake -B build \
    -DONNXRUNTIME_ROOT=/home/user/onnxruntime-linux-x64-gpu_cuda13-1.24.4 \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cp speak-daemon.sh ~/models/Whisper/speak-daemon.sh && chmod +x ~/models/Whisper/speak-daemon.sh
```


## Running

```bash
# Basic
./launch-athena.sh

# With timestamped logging
./launch-athena.sh 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' | tee athena.log
```


## System Setup

**Kernel tuning** — create `/etc/sysctl.d/99-athena.conf`:

```ini
vm.swappiness = 0
vm.max_map_count = 262144
vm.dirty_ratio = 5
vm.dirty_background_ratio = 2
```

**Memory locking** — create `/etc/security/limits.d/athena.conf`:

```
user    soft    memlock    unlimited
user    hard    memlock    unlimited
```

**Disable swap:**

```bash
sudo swapoff -a
sudo sed -i '/\sswap\s/s/^/#/' /etc/fstab
```


## Hardware Configurations

### ThinkPad T15g Gen 2 (`launch-athena.sh`)

| | Spec |
|-|------|
| CPU | Intel i7-11800H (8C/16T) |
| RAM | 128 GB DDR4-3200 |
| GPU | NVIDIA RTX 3080 Laptop 16 GB (384 GB/s) |
| LLM | Qwen3.5-122B-A10B UD-Q6_K_XL |
| Context | 81,920 tokens |
| Threads | 15 (1 reserved for OS/audio pipeline) |

### ThinkPad P16 Gen 3 (`launch-athena (for P16 Gen 3).sh`)

| | Spec |
|-|------|
| CPU | Intel Core Ultra 9 285HX (8P+16E / 32T) |
| RAM | 192 GB DDR5-4000 |
| GPU | NVIDIA RTX PRO 5000 Blackwell 24 GB GDDR7 (896 GB/s) |
| LLM | Qwen3.5-397B-A17B UD-Q3_K_XL |
| Context | 131,072 tokens |
| Threads | 28 (4 reserved for OS/audio pipeline) |


### VRAM Budget — T15g Gen 2

| Component | VRAM |
|-----------|------|
| 122B-A10B non-expert weights (Q6_K) | ~6.4 GB |
| KV cache bf16 (12 attn layers, 82K ctx) | ~1.9 GB |
| GatedDeltaNet state (36 linear-attn layers) | ~0.15 GB |
| Compute buffers | ~1.0 GB |
| Orpheus UD-Q4_K_XL + KV (q8_0, 9216, 2 slots) | ~2.75 GB |
| Whisper small | ~0.65 GB |
| SNAC decoder (GPU) | ~0.1 GB |
| CUDA overhead | ~0.5 GB |
| **Total** | **~13.5 GB** |
| **Free (16 GB GPU)** | **~2.5 GB** |


## Orpheus TTS Pipeline

### Parallel chunked processing

Long responses (>300 characters) are split into character-limited chunks and
processed through a pipelined architecture:

1. **Character-based chunking** — groups sentences until the next would exceed
   300 characters. Keeps each request within per-slot context budget.

2. **All-ahead parallel submission** — all chunks submitted as concurrent HTTP
   requests with 200ms stagger. llama-server processes 2 simultaneously.

3. **Retry with backoff** — HTTP 503 triggers up to 3 retries (500ms/1000ms).

4. **Pipelined playback** — each chunk plays via background thread with
   double-buffered WAV files while later chunks generate.

5. **PulseAudio drain guard** — prevents buffer flush on long audio chunks.

### Context budget formula

```
per_slot = total_context / n_slots
per_slot ≥ (CHUNK_CHAR_LIMIT × 6.6) + 50 prompt + 100 margin
```

### Measured performance (RTX 3080 Laptop)

| Metric | Sequential (-np 1) | Parallel (-np 2) |
|--------|-------------------|-------------------|
| Single-shot latency | 5.6s | 3.4s |
| Wall-time efficiency | 53% | 80% |
| First-audio latency (long) | ~67s | ~34s |

### Voices

| Voice | Description |
|-------|-------------|
| tara | Female, conversational (best rated) |
| leah, jess, mia, zoe | Female |
| leo, dan, zac | Male |

### Emotion tags

```
<laugh>  <chuckle>  <sigh>  <cough>  <sniffle>  <groan>  <yawn>  <gasp>
```


## Technical Notes

### SNAC token format

Orpheus outputs 7 tokens per audio frame. Each frame encodes 3 RVQ codebook
layers in interleaved order: `[L0, L1a, L2a, L2b, L1b, L2c, L2d]`.
Each codebook has 4096 entries. Audio tokens start at `<custom_token_10>`.

### Audio properties

24,000 Hz mono. Each SNAC frame = 512 samples ≈ 21.3ms. ~47 frames/second.
7 tokens/frame = ~329 tokens per second of audio.


## Troubleshooting

| Problem | Solution |
|---------|----------|
| "no audio tokens in response" | Verify llama-server is running and Orpheus model loaded |
| CUDA EP unavailable | Install cuDNN 9, check `/etc/ld.so.conf`, run `sudo ldconfig` |
| Audio truncation | Ensure per-slot context ≥ 2048 tokens. Check `CHUNK_CHAR_LIMIT` |
| SNAC decoder init failed | Delete `.optimized` cache, retry. Add `--snac-cpu` if persistent |
| OOM | Reduce `--ctx-size`, drop Orpheus to Q4_K_M, check VRAM budget |
| Echo | Increase 200ms sleep in talk-llama.cpp |
| speak-daemon.sh hangs | Check orpheus-speak is running and file paths match |


## Files

| File | Purpose |
|------|---------|
| `orpheus-speak.cpp` | TTS pipeline: HTTP client, parallel submission, SNAC decoder, chunked playback with drain guard |
| `CMakeLists.txt` | Build configuration for orpheus-speak |
| `talk-llama.cpp` | Custom whisper.cpp talk-llama: Athena prompt, MoE offload, VAD, goodbye detection, date awareness |
| `speak-daemon.sh` | talk-llama → orpheus-speak bridge (file-based IPC) |
| `launch-athena.sh` | Launcher for T15g Gen 2 with perf tuning |
| `launch-athena (for P16 Gen 3).sh` | Launcher for P16 Gen 3 with 397B support |
| `valperf.sh` | Runtime performance settings validator |
| `README.md` | This file |


## License

MIT licensed. Orpheus weights: Llama 3.2 community license. SNAC codec: MIT. Qwen3.5: Apache-2.0.
