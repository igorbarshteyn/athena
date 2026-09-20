# Athena

A fully offline, privacy-first voice assistant that runs entirely on local hardware. Athena combines a large mixture-of-experts language model (Qwen3.5), neural text-to-speech (Orpheus 3B), real-time speech recognition (Whisper), and a SNAC neural audio codec into a four-process pipeline — all in C++ with **zero Python at runtime** (one optional Python script exists for a one-time, offline model conversion at setup).

Athena speaks with natural emotion (laughs, sighs, gasps), **reads the emotion in your voice** and responds to it, **remembers across sessions** (an evolving long-term memory and personality that persist between conversations), maintains long conversational context, is **interruptible mid-sentence** (speak over her and she stops, keeping what she already said in context), and runs on a single consumer GPU + system RAM. No cloud, no telemetry, no API keys.

The TTS path is built around **SSE token streaming with incremental SNAC decoding**: audio for each chunk begins ~200–400 ms after the chunk is submitted, playback is gapless through a persistent raw-PCM sink with an anti-underrun write clock, and end-of-speech detection is parameterized. The stack is also **self-healing**: a health watchdog auto-restarts the TTS backend on a crash, kernel/GPU faults are captured durably for diagnosis, and the launcher hardens the NVIDIA driver configuration at install time (see [Stability, diagnostics & crash capture](#stability-diagnostics--crash-capture)).


## Demo

<p align="center">
  <a href="https://www.youtube.com/watch?v=8HuRUpJ4_as&t=237s">
    <img src="https://img.youtube.com/vi/8HuRUpJ4_as/maxresdefault.jpg" alt="ATHENA — live demo video" width="720">
  </a>
</p>
<p align="center"><em>▶ Watch the demo on YouTube</em> — a live voice-to-voice session (offline, single GPU).</p>


## Quick start

Models are **not** bundled — the repo ships source, scripts, and the ATHENA patches; `install.sh` fetches everything else. Assuming the GPU stack is already installed (NVIDIA **595.x** driver, CUDA **12.9**, cuDNN **9.x** — see [Install the NVIDIA driver + CUDA 12.9](#install-the-nvidia-driver--cuda-129-ubuntu-2404)):

```bash
git clone <this-repo-url> athena && cd athena   # clone ANYWHERE — paths auto-derive
./install.sh          # prints system requirements, then: models (~179 GB) →
                      # ONNX Runtime → llama.cpp (master) → whisper.cpp (pinned)
                      # + patches → orpheus-speak → integrity check
./launch-athena-397b.sh
```

The repo is **relocatable**: the launcher resolves its own directory at runtime (through symlinks, from any working dir), so it runs from wherever you clone it — no fixed path required. `install.sh` is **idempotent and resumable** — re-run any time; verified-complete downloads and existing binaries are skipped. Options: `--check-only` (verify an existing install), `--force` (rebuild), `--skip-models`/`--skip-ort`/`--skip-llama`/`--skip-whisper`/`--skip-orpheus`/`--skip-emotion`, `--jobs N`, `-y`. The manual step-by-step (to understand what the script does) is in [Download components](#download-components) onward — its `cd /home/user/ATHENA/...` examples are the reference machine's path; substitute your checkout.


## Repository layout

Everything lives under one root (`/home/user/ATHENA` on the reference machine — adjust paths to your checkout):

```
ATHENA/
├── launch-athena-397b.sh        # the orchestrator/supervisor (bash) — the real entry point
├── Athena 397B.desktop          # GNOME one-click launcher (note: filename contains a space)
├── llama.cpp/                   # vendored, built in-tree → build/bin/llama-server
├── whisper.cpp/                 # vendored, built in-tree → build/bin/whisper-talk-llama
│   └── examples/talk-llama/     #   ← custom talk-llama.cpp + athena_memory.h copied here pre-build
├── orpheus/                     # ATHENA's own TTS engine
│   ├── orpheus-speak.cpp        #   SSE ingestion + sliding-window SNAC decode + playback
│   ├── snac24_dynamic_fp16.onnx #   SNAC decoder (see Downloads)
│   └── build/orpheus-speak
├── onnxruntime-linux-x64-gpu_cuda12-1.27.0/   # prebuilt ONNX Runtime (never built)
├── patches/talk-llama/          # the 5 ATHENA files patched into whisper.cpp (source of truth)
├── models/                      # ALL model files, FLAT (incl. all 5 Qwen shards)
│   └── emotion2vec-reexport.py  #   one-time ONNX export script (setup-time Python)
├── athena-gpu-monitor.sh        # per-run GPU/kernel telemetry sampler
├── athena-kmsg-logger.cpp       # durable kernel-log capture (C++; built by apply-crash-capture.sh)
├── apply-gsp-stability.sh       # installs the driver-hardening modprobe confs (+ UVM hardening)
├── apply-crash-capture.sh       # arms panic capture (sysctls + efi-pstore + kmsg logger)
├── check-kernel-fix.sh          # watches for the CVE-2026-43303-fixed kernel in apt
├── STABILITY-RUNBOOK.md         # step-by-step kernel/driver stability procedure
├── CHANGES.MD                   # session-by-session change log — read this first when hacking
├── CLAUDE.md                    # repo guide (architecture ground truth + gotchas)
└── athena_memory/               # cross-session memory + personality (created at first goodbye)
```


## Architecture

```
                                  ┌───────────────────────────┐
                                  │ llama-server              │
                                  │ (Orpheus 3B GGUF, GPU)    │
                                  │ -np 2 parallel slots      │
                                  └──────────┬────────────────┘
                                             │ HTTP /completion
                                             │ "stream": true  (SSE)
┌────────────────────────────────────┐       │
│ orpheus-speak (daemon, --stream-tts)│◄──────┘
│                                    │
│  per-chunk SSE thread:             │
│   token scan (split-safe)          │
│   → 4-frame sliding-window         │
│     SNAC decode (ONNX, GPU)        │
│  ordered playback thread:          │
│   startup watermark (350 ms)       │
│   → raw S16 PCM → aplay (pipe)     │
│   device write clock + 60 ms       │
│   low-water silence guard          │
│   stdin close = exact drain        │
└────────────┬───────────────────────┘
             │ inotify on trigger file
             │ (IN_CLOSE_WRITE = streaming session,
             │  IN_MOVED_TO    = legacy batch via speak-daemon.sh)
┌────────────┴───────────────────┐        ┌────────────────────────┐
│ speak-daemon.sh                │◄───────│ talk-llama             │
│ (legacy batch bridge —         │  -s/-sf│ (Whisper STT + Silero  │
│  dormant when --stream-file    │        │  VAD + prosodic        │
│  is active)                    │        │  endpointing + Qwen3.5 │
└────────────────────────────────┘        │  MoE CPU offload +     │
                                          │  emotion2vec + memory) │
                                          └───────────┬────────────┘
                                                      │ --stream-file: sentence
                                                      │ lines + ---END--- sentinel
                                                      ▼
                                              trigger file (speak_tts.txt)
```

Four processes run simultaneously (all started by `launch-athena-397b.sh`; the three GPU clients share one CUDA context via **MPS**):

1. **llama-server** — runs Orpheus 3B on GPU with **2 parallel slots** and continuous batching (was 4; single-user gains no latency from more slots, and halving peak concurrent decode grows the per-stream real-time margin to ~1.5×). Accepts text prompts via HTTP and streams SNAC audio tokens back as server-sent events. A **watchdog** in the launcher auto-restarts it within seconds if it ever dies (see Stability).

2. **orpheus-speak** (daemon, `--stream-tts`) — watches the trigger file. Sentences are grouped into ≤300-char chunks and submitted as concurrent SSE requests (up to 5 in flight). Tokens are decoded **as they arrive** through a 4-frame sliding window; PCM flows in chunk order into one raw-PCM player per response (spawned via `posix_spawn`, keeping the CUDA-laden process out of the kernel's fork paths). The legacy batch pipeline remains available via `--no-stream-tts`.

3. **talk-llama** — Whisper speech-to-text with Silero VAD and optional prosodic endpointing, the Qwen3.5-397B-A17B chat LLM (run **in-process**, not over HTTP), an inline **emotion2vec** speech-emotion tagger, and **cross-session memory**. Streams assistant sentences into the trigger file as they are generated. Supports MoE CPU offloading — expert FFN tensors live in system RAM while attention and shared experts run on GPU. Handles **barge-in**: it monitors the mic during playback, signals the TTS daemon to stop when you interrupt, and rolls its own transcript back to just the words she actually spoke.

4. **speak-daemon.sh** — legacy bridge for talk-llama's `-s/-sf` interface. Dormant when `--stream-file` is set (the current configuration); kept as a fallback.

> Two LLMs run at once: `llama-server` on `127.0.0.1:8080` serves **only** Orpheus TTS; the conversational Qwen model runs **inside** `whisper-talk-llama`.


## Streaming TTS pipeline (`--stream-tts`)

The default in `launch-athena-397b.sh`. Per response ("session"):

1. **Chunking** — sentence lines from talk-llama are grouped into chunks of ≤300 chars. When the pipeline is empty (start of turn), the first partial sentence is flushed immediately for minimum latency.
2. **SSE ingestion** — each chunk POSTs with `"stream":true`. A split-safe scanner extracts `<custom_token_N>` even when a token string straddles two SSE events; the final `"stop":true` event is never scanned (some llama-server builds repeat the accumulated content there, which would duplicate audio).
3. **Sliding-window decode** — to emit frame *e*, frames [e−1 .. e+2] are decoded at a constant `[1,4]/[1,8]/[1,16]` shape and the middle 2048-sample slice is kept (1 frame left context, 2 frames lookahead). The first window emits frames 0–1; the last window flushes the tail — **no frames are dropped**. Constant tiny shapes make each decode a few ms. The window shape is pre-warmed at daemon start so the first session pays no ONNX setup.
4. **Startup watermark** — each session's first write is held until `--prebuffer-ms` (default 350) of audio is buffered, or chunk 1 completes, or the session ends — whichever is first. Never applies mid-turn.
5. **Persistent raw-PCM sink** — one `aplay -q -t raw -f S16_LE -c 1 --buffer-time=300000 -r 24000 -` per session, fed over a pipe with blocking writes for backpressure. Closing stdin makes the player drain and exit, so the `.done` signal fires exactly when the last sample has played.
6. **Device write clock** — buffered audio ≈ samples written − wall time since first write. Silence is injected **only** when this estimate drops below 60 ms — a genuine gap waiting on the next chunk's text.

With `-v`, each session prints a summary (`first audio`, `stall silence`, `low water`, per-chunk `first audio in X ms`). `stall silence` should be 0 or small.

**Per-session capture** (`ATHENA_TTS_CAPTURE=1`, default): each TTS session's raw tokens (`.tokens`), parsed audio codes (`.codes`), decoded audio (`.wav`), and summary (`.meta`) are dumped under the run's diag dir — the data needed to root-cause any audio corruption offline (`--decode-codes` re-decodes a capture on CPU; `--compare-wav` prints a MATCH/MISMATCH verdict).

### Protocol

talk-llama appends sentence lines to the trigger file and terminates with `---END---`. orpheus-speak signals completion by creating the `.done` file **after playback fully drains** (`COMPLETE`, or `INTERRUPTED <lines> <chars>` on a barge-in), deletes the trigger file, and drains stale inotify events. `IN_MOVED_TO` (atomic mv from speak-daemon.sh) selects batch mode; `IN_CLOSE_WRITE` selects a streaming session.


## Voice activity detection

talk-llama uses the **Silero VAD** (`--vad-engine silero`, model `ggml-silero-v6.2.0.bin`) with a tunable confirmation window, plus whisper.cpp's energy detector as the underlying edge trigger.

| Flag | Launcher | Meaning |
|------|--------:|---------|
| `--vad-engine silero` | on | neural VAD (Silero v6.2) |
| `--vad-model` | `models/ggml-silero-v6.2.0.bin` | Silero weights |
| `-vwm`, `--vad-window-ms` | 700 | trailing window the energy VAD inspects |
| `-vlm`, `--vad-last-ms` | 400 | required trailing-silence duration before processing |
| `--silero-min-run-ms` | 100 | minimum voiced run Silero must report |
| `--silero-debug` | on (tuning) | per-poll Silero trace — remove for quiet logs |

| Profile | window | last-ms | ~end-of-speech latency |
|---------|-------:|--------:|-----------------------:|
| Conservative | 1000 | 500 | ~1.2 s |
| **Balanced (launcher)** | **700** | **400** | **~0.75 s** |
| Aggressive | 500 | 300 | ~0.55 s |

Clipped mid-pause → raise the window first. Laggy → lower the window first. An optional **trailing-silence trim** (`ATHENA_TRIM_TRAILING_MS=N`, off by default) keeps just N ms past the last voiced frame before Whisper — a hallucination mitigation; emotion2vec still sees the full buffer.

### Prosodic endpointing (`--endpoint`)

Optional; the launcher passes it. The single fixed end-of-turn silence wait is replaced with **two**, chosen from the pitch/energy contour of the speech just before the pause: a sentence-final **fall or trail-off** → short wait (answer fast); a **flat or rising** "not done yet" pause → long wait (keep listening).

| Flag | Launcher value | Meaning |
|------|---------------:|---------|
| `--endpoint` | on | enable prosodic endpointing |
| `--endpoint-short-ms` | **800** | silence wait when the ending sounds turn-final. **Set equal to `--endpoint-long-ms` — the prosodic short path is collapsed** (CHANGES.MD §23.7): `turn_final` is an OR of f0-fall/energy-decay with no minimum-silence floor on the Silero path, so the old 350 ms short target cut mid-thought pauses in slow/emotional speech. With short==long, prosody's verdict no longer affects timing. |
| `--endpoint-long-ms` | **800** | silence wait when the speaker seems mid-thought. 800 = above this box's genuine turn-ends (~700–770 ms) and above the ~640 ms within-turn-pause median; drop both to 700 for snappier turn-taking, raise to 850 for more margin. (Long-ms had been lowered 1200→700 because the pitch detector reads voiced≈0 on quiet post-speech audio, so the long target already fired nearly every turn.) |
| `--endpoint-f0-fall` | 60 | F0 slope below −this (Hz/s) counts as falling |
| `--endpoint-energy-decay` | 4.0 | log-energy slope below −this (/s) counts as trailing off |
| `--endpoint-f0-fall-strong` | 0 (off) | a fall steeper than −this is turn-final on its own |
| `--endpoint-extend-ms` | **1200** (0=off) | **prosody-to-EXTEND** (CHANGES.MD §23.8): on a "not-done" pre-pause contour, wait this long instead of the base target — **upward only** (`max()`), never shortens, so it can't regress §23.7. The robust use of prosody: grant *more* time when the acoustics say the speaker is mid-thought. Capped at base+400 to bound the dead-air of any false-positive. |
| `--endpoint-f0-rise` | 50 | F0 slope above +this (Hz/s), voiced ⇒ pitch rising |
| `--endpoint-energy-rise` | 0.5 | log-energy slope above +this (/s), voiced ⇒ energy rising/held. **Continuation requires `!turn_final` AND pitch-rise AND energy-rise (an AND).** Why: a wrong extend is *unrecoverable* (Athena answers late, user is silent, no barge to recover); a turn-final yes/no **question** rises in pitch but its energy *tapers* into the pause → energy-rise fails → not extended, while a genuine held continuation both rises and sustains energy. If questions still over-extend, raise `--endpoint-f0-rise` or set `--endpoint-extend-ms 0`. |

With `-pe`, each turn prints one `endpoint f0_slope=.. e_slope=.. voiced=.. -> turn-final/continue` line for threshold tuning.


## Interruptibility (barge-in)

Athena can be interrupted mid-sentence: playback stops within tens of milliseconds and she responds to what you said.

1. **Playback aborts fast** — the TTS daemon kills the active audio stream once the interruption is confirmed.
2. **Only what she actually said stays in context** — the LLM transcript is rolled back (via a full state snapshot taken in the dead air between turns) to the original user line plus *only the words she got out*, followed by your interruption.
3. **She recovers conversationally** — the spoken prefix remains in context, so follow-ups stay coherent.

Backchannels — "sure", "mm-hm", "yeah" — are treated as listening noises, not interruptions (distinguished by transcript **content**, not just energy).

| Flag | Launcher value | Meaning |
|------|---------------:|---------|
| `--barge-rms` | 0.0020 | absolute energy floor a barge must exceed |
| `--barge-ratio` | 1.5 | multiple of measured ambient it must also exceed |
| `--barge-ms` | 150 | how long that energy must be sustained |
| `--barge-blackout-ms` | 200 | arm delay after each utterance begins |

Audio routing is config-only, pinned **per stream** via `PULSE_SOURCE`/`PULSE_SINK`:

- `ATHENA_AUDIO=direct` (default) — mic + playback on the bare headset (`HEADSET_PATTERN`, default `Logi_USB_Headset`).
- `ATHENA_AUDIO=aec` — WebRTC `module-echo-cancel` for open speakers (requires `pactl`; falls back to direct on failure).

On a confirmed barge, talk-llama creates the `STOP_FILE` (`speak_tts.stop`); orpheus-speak kills audio within one ALSA period and reports `INTERRUPTED <lines> <chars>` in `.done`. Because Qwen3.5 is a **hybrid** model (GatedDeltaNet recurrent layers cannot un-integrate a token suffix), rollback restores a full-state snapshot (~0.4 GiB host RAM, ~40 ms warm) and re-decodes the user line plus the spoken prefix.

> **A headset is recommended.** Self-echo rejection on open speakers is not fully tuned; a headset makes the problem moot.

> The rolled-back turn keeps the spoken prefix with **no special end marker** — earlier cut-off markers taught the model to imitate them.


## Emotion detection (emotion2vec)

talk-llama reads the **emotion in your voice** from each utterance's audio and appends an `[emotion: <label>]` tag to the transcript line the LLM sees (you never hear it). The persona prompt tells the model to treat the tag as tone of voice and respond to the feeling without naming it.

Detection is an [emotion2vec](https://huggingface.co/emotion2vec) `plus_large` ONNX model, run **inline in talk-llama on GPU 0** (CUDA EP; force CPU with `ATHENA_EMOTION_CPU=1`). It compiles in automatically when the whisper.cpp build passes `-DONNXRUNTIME_ROOT`. Without ONNX Runtime (or with the model file absent), tagging compiles out / disables and everything else is unchanged.

### What the model can actually resolve

The raw model emits nine classes; six are "real" emotions and three (neutral, other, unk) produce no tag. Calibration on this mic + model showed it resolves **coarse valence only**: among the negative emotions it cannot reliably separate angry / disgusted / fearful / sad. Production therefore runs with **negative-collapse** on: `{angry, disgusted, fearful, sad}` fold to a single `[emotion: negative]`, leaving an honest **happy / surprised / negative** signal.

### Two profiles (selected in the launcher)

- **Production** (default) — collapse on; floors: base `0.50`, `happy=0.50`, `sad=0.60`. (`happy` was originally floored at 0.95, but genuine happy delivery peaked at ~0.58 in field runs — the 0.95 floor was unreachable. 0.50 is a deliberate demo lever that accepts a small fear-leak mislabel risk; raise it back toward 0.95 if upset speech starts tagging happy. `sad=0.60` trims warm/playful false-positives (~0.55) while keeping true sad (≥0.92).)
- **Calibration** (`ATHENA_EMOTION_CALIBRATION=1`) — collapse off, per-class precision floors (`happy/sad/disgusted = 0.95/0.90/0.90`), for measuring individual classes.

```bash
./launch-athena-397b.sh                              # production
ATHENA_EMOTION_CALIBRATION=1 ./launch-athena-397b.sh # per-class calibration
```

### Environment variables (set in the launcher, not CLI flags)

| Variable | Launcher | Meaning |
|----------|----------|---------|
| `ATHENA_EMOTION_ONNX` | `models/emotion2vec_plus_large.onnx` | enables tagging; unset/missing disables it |
| `ATHENA_EMOTION_CPU` | unset | force the CPU execution provider |
| `ATHENA_EMOTION_DEBUG` | `1` | log every classification + active floors at startup |
| `ATHENA_EMOTION_CALIBRATION` | `0` | profile selector |
| `ATHENA_EMOTION_COLLAPSE_NEGATIVE` | set (production) | fold negatives → `[emotion: negative]` |
| `ATHENA_EMOTION_MIN` | `0.50` | base probability floor |
| `ATHENA_EMOTION_MIN_<LABEL>` | `HAPPY=0.50`, `SAD=0.60` | per-class floor overrides |


## Cross-session memory & personality

With `--memory <dir>` set (the launcher points it at `/home/user/ATHENA/athena_memory`), Athena keeps an **evolving long-term memory and personality** that persist between conversations. The logic lives in the header-only `athena_memory.h` (namespace `amem`, std-only), patched in next to `talk-llama.cpp` at build time.

Consolidation runs **only on a graceful "Goodbye Athena"** exit — Ctrl+C does not write memory. At goodbye, the session is distilled: **extract** memorable moments (importance blended with emotion2vec vocal salience), **decay** existing memories on an Ebbinghaus curve (flashbulb memories resist pruning), **compact** faded memories into gist, **render** `memory.txt`, and threshold-gated **personality** revision via `personality.ledger`. All writes are atomic. Next session, `memory.txt` and `personality.txt` are injected with a fresh "time since last session" header.

| File (in the memory dir) | Role |
|------|------|
| `memory.txt` | memories injected into next session's prompt |
| `personality.txt` | evolving personality description |
| `memory.state.tsv` | per-memory strength / salience / timestamps |
| `personality.ledger` | accumulating evidence for the threshold-gated revision |
| `meta` | last-session exit timestamp |

| Flag | Launcher | Meaning |
|------|----------|---------|
| `--memory DIR` | `/home/user/ATHENA/athena_memory` | enable memory + personality |
| `--memory-words N` | `2048` | soft word budget for the injected block |
| `--time-refresh-min N` | `15` | re-show current time after this many minutes |
| `--personality-reflect-every N` | `1` (**temporary demo edit** — lowers the revision threshold so one session can fire; default 4, drop the flag to revert) | personality-integration cadence |


## Prerequisites

- **Ubuntu 24.04.4 LTS** (the tested platform; kernel `6.17.0-1028-oem`)
- CMake ≥ 3.18, C++17 compiler
- libcurl development headers
- SDL2 development headers (microphone capture)
- NVIDIA driver **595.x (open kernel modules)** + CUDA toolkit **12.9** + cuDNN **9.x full** (install below)
- ONNX Runtime **1.27.0** (prebuilt C/C++ GPU, CUDA 12 — download below; never built)

```bash
sudo apt install build-essential cmake git libcurl4-openssl-dev libsdl2-dev moreutils pulseaudio-utils
```

`moreutils` provides the `ts` timestamp utility used for logging; `pulseaudio-utils` provides `pactl` for the `aec` audio path.


## Verified build environment

This stack is **pinned and reproducible** — the versions below were built and tested together on the Blackwell reference machine (`sm_120a`). The Qwen3.5 / GatedDeltaNet CUDA path is version-sensitive on this hardware, so build the exact llama.cpp release and whisper.cpp commit named here rather than current `master`.

| Component | Pinned version |
|-----------|----------------|
| GPU architecture | NVIDIA Blackwell, `sm_120a` (`-DCMAKE_CUDA_ARCHITECTURES="120a;86"`) |
| OS | **Ubuntu 24.04.4 LTS** (kernel `6.17.0-1028-oem`) |
| NVIDIA driver | **595.71.05** (open kernel modules, DKMS, from the NVIDIA CUDA apt repo) |
| CUDA toolkit | **12.9** (nvcc V12.9.86 = `cuda-toolkit-12-9` 12.9.2) |
| cuDNN | **9.23.2.1-1** (`cudnn9-cuda-12`, FULL variant) |
| Host compiler | GCC **13.3.0** (Ubuntu 13.3.0-6ubuntu2~24.04.1) |
| llama.cpp | release **b9253** |
| whisper.cpp | commit `afa2ea544fb4b0448916b4a31ecd33c8685bd482` |
| ONNX Runtime | **1.27.0** (`linux-x64-gpu_cuda12`) |

`nvcc --version` on the reference machine:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_May_27_02:21:03_PDT_2025
Cuda compilation tools, release 12.9, V12.9.86
Build cuda_12.9.r12.9/compiler.36037853_0
```


## Install the NVIDIA driver + CUDA 12.9 (Ubuntu 24.04)

The driver and toolkit come from NVIDIA's CUDA apt repo for Ubuntu 24.04, installed as debs with **DKMS** (so kernel updates rebuild the module automatically). The `ubuntu2404` repo carries the driver, the CUDA **12.9** toolkit, and cuDNN all in one place — no second repo needed:

```bash
# 1) NVIDIA CUDA repo for Ubuntu 24.04 (driver + CUDA 12.9 toolkit + cuDNN)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 2) Open-kernel-module driver, 595 branch (the versioned package pins the branch)
sudo apt install -y nvidia-driver-595-open       # → 595.71.05-0ubuntu0.24.04.1 (DKMS build)

# 3) CUDA 12.9 toolkit (nvcc V12.9.86)
sudo apt install -y cuda-toolkit-12-9            # → 12.9.2
echo 'export PATH=/usr/local/cuda-12.9/bin:$PATH' >> ~/.bashrc
```

Reboot after the driver install, then verify: `nvidia-smi --query-gpu=driver_version --format=csv,noheader` → `595.71.05`, and `nvcc --version` → `V12.9.86`.

> The versioned `nvidia-driver-595-open` package keeps `apt upgrade` on the tested 595 branch (it won't jump to the 610 feature branch). Newer 595.x point releases install in place with `sudo apt install nvidia-driver-595-open` once the repo publishes them.


## Install cuDNN 9

ONNX Runtime GPU requires cuDNN 9.x. The CUDA toolkit does **not** include cuDNN. Use the **FULL** variant, not JIT — JIT ships only runtime-fusion engines and compiles kernels at first use (warm-up latency), which matters on a brand-new arch like `sm_120a`.

**Network repo (simplest — uses the ubuntu2404 repo added above):**

```bash
sudo apt install -y cudnn9-cuda-12         # latest = 9.23.2.1-1 (the tested pin)
```

**Local repo deb (offline-friendly alternative; direct URL, no login needed):**

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.23.2/local_installers/cudnn-local-repo-ubuntu2404-9.23.2_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.23.2_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2404-9.23.2/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt update && sudo apt install -y cudnn9-cuda-12
```

Finally, make sure the ONNX Runtime library path is known to the loader: add it to `/etc/ld.so.conf.d/` (or `/etc/ld.so.conf`) and run `sudo ldconfig` — without this the CUDA execution provider is silently unavailable and SNAC/emotion2vec fall back or fail.


## Download components

All model files go **flat into `models/`** (the launcher's preflight expects exactly this layout — including all five Qwen shards; if you relocate them into a subdirectory, update `QWEN_MODEL` in the launcher to match). All URLs below were verified live (2026-07-02); they are plain `wget`-able links — no Hugging Face CLI or Python tooling required.

### ONNX Runtime 1.27.0 (C/C++ GPU, CUDA 12)

Since 1.27.0 the CUDA version is part of the asset name, and the tarball extracts to exactly the directory name the build commands reference — no renaming needed:

```bash
cd /home/user/ATHENA
wget https://github.com/microsoft/onnxruntime/releases/download/v1.27.0/onnxruntime-linux-x64-gpu_cuda12-1.27.0.tgz
tar xzf onnxruntime-linux-x64-gpu_cuda12-1.27.0.tgz     # → onnxruntime-linux-x64-gpu_cuda12-1.27.0/
```

Compatibility: the `gpu_cuda12` build works with **any CUDA 12.x** and **cuDNN 9.x**. (NVIDIA's release notes deprecate the CUDA-12 line — a `gpu_cuda13` asset exists if this stack ever moves to CUDA 13.)

### Qwen3.5-397B-A17B (chat LLM) — 5 shards, ~179 GB

The supported chat model. **Note the `UD-Q3_K_XL/` subfolder in the URL** — it is required:

```bash
cd /home/user/ATHENA/models
for i in $(seq -w 1 5); do
  wget -c "https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF/resolve/main/UD-Q3_K_XL/Qwen3.5-397B-A17B-UD-Q3_K_XL-0000${i}-of-00005.gguf"
done
```

`QWEN_MODEL` in the launcher points at shard 1; llama.cpp loads the rest automatically (the preflight verifies all five exist — a missing later shard would otherwise fail minutes into load). The model is a 397B-parameter MoE with ~17B active per token; under `--cpu-moe` the expert FFN tensors stay in system RAM (~161 GB resident — hence the 192 GB requirement) while attention, shared experts, and the router run on the GPU.

### Orpheus 3B GGUF (TTS)

```bash
cd /home/user/ATHENA/models
wget https://huggingface.co/unsloth/orpheus-3b-0.1-ft-GGUF/resolve/main/orpheus-3b-0.1-ft-UD-Q4_K_XL.gguf
```

| Variant | Size | Notes |
|---------|------|-------|
| **UD-Q4_K_XL** (used) | 2.13 GB | tested pin |
| Q4_K_M | 1.94 GB | smaller |
| Q8_0 | 4.03 GB | best quality, costs VRAM |

### SNAC 24 kHz decoder (ONNX, fp16, dynamic axes)

The decoder-only fp16 export from `onnx-community` — this exact file (byte-verified against the reference install) goes into `orpheus/`, where the launcher passes it explicitly (`orpheus-speak`'s compiled-in default filename does not exist — the launcher's `--snac` flag is what makes it work):

```bash
wget -O /home/user/ATHENA/orpheus/snac24_dynamic_fp16.onnx \
  https://huggingface.co/onnx-community/snac_24khz-ONNX/resolve/main/onnx/decoder_model_fp16.onnx
```

> orpheus-speak writes an optimized graph to `<snac>.optimized` at startup; the launcher removes it before each run (a GPU-optimized cache breaks a later CPU run).

### Whisper small.en (speech-to-text)

```bash
cd /home/user/ATHENA/models
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin
```

English-only `small.en` (better accuracy on English than multilingual `small`).

### Silero VAD v6.2 (voice activity detection)

Required — the launcher passes `--vad-engine silero` and preflight-checks the file:

```bash
cd /home/user/ATHENA/models
wget https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin
```

### emotion2vec plus_large → ONNX (speech emotion — optional, one-time Python conversion)

There is no ready-made ONNX with the dynamic sequence axis Athena needs (a naive export bakes the trace-time clip length into the graph and fails on any utterance of a different length, on both CPU and CUDA). The repo ships a **one-time, offline conversion script** — `models/emotion2vec-reexport.py` — that downloads the source `iic/emotion2vec_plus_large` checkpoint itself (via FunASR/ModelScope), exports with a **symbolic sequence dimension**, and validates honestly: it runs clips both shorter *and longer* than the trace length (2–6 s) and asserts every one executes without a Reshape error **and matches FunASR's own output**.

```bash
cd /home/user/ATHENA/models
pip install -U funasr torch torchaudio modelscope onnx onnxruntime
python3 emotion2vec-reexport.py 2>&1 | tee emotion2vec-reexport.out
# ship the resulting emotion2vec_plus_large.onnx ONLY if a strategy reports FULLY DYNAMIC
```

This is the only Python in the project, and it never runs at assistant runtime — once `emotion2vec_plus_large.onnx` exists in `models/`, the toolchain can be removed. **Optional**: if the file is absent (or the build was made without ONNX Runtime), the launcher disables tagging and Athena runs without emotion tags.


## Get the sources (llama.cpp + whisper.cpp)

Both upstreams are cloned **into the ATHENA folder** and built in-tree. Two supported paths:

- **Pinned (tested — recommended):** the exact versions this stack was validated on. The Qwen3.5/GatedDeltaNet CUDA path is version-sensitive on `sm_120a`.
- **Current master:** supported by the patch set below (the patched `CMakeLists.txt` auto-detects newer llama syncs — see the DSA note), but upstream API drift in the bundled llama sources can require `talk-llama.cpp` adjustments. If a current-master build fails, fall back to the pin.

```bash
cd /home/user/ATHENA

# llama.cpp — pinned release b9253 (tested)      [current: omit the checkout]
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && git checkout b9253 && cd ..

# whisper.cpp — pinned commit (tested)           [current: omit the checkout]
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp && git checkout afa2ea544fb4b0448916b4a31ecd33c8685bd482 && cd ..
```

### Apply the ATHENA patches to whisper.cpp

`patches/talk-llama/` contains the **five** files that turn upstream `talk-llama` into Athena. Copy them in **before** configuring:

```bash
cd /home/user/ATHENA
cp patches/talk-llama/* whisper.cpp/examples/talk-llama/
```

| File | Role |
|------|------|
| `talk-llama.cpp` | the Athena brain: persona, MoE offload, streaming, emotion tagging, memory glue, barge-in, brain fail-safe, anchored turn tags |
| `athena_memory.h` | header-only cross-session memory + personality (`#include`d by talk-llama.cpp) |
| `silero-endpointer.h` | header-only Silero streaming end-of-turn detector (`#include`d by talk-llama.cpp) |
| `silero-turn-state.h` | turn-state machine used by the endpointer (`#include`d by silero-endpointer.h) |
| `CMakeLists.txt` | upstream build file **plus** the ONNX Runtime/emotion2vec wiring (`ATHENA_EMOTION_ORT=1`, RPATH) **and** a conditional source guard for `llama-kv-cache-dsa.cpp` |

Notes:
- The four sources need no CMake registration — only `.cpp` files are compiled directly, and the three headers are included from the same directory.
- **DSA guard:** current whisper.cpp ships `llama-kv-cache-dsa.cpp/.h` (DeepSeek Sparse Attention) in this example; the pinned commit predates it. The patched `CMakeLists.txt` compiles that file only if it exists, so the same patch set builds **both** trees — an unpatched ATHENA CMakeLists on current master would fail to link, and upstream's own CMakeLists would silently drop emotion2vec.
- A clean re-clone or `git checkout .` in whisper.cpp **erases the patches** — re-copy them after any upstream sync. `patches/talk-llama/` is the source of truth; if you edit the build-tree copies, sync them back.

## Build llama.cpp

Produces `llama.cpp/build/bin/llama-server` (the Orpheus TTS backend). Build command as tested on Blackwell (`sm_120a`) with GCC 14.2 + CUDA 12.9:

```bash
cd /home/user/ATHENA/llama.cpp

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="120a;86" \
  -DGGML_CUDA_FORCE_CUBLAS=OFF \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_F16=ON \
  -DGGML_SCHED_MAX_COPIES=1 \
  -DGGML_NATIVE=ON \
  -DGGML_LTO=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA_GRAPHS=ON \
  -DLLAMA_BUILD_UI=OFF

cmake --build build -j 12
```

`120a` targets Blackwell's architecture-specific features; for other GPUs change `-DCMAKE_CUDA_ARCHITECTURES` (e.g. `86` for Ampere). Base CUDA graphs (`GGML_CUDA_GRAPHS=ON`) stay on; the experimental `GGML_CUDA_GRAPH_OPT` runtime flag is **not** used anywhere (it has a buffer-reuse race — the "unspecified launch failure" class — and measured no benefit).

## Build whisper.cpp with the custom talk-llama

Produces `whisper.cpp/build/bin/whisper-talk-llama`. Requires the [ATHENA patches](#apply-the-athena-patches-to-whispercpp) applied **before** configuring:

```bash
cd /home/user/ATHENA/whisper.cpp

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="120a;86" \
  -DONNXRUNTIME_ROOT=/home/user/ATHENA/onnxruntime-linux-x64-gpu_cuda12-1.27.0 \
  -DGGML_CUDA_FORCE_CUBLAS=OFF \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_F16=ON \
  -DGGML_SCHED_MAX_COPIES=1 \
  -DGGML_NATIVE=ON \
  -DGGML_LTO=ON \
  -DWHISPER_SDL2=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j 12
```

`-DONNXRUNTIME_ROOT` enables emotion2vec (`ATHENA_EMOTION_ORT=1`); `-DWHISPER_SDL2=ON` enables mic capture. `whisper-talk-llama` bundles its **own** copy of the llama sources — it does not link against the `llama.cpp/build` tree.

## Build orpheus-speak

```bash
cd /home/user/ATHENA/orpheus
cmake -B build \
  -DONNXRUNTIME_ROOT=/home/user/ATHENA/onnxruntime-linux-x64-gpu_cuda12-1.27.0 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

`-DONNXRUNTIME_ROOT` is mandatory (CMake errors without it). The ORT `.so` is resolved at runtime via a RUNPATH baked into the binary — moving/renaming the onnxruntime directory breaks the binary until rebuilt. `cmake --install` is not supported; the binary always runs in place from `build/`. Dependencies: libcurl + ONNX Runtime only.


## Running

```bash
cd /home/user/ATHENA

# Basic (production defaults: MPS on, pageable GPU staging, diagnostics on)
./launch-athena-397b.sh

# With timestamped logging (this is exactly what the desktop icon runs)
./launch-athena-397b.sh 2>&1 | ts '[%Y-%m-%d %H:%M:%.S]' | tee athena.log
```

`Athena 397B.desktop` (note: the filename contains a space) is the one-click shortcut wrapping the launcher with the timestamped-logging pipeline. `GGML_CUDA_NO_PINNED=1` and `ATHENA_MPS=1` are **defaulted inside the launcher** — desktop and shell launches now run the same configuration.

The launcher takes **no positional arguments** — every mode is an environment variable. It uses `sudo` for GPU/system tuning (it will block on a password prompt if run non-interactively) and reverts all tuning in `cleanup()` on exit. Startup order: prior-crash harvest → GPU setup → MPS → system/audio setup → preflight (all binaries + models, incl. all 5 Qwen shards) → telemetry sampler → llama-server (poll `/health`) → watchdog + keepalive → orpheus-speak → talk-llama (foreground).

**Cross-session memory is consolidated only on a graceful spoken "Goodbye Athena"** (a ~20–30 s generation at exit). Ctrl+C does not write memory.

### Launcher environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `ATHENA_MPS` | `1` | share one CUDA context across the three GPU clients via MPS (liveness-probed; falls back to no-MPS if the server can't spawn) |
| `GGML_CUDA_NO_PINNED` | `1` | all GPU↔host staging on pageable memory (no `cudaMallocHost`) |
| `ATHENA_DIAG` | `1` | per-run diagnostics dir `athena-diag/<ts>/` (GPU CSV, kernel log, TTS captures, watchdog log) |
| `ATHENA_TTS_CAPTURE` | `1` | per-session TTS token/code/WAV dumps (a few MB per session) |
| `ATHENA_AUDIO` | `direct` | `direct` (headset) or `aec` (open speakers, WebRTC echo-cancel) |
| `HEADSET_PATTERN` | `Logi_USB_Headset` | PulseAudio device match for `direct` |
| `ATHENA_WATCHDOG` | `1` | auto-restart llama-server if it dies or wedges (cap 5 restarts / 300 s) |
| `ATHENA_KEEPALIVE` | `1` | 1-token poke to llama-server every 30 s **only while the GPU is idle** (skipped above `ATHENA_KEEPALIVE_MAX_UTIL`, default 15 %) |
| `ATHENA_COREDUMP` | `0` | scoped core dumps for llama-server + orpheus-speak (never talk-llama); **off by default** — on a memory-corrupting system the coredump's large ext4 write draws poisoned pages and can turn a survivable client abort into a reboot (CHANGES.MD §19a). Set `=1` for forensics on a healthy kernel |
| `ATHENA_GPU_SAMPLE_MS` | `1000` | telemetry cadence (persistent `nvidia-smi --loop-ms`, not spawn-per-sample) |
| `ATHENA_LOCK_MEMCLK` | `0` | GPU memory-clock lock — off by default (clock/power set-calls are ignored on this platform; see GPU power notes) |
| `ATHENA_GPU_POWER_LIMIT` / `ATHENA_GPU_POWER_MAX` | unset | opt-in `-pl` A/B levers only |
| `ATHENA_GPU_CLOCK_CAP` | unset | escalation-only graphics-clock cap |
| `ATHENA_SHUTDOWN_GRACE_DS` | `600` | talk-llama kill grace at teardown (60 s — covers goodbye consolidation) |
| `ATHENA_EMOTION_*` | see Emotion | emotion tagging profile/floors |
| `ATHENA_TRIM_TRAILING_MS` | unset | optional trailing-silence trim before Whisper |


## Command-line reference

### orpheus-speak

| Flag | Default | Description |
|------|---------|-------------|
| `"text"` / `-f FILE` | — | speak literal text / file contents (single-shot mode) |
| `-o FILE` | `/dev/shm/orpheus_tts.wav` | output WAV path (legacy/batch paths) |
| `--voice NAME` | `tara` | Orpheus voice |
| `--api URL` | `http://127.0.0.1:8080/completion` | llama-server endpoint (compiled-in default) |
| `--snac PATH` | (nonexistent default — launcher passes `orpheus/snac24_dynamic_fp16.onnx`) | SNAC ONNX decoder |
| `--watch FILE` | — | daemon mode: watch FILE, speak each session |
| `--stream-tts` / `--no-stream-tts` | off | SSE streaming pipeline (launcher enables it) |
| `--play CMD` / `--play-raw CMD` | `aplay …` | players for batch / streaming paths |
| `--prebuffer-ms N` | 350 | startup watermark |
| `--temp` / `--top-p` / `--rep-pen` | 0.6 / 0.9 / 1.1 | Orpheus sampling |
| `--max-tokens N` | 2500 | per-chunk generation cap (~30 s of audio) |
| `--snac-cpu` | off | force SNAC to CPU (frees VRAM; also used by `--decode-codes` for a contention-free reference) |
| `--capture-dir DIR` | — | per-session token/code/WAV/meta dumps (launcher passes the run dir) |
| `--decode-codes FILE` | — | offline: re-decode a captured `.codes` → `FILE.redecode.wav` |
| `--compare-wav A B` | — | correlation/RMS verdict between two WAVs (clean ≈0.999 MATCH) |
| `--diag` | — | real-time device-fill heartbeat + stall markers |
| `-v`, `--verbose` | off | per-chunk and per-session timing |

Voices: tara, leah, jess, leo, dan, mia, zac, zoe.
Emotion tags: `<laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>`

### talk-llama (key flags as used by the launcher)

| Flag | Launcher value | Description |
|------|----------------|-------------|
| `-ml` / `-mw` | Qwen3.5-397B shard 1 / ggml-small.en | LLM and Whisper models |
| `--ctx-size` | **80000** | conversation context (bf16 KV scales with it) |
| `--mlock --no-mmap --cpu-moe` | on | lock weights, no mmap, all expert FFNs on CPU; paired with `-ngl 99` (every non-expert layer on GPU) |
| `-t 16` (+ `taskset -c 0-15`) | — | 16 threads on cores 0–15; measured ~+30 % MoE decode vs `-t 8` |
| `-ctk/-ctv bf16 -fa` | — | KV cache type (bf16 — a quantized Qwen KV degraded output on this model) + flash attention |
| `--temp/--top-p/--top-k/--min-p` | 0.7 / 0.8 / 20 / 0.0 | Qwen sampling |
| `--presence-penalty / --repeat-penalty` | 1.5 / 1.0 | repetition control (Unsloth-recommended for Qwen3.5) |
| `--reasoning off` | on | disables the thinking channel (voice wants direct replies) |
| `--stream-file FILE` | trigger file | streaming sentence protocol (the active path) |
| `-s` / `-sf` | speak-daemon.sh / speakfile | legacy batch bridge (dormant) |
| `-p` / `-bn` | Igor / Athena | participant names (the user tag is **newline-anchored** as a stop — see safeguards below) |
| `-mt 256` | — | Whisper max tokens per transcription segment (raised 128→256 with the wider window: 25 s of dense speech ≈ 110–140 tokens would hit a 128 cap and tail-truncate) |
| `-vms 25000` | — | capture window handed to Whisper (raised 15000→25000: capture happens ~1.4–1.7 s after last speech — Silero silence + confirm hangover — so a 15 s window covers only ~13.3 s of speech and beheaded a 13 s read turn; ring buffer is 30 s) |
| `--vad-*` / `--silero-*` | see VAD | Silero VAD + confirmation window |
| `--endpoint` (+ `--endpoint-*`) | on | prosodic end-of-turn detection |
| `--barge-in` (+ `--barge-*`) | on | interruptibility |
| `--memory` (+ memory flags) | dir set | cross-session memory |
| `-pe` | on | per-turn VAD/endpoint diagnostics (noisy) |

### Raw-transcript safeguards

talk-llama feeds the model a **raw transcript** with `Igor:` / `Athena:` headers — no chat template. Three guards keep that format from leaking artifacts:

- **Stop on `<|im_start|>`.** Qwen3.5 is ChatML-trained and may try to open a fresh ChatML turn; the token is treated as end-of-turn so the role word after it never leaks into speech.
- **Newline-anchored turn tags.** The stop patterns are `"\nIgor:"` / `"\nAthena:"` — anchored to line starts, exactly how real turn boundaries are written. An unanchored `"Igor:"` once truncated a reply **mid-sentence** when the model addressed the user by name followed by a colon ("…this is peak Igor: …"), and the truncated turn then replayed verbatim on every re-ask. Detection runs *before* the sentence flusher, so a completed tag can never be spoken.
- **Sanitize net.** A trailing `assistant`/`user` welded directly to sentence punctuation (the ChatML-leak signature) is stripped before TTS.


## Stability, diagnostics & crash capture

The diagnostic and hardening tooling below came out of an extended forensic campaign on the reference machine (kernel-level crashes root-caused across `CHANGES.MD` §10–§22). That campaign **acquitted ATHENA and every one of its components** (§22): the corruption reproduced under bare `llama-cli` / `llama-server` GPU load with no ATHENA code in the loop, across multiple drivers and kernels — the remaining suspect is that one machine's power delivery. So the tooling here is **optional** — useful if you ever hit kernel instability, not required to run ATHENA. On the tested **Ubuntu 24.04.4 LTS** config the stack runs stably without any of it.

### Kernel memory corruption (only if you hit it)

Symptoms — random `Bad page state`, `pagealloc: memory corruption`, GPFs in `lruvec_stat_mod_folio`, sporadic **Xid 69** (a co-symptom, the GPU consuming corrupted host memory), and eventual kernel panics — point at a Linux kernel bug (**CVE-2026-43303**, "mm/page_alloc: clear page->private in free_pages_prepare") exercised by sustained CUDA/UVM activity, or a marginal platform. If you see them:

- Run a kernel that carries the CVE-2026-43303 fix; `./check-kernel-fix.sh` reports whether one is installable on your system.
- Meanwhile boot with `page_poison=1 nohugevmalloc` as containment tripwires (they reduce and expose the corruption; they don't fully prevent it).
- `STABILITY-RUNBOOK.md` walks a full kernel-swap procedure — written for the reference machine's original OS and kept for reference; the tested Ubuntu 24.04.4 config needs none of it.

### One-time hardening installs

```bash
./apply-gsp-stability.sh     # driver modprobe confs (reboot to apply):
                             #  - zzz-nvidia-athena-gsp-stability.conf: disables Runtime-D3 + S0ix
                             #    (GSP RPC corruption mitigation; the zzz- prefix is load-bearing —
                             #    modprobe.d is last-assignment-wins and the distro nvidia.conf
                             #    would otherwise override it)
                             #  - athena-uvm-hardening.conf: uvm_disable_hmm=1 uvm_ats_mode=0
                             #    (removes UVM host folio-lifecycle machinery; zero cost for a
                             #    cudaMalloc-only stack)
./apply-crash-capture.sh     # panic capture (applies live + next boot):
                             #  - builds athena-kmsg-logger (C++): unfiltered kernel log,
                             #    fdatasync'd per line → kmsg-full.log survives a hard lock
                             #  - sysctls: panic_on_oops=1, hardlockup_panic=1, panic=30
                             #    (a wedge becomes a captured panic + 30 s auto-reboot)
                             #  - efi-pstore verified as the durable dump sink
```

Verify after reboot (in the GPU-passthrough container use `/proc`, not `/sys/module/nvidia`):
`grep -iE 'EnableS0ix|DynamicPower' /proc/driver/nvidia/params` → both `0`;
`cat /sys/module/nvidia_uvm/parameters/uvm_disable_hmm` → `Y`.

### Always-on runtime defenses (in the launcher)

- **llama-server watchdog** — detects death (`kill -0` fast path) or a wedge (bounded `/health` probes) and respawns the byte-identical server; MPS accepts the fresh client and orpheus-speak reconnects on its next request. TTS recovers in seconds; restarts are logged to `<run>/watchdog.log`.
- **Idle-gated keepalive** — a 1-token poke keeps the CUDA context warm across long idle gaps, skipped whenever the GPU is busy (a busy context is already warm, and injecting concurrent work at the power cap correlated with engine faults).
- **Scoped core dumps** — llama-server and orpheus-speak dump cores into the run dir on a crash (`gdb <bin> <core> -ex bt`); talk-llama is explicitly excluded (a ~161 GiB core is not a diagnostic).
- **Ordered teardown** — orpheus-speak and llama-server stop first; talk-llama stops **last** with a 60 s grace so goodbye consolidation completes (a SIGKILL mid-decode under MPS with ~161 GiB pinned was observed to corrupt kernel page state).
- **Brain fail-safe** — on a poisoned CUDA context (a peer's device fault under MPS reaches every client), talk-llama ends the session cleanly (`_exit(0)`, memory timestamp preserved) instead of SIGABRT.

### Per-run diagnostics (`ATHENA_DIAG=1`, default)

`athena-diag/<timestamp>/` collects: `gpu.csv` (clocks/power/temp/util/VRAM at 1 Hz via one **persistent** `nvidia-smi` per stream — spawn-per-sample churn was itself a kernel-bug trigger), `gpu-proc.log`, `gpu-throttle.log`, `xid.log` (NVRM/Xid subset), **`kmsg-full.log`** (complete kernel log, durable), `power-state.log`, `orpheus-server.log`, `watchdog.log`, `tts-capture/`, and any `core.*`. After a panic, the next launch auto-harvests the kernel's crash dumps (from `/sys/fs/pstore` *and* `/var/lib/systemd/pstore`, which races it) into `athena-diag/crash-<ts>/` — the full panic backtrace lands in `…/dmesg.txt`.


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

**Automatic runtime tuning (the launcher).** Applied each run and reverted on exit:

- CPU governor → `performance`; **EPP** → `performance` (on `intel_pstate` the governor alone does not pin EPP); the active power-profile manager (`power-profiles-daemon`/`tuned`) is neutralized for the session — it can pin a degraded P-state invisible to sysfs checks (a documented 20–30 % loss).
- Deep C-states disabled (best effort), transparent huge pages → `always`, NUMA balancing off.
- GPU: persistence mode on; `nvidia-powerd` verified active (an unmanaged Dynamic Boost is a documented fault trigger on this platform). **Clock/power set-calls are deliberately not attempted by default** — on this platform the SBIOS power handshake fails at boot (`PlatformRequestHandler … NV_ERR_INVALID_DATA`, a known cross-vendor 595.x laptop issue) and `nvidia-smi -pl/-lmc` requests are rejected or ignored; the opt-in `ATHENA_LOCK_MEMCLK` / `ATHENA_GPU_POWER_LIMIT` / `ATHENA_GPU_CLOCK_CAP` levers remain for A/B testing.

All require `sudo` and are best-effort.


## Hardware Configuration (reference: ThinkPad P16 Gen 3)

| | Spec |
|-|------|
| CPU | Intel Core Ultra 9 285HX (24 cores / 24 threads — no SMT) |
| RAM | 192 GB DDR5-4000 |
| GPU | NVIDIA RTX PRO 5000 Blackwell Laptop, 24 GB GDDR7, 896 GB/s, **110 W firmware cap (95 W default TGP)** |
| LLM | Qwen3.5-397B-A17B UD-Q3_K_XL, 80,000-token context (experts offloaded to RAM via `--cpu-moe`) |

### VRAM budget (397B-A17B, `--cpu-moe`)

The expert FFN tensors — the bulk of the ~179 GB model — live in **system RAM** (measured ~161 GB resident), not VRAM. The GPU holds the dense weights, KV cache, GatedDeltaNet recurrent state, and the audio stack:

| Component | VRAM |
|-----------|------|
| Qwen dense (non-expert) weights (Q3_K) | 8.92 GB (measured) |
| Qwen KV bf16 (80 K ctx, 15 full-attn layers) | ~2.3 GB (scales with `--ctx-size`) |
| GatedDeltaNet recurrent state | 0.18 GB (measured) |
| Orpheus UD-Q4_K_XL weights | ~2.0 GB |
| Orpheus KV **f16** (c=13824, 2 slots) | ~1.6 GB |
| Whisper small.en | ~0.65 GB |
| SNAC decoder (CUDA EP) | ~1.7 GB session peak |
| Compute buffers + CUDA overhead | ~2.4 GB |
| **Total** | **~19 GB of 24 (~4–5 GB free)** |

The Qwen KV must stay **bf16** — a quantized KV degraded output on this model (a literal `<think>` token by turn 3, Unsloth's documented symptom). OOM levers: `--ctx-size` and the Orpheus quant — never the experts, which don't touch VRAM.

### GPU power notes (measured 2026-06/07)

The platform enforces a 95 W default TGP (110 W firmware max). `nvidia-powerd` (Dynamic Boost) is active and holds an effective ~105 W ceiling; because the SBIOS handshake fails on this platform, **`nvidia-smi -pl` requests are rejected and `-lmc` locks don't hold** — telemetry-verified. Under sustained decode the GPU rides the SW power cap (throttle reason `0x4`) at ~94 W. Bench-verified consequences:

- Graphics clock deliberately **not** locked (unlocked 167.2 tok/s beat every fixed lock).
- Memory clock left to the driver (it runs the 13801 top bin under load regardless; the once-recommended lock is a no-op here and defaults off).
- `GGML_CUDA_GRAPH_OPT` **removed everywhere** (experimental buffer-reuse race, no measured benefit); base CUDA graphs stay on (nsys: exactly 1 `cudaGraphLaunch` per token).
- Orpheus KV f16 vs q8_0: latency within noise; **f16 kept** for clearly better vocal quality.


## Performance (measured)

### SNAC / Orpheus token facts

SNAC 24 kHz is **multi-rate**: three codebooks at 11.7 + 23.4 + 46.9 Hz = **82 tokens per second of audio**. Each 7-token Orpheus frame (interleaved `[c0, c1a, c2a, c2b, c1b, c2c, c2d]`) spans **2048 samples = 85.3 ms**; frame rate 11.72 fps. Audio codes are `<custom_token_N>` with offset 10.

### Throughput (Orpheus UD-Q4_K_XL on the P16)

| Configuration | tok/s | × real-time (82 tok/s) |
|---------------|------:|------------------------:|
| single stream | 160–167 | 1.95–2.04 |
| **2 parallel (launcher, `-np 2`)** | 245 | 2.99 |
| 3 parallel | 299 | 3.65 |
| 4 parallel | 350 | 4.27 |

The launcher pins `-np 2`: a single user gains zero latency from more slots, per-stream real-time margin grows (~1.5× vs ~1.07× at 4), and peak concurrent decode — shared MPS stream pressure — halves. Single-stream ceiling is set by the power wall, not software.

### Qwen LLM decode (397B-A17B, `--cpu-moe`)

Decode is bound by **system memory bandwidth** (~3.07 GB of routed-expert reads per token). Field A/B: `-t 16` + `taskset -c 0-15` measured **~6.4 tok/s** (+30 % over `-t 8`); cores 16–23 are left free to absorb audio/daemon wakeups.

### Latency (streaming pipeline, from logs)

| Metric | Measured |
|--------|----------|
| Per-chunk first audio (POST → first PCM) | ~190–410 ms |
| Session first audio (incl. 350 ms watermark) | ~310–430 ms |
| VAD hangover (end-of-speech confirmation) | ~400 ms (+ ~350 ms detector floor) |
| Whisper STT | ~50–100 ms |
| End-of-speech → first audible response | **~2.2–2.5 s** end to end |


## Troubleshooting

| Problem | Solution |
|---------|----------|
| "no audio tokens in response" | Verify llama-server is running and the Orpheus model loaded |
| TTS goes silent mid-session | Check `<run>/watchdog.log` — a llama-server crash is auto-restarted within seconds; if restarts hit the cap, the backend is hard-down (see kernel requirement) |
| `chunk N: first audio` never prints (`--stream-tts`) | SSE format mismatch — capture one raw `curl -N` streaming response; fall back with `--no-stream-tts` |
| First utterance chops/stutters | `stall silence` in the session summary should be 0; raise `--prebuffer-ms` (500) if not |
| Audio doubled at end of a chunk | The `"stop":true` SSE guard needs widening for your server build |
| Reply cuts off right after Athena says your name | Should not happen on current builds (newline-anchored turn tags); if it does, verify the §14 talk-llama fix is in your binary |
| Sentences clipped mid-pause | Raise `--vad-window-ms`, then `--vad-last-ms` |
| Slow to respond | Lower `--vad-window-ms` (500) and `--vad-last-ms` (300) |
| CUDA EP unavailable | Install cuDNN 9 **full**, add the ORT lib path to `ld.so.conf.d`, `sudo ldconfig` |
| SNAC decoder init failed | Delete the `.optimized` cache; orpheus-speak retries on CPU automatically and prints the real ORT error |
| MPS didn't engage (3 clients time-slicing) | `mps-log/` empty in the run dir is the tell; the launcher liveness-probes and self-checks (some distros omit `/usr/sbin` from the user PATH for the MPS server — the launcher prepends it) |
| `Bad page state` / `pagealloc: memory corruption` / Xid 69 / kernel panics | **Kernel bug, not Athena** — run a CVE-2026-43303-fixed kernel; see [Stability](#stability-diagnostics--crash-capture) and `STABILITY-RUNBOOK.md`; panic backtraces auto-harvest to `athena-diag/crash-<ts>/` |
| OOM | Reduce `--ctx-size`, drop Orpheus to Q4_K_M, recheck the VRAM budget |
| speak-daemon.sh hangs (legacy path) | Check orpheus-speak is running and the three file paths match the launcher's |


## Files

| File | Purpose |
|------|---------|
| `launch-athena-397b.sh` | Orchestrator: GPU/CPU tuning, MPS, preflight, watchdog + keepalive, crash harvest, ordered teardown |
| `Athena 397B.desktop` | One-click desktop entry (timestamped logging) |
| `orpheus/orpheus-speak.cpp` | TTS daemon: SSE ingestion, sliding-window SNAC decode, watermark + write-clock playback, capture instrumentation, CPU fallback |
| `patches/talk-llama/` | **Source of truth** for the 5 files patched into `whisper.cpp/examples/talk-llama/`: `talk-llama.cpp` (Athena persona, MoE offload, Silero VAD + endpointing, streaming, emotion tagging, memory glue, barge-in, brain fail-safe, anchored turn tags), `athena_memory.h` (Ebbinghaus memory + personality), `silero-endpointer.h` + `silero-turn-state.h` (streaming end-of-turn detection), `CMakeLists.txt` (ORT wiring + conditional DSA source) |
| `speak-daemon.sh` | Legacy batch bridge (dormant) |
| `athena-gpu-monitor.sh` | Telemetry sampler (persistent nvidia-smi streams + kmsg capture) |
| `athena-kmsg-logger.cpp` | Durable unfiltered kernel-log capture (fdatasync per line) |
| `apply-gsp-stability.sh` + `zzz-nvidia-athena-gsp-stability.conf` + `athena-uvm-hardening.conf` | Driver hardening installs |
| `apply-crash-capture.sh` + `athena-crash-capture.sysctl.conf` + `zzz-athena-gsp-diag.conf` | Panic-capture arming |
| `check-kernel-fix.sh` | CVE-2026-43303 kernel-fix watcher |
| `athena-autostart.desktop` | Optional login auto-relaunch (staged; see header caveats) |
| `models/emotion2vec-reexport.py` | One-time emotion2vec → ONNX export (dynamic axis, validated vs FunASR) |
| `STABILITY-RUNBOOK.md` | Step-by-step stability procedure (kernel swap, driver update) |
| `CHANGES.MD` | Session-by-session engineering log — the ground truth for what changed and why |
| `CLAUDE.md` | Repo guide: architecture ground truth, build commands, gotchas |
| `README.md` | This file |


## Roadmap

- **Latency-masked fillers.** Emit a near-instant filler ("hmm," "okay so") while the full response generates, varied by detected emotion and complexity.
- **Paralinguistic mirroring.** Extend emotion2vec with speaking rate and vocal arousal so Athena matches the user's energy.
- **Output prosody from the emotional trajectory.** Condition Orpheus's expressive tags on the running emotional state through the system prompt.
- **Full-stack auto-relaunch.** The optional autostart entry exists; unattended relaunch still needs targeted sudo NOPASSWD rules (see `athena-autostart.desktop`).


## License

<!-- ATHENA-NCRD-LICENSE-START -->
**Athena is source-available for permitted noncommercial use.**
Covered original contributions use the
[Athena Noncommercial, Attribution and Responsible Deployment License, version 2.0](LICENSE).
Personal use, qualifying noncommercial research, updates, modification and free
redistribution are permitted under its conditions. Commercial use, including
internal business use, commercial R&D, paid services and monetized deployments,
is not licensed. There is no automatic commercial conversion or buyout.

**Original Athena contributions by Igor Barshteyn.**
Third-party components belong to their respective authors.
Source: https://github.com/igorbarshteyn/athena
Preserve this credit for Igor-authored Covered Material, applicable upstream
notices, and notices identifying Your changes. See [NOTICE](NOTICE).

The license prohibits deliberate mistreatment and requires responsible, lawful
deployment. It preserves personal control, ordinary shutdown, privacy deletion,
repair and proportionate noncommercial safety research. It does not require
agreement that Athena is conscious or give an AI authority to waive these terms.

**Read LICENSE Sections 12-14:** warranty disclaimers, liability exclusions and
limits, and a defense/indemnification obligation benefiting Igor Barshteyn and
other Protected Parties, subject to stated exclusions and mandatory law.
Contractual duties require legally effective acceptance. An ordinary guest does
not become an indemnitor merely by talking to someone else's deployment.

**Earlier valid licenses and third-party rights remain in force.** This license
does not retract older MIT permissions or relicense models, NVIDIA software or
upstream code. Read the [scope](LICENSING-SCOPE.md),
[transition](LICENSING-TRANSITION.md), [software notices](THIRD-PARTY-NOTICES.md),
[model licenses](MODEL-LICENSES.md), [deployment guide](DEPLOYMENT-COMPLIANCE.md),
[FAQ](LICENSE-FAQ.md), and [acceptance guidance](licensing/ACCEPTANCE-AND-RECORDS.md).
<!-- ATHENA-NCRD-LICENSE-END -->
