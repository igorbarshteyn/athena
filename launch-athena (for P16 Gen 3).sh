#!/usr/bin/env bash
# launch-athena.sh — One-click launcher for the Athena voice assistant
#
# Starts all three processes (llama-server, orpheus-speak, talk-llama),
# waits for Ctrl+C, then shuts everything down and cleans up temp files.
#
# Suitable for launching from a desktop shortcut.
#
# Hardware: ThinkPad P16 Gen 3
#   CPU:  Intel Core Ultra 9 285HX (8P + 16E = 24 cores / 32 threads)
#   RAM:  192 GB DDR5-4000
#   GPU:  NVIDIA RTX PRO 5000 Blackwell Laptop 24 GB GDDR7 (896 GB/s)
#
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ── Paths (edit these to match your setup) ────────────────────────────────────

LLAMA_SERVER="/home/user/llama.cpp/build/bin/llama-server"
ORPHEUS_SPEAK="/home/user/Orpheus/build/orpheus-speak"
TALK_LLAMA="/home/user/whisper.cpp/build/bin/whisper-talk-llama"

ORPHEUS_MODEL="/home/user/models/Orpheus/orpheus-3b-0.1-ft-UD-Q4_K_XL.gguf"
SNAC_MODEL="/home/user/Orpheus/snac24_dynamic_fp16.onnx"
QWEN_MODEL="/home/user/models/QWEN-3.5-397B/Qwen3.5-397B-A17B-UD-Q3_K_XL-00001-of-00006.gguf"
WHISPER_MODEL="/home/user/models/Whisper/ggml-small.bin"
SPEAK_DAEMON="/home/user/models/Whisper/speak-daemon.sh"

# Temp files
SPEAK_FILE="/home/user/speakfile.temp"
TRIGGER_FILE="/home/user/speak_tts.txt"
DONE_FILE="/home/user/speak_tts.done"
WAV_FILE="/dev/shm/orpheus_tts.wav"
OPT_CACHE="${SNAC_MODEL}.optimized"

# PIDs for cleanup
PID_LLAMA=""
PID_ORPHEUS=""
PID_TALK=""

# Track whether GPU clocks were successfully locked (for cleanup)
GPU_CLOCKS_LOCKED=false

# ── GPU performance setup ─────────────────────────────────────────────────────

setup_gpu_performance() {
    echo "[launch-athena] configuring GPU for maximum performance..."

    if ! command -v nvidia-smi &>/dev/null; then
        echo "[launch-athena] WARNING: nvidia-smi not found, skipping GPU tuning"
        return
    fi

    # Enable persistence mode
    sudo nvidia-smi -pm 1 2>/dev/null && echo "  persistence mode: enabled"

    # Lock GPU clocks to maximum
    local max_gfx max_mem
    max_gfx=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    max_mem=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')

    if [[ -n "$max_gfx" && "$max_gfx" != "[N/A]" ]]; then
        if sudo nvidia-smi -lgc "$max_gfx" 2>/dev/null; then
            GPU_CLOCKS_LOCKED=true
            echo "  GPU clocks locked: ${max_gfx} MHz"
        fi
    fi

    if [[ -n "$max_mem" && "$max_mem" != "[N/A]" ]]; then
        sudo nvidia-smi -lmc "$max_mem" 2>/dev/null && echo "  memory clocks locked: ${max_mem} MHz"
    fi

    # Attempt max power limit (blocked on laptop GPUs — harmless if it fails)
    local max_power
    max_power=$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "$max_power" && "$max_power" != "[N/A]" ]]; then
        sudo nvidia-smi -pl "$max_power" 2>/dev/null
    fi

    echo "[launch-athena] GPU configured."
}

restore_gpu_performance() {
    echo "  restoring GPU to normal..."

    if ! command -v nvidia-smi &>/dev/null; then return; fi

    if $GPU_CLOCKS_LOCKED; then
        sudo nvidia-smi -rgc 2>/dev/null && echo "    GPU clocks: reset"
        sudo nvidia-smi -rmc 2>/dev/null && echo "    memory clocks: reset"
    fi

    sudo nvidia-smi -pm 0 2>/dev/null && echo "    persistence mode: disabled"
}

# ── CPU / system performance setup ────────────────────────────────────────────

setup_system_performance() {
    echo "[launch-athena] configuring system for maximum performance..."

    # CPU governor → performance
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance | sudo tee "$gov" >/dev/null 2>&1
    done
    echo "  CPU governor: performance"

    # Disable deep C-states
    for state in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
        echo 1 | sudo tee "$state" >/dev/null 2>&1
    done
    echo "  C-states: disabled"

    # Enable transparent huge pages
    echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null 2>&1
    echo always | sudo tee /sys/kernel/mm/transparent_hugepage/defrag >/dev/null 2>&1
    echo "  transparent huge pages: always"

    # Disable NUMA balancing
    echo 0 | sudo tee /proc/sys/kernel/numa_balancing >/dev/null 2>&1
    echo "  NUMA balancing: disabled"

    echo "[launch-athena] system configured."
}

restore_system_performance() {
    echo "  restoring system to normal..."

    # CPU governor → powersave (default on most Linux distros)
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo powersave | sudo tee "$gov" >/dev/null 2>&1
    done
    echo "    CPU governor: powersave"

    # Re-enable C-states
    for state in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
        echo 0 | sudo tee "$state" >/dev/null 2>&1
    done
    echo "    C-states: re-enabled"

    # THP → madvise (default on most Linux distros)
    echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null 2>&1
    echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag >/dev/null 2>&1
    echo "    transparent huge pages: madvise"

    # Re-enable NUMA balancing
    echo 1 | sudo tee /proc/sys/kernel/numa_balancing >/dev/null 2>&1
    echo "    NUMA balancing: enabled"
}

# ── Cleanup function ──────────────────────────────────────────────────────────

cleanup() {
    set +e  # Don't exit on error during cleanup
    echo ""
    echo "[launch-athena] shutting down..."

    # Kill in reverse order (SIGTERM first, then SIGKILL if stuck)
    for pid_var in PID_TALK PID_ORPHEUS PID_LLAMA; do
        pid="${!pid_var}"
        [ -z "$pid" ] && continue
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            # Wait up to 3 seconds for graceful exit
            for i in $(seq 1 30); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.1
            done
            # Force kill if still alive
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null
            fi
            echo "  stopped $pid_var ($pid)"
        fi
    done

    # Remove temp files
    rm -f "$SPEAK_FILE" "$TRIGGER_FILE" "$DONE_FILE" "$WAV_FILE"
    echo "  cleaned up temp files"

    # Restore system and GPU to pre-launch state
    restore_system_performance
    restore_gpu_performance

    echo "[launch-athena] done."
}

trap cleanup EXIT INT TERM

# ── GPU performance tuning ────────────────────────────────────────────────────

setup_gpu_performance

# ── CPU / system performance tuning ───────────────────────────────────────────

setup_system_performance

# ── Preflight checks ──────────────────────────────────────────────────────────

check_file() {
    if [ ! -f "$1" ]; then
        echo "[launch-athena] ERROR: $2 not found: $1"
        exit 1
    fi
}

check_file "$LLAMA_SERVER"  "llama-server binary"
check_file "$ORPHEUS_SPEAK" "orpheus-speak binary"
check_file "$TALK_LLAMA"    "whisper-talk-llama binary"
check_file "$ORPHEUS_MODEL" "Orpheus GGUF model"
check_file "$SNAC_MODEL"    "SNAC ONNX decoder"
check_file "$QWEN_MODEL"    "Qwen3.5 GGUF model"
check_file "$WHISPER_MODEL" "Whisper model"
check_file "$SPEAK_DAEMON"  "speak-daemon.sh"

# Clean stale temp files from previous runs
rm -f "$SPEAK_FILE" "$TRIGGER_FILE" "$DONE_FILE"

# Remove stale ONNX optimized graph cache — a GPU-optimized cache will cause
# VRAM allocation even when running SNAC on CPU via --snac-cpu
rm -f "$OPT_CACHE"

# ── Terminal 1: llama-server (Orpheus TTS) ────────────────────────────────────

echo "[launch-athena] starting llama-server (Orpheus TTS)..."

"$LLAMA_SERVER" \
    -m "$ORPHEUS_MODEL" \
    -c 9216 \
    -np 2 \
    -ngl 99 \
    --host 127.0.0.1 \
    --port 8080 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --cache-ram 0 \
    -fa on \
    --no-warmup \
    -t 0 \
    >/dev/null 2>&1 &
PID_LLAMA=$!

# Wait for llama-server to be ready
echo "[launch-athena] waiting for llama-server to start..."
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:8080/health >/dev/null 2>&1; then
        echo "[launch-athena] llama-server ready."
        break
    fi
    if ! kill -0 "$PID_LLAMA" 2>/dev/null; then
        echo "[launch-athena] ERROR: llama-server exited unexpectedly"
        exit 1
    fi
    sleep 0.5
done

if ! curl -s http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "[launch-athena] ERROR: llama-server did not start within 60s"
    exit 1
fi

# ── Terminal 2: orpheus-speak daemon (SNAC decoder) ───────────────────────────

echo "[launch-athena] starting orpheus-speak daemon..."

"$ORPHEUS_SPEAK" \
    --watch "$TRIGGER_FILE" \
    --snac "$SNAC_MODEL" \
    --play "aplay -q -D pulse" \
    -v &
PID_ORPHEUS=$!

# Give it a moment to load the ONNX model
sleep 2

if ! kill -0 "$PID_ORPHEUS" 2>/dev/null; then
    echo "[launch-athena] ERROR: orpheus-speak exited unexpectedly"
    exit 1
fi

# ── Terminal 3: talk-llama (voice assistant) ──────────────────────────────────

echo "[launch-athena] starting talk-llama..."
echo "[launch-athena] ═══════════════════════════════════════════════════"
echo "[launch-athena] Athena is ready. Speak into your microphone."
echo "[launch-athena] Press Ctrl+C to quit."
echo "[launch-athena] ═══════════════════════════════════════════════════"
echo ""

export SDL_AUDIODRIVER=pulse

"$TALK_LLAMA" \
    -ml "$QWEN_MODEL" \
    -mw "$WHISPER_MODEL" \
    --ctx-size 131072 \
    --mlock \
    --cpu-moe \
    -t 23 -ngl 99 \
    -ctk bf16 -ctv bf16 -fa \
    --temp 1.0 --top-p 0.8 --top-k 20 --min-p 0.00 \
    --presence-penalty 1.2 --repeat-penalty 1.0 \
    -s "$SPEAK_DAEMON" -sf "$SPEAK_FILE" \
    -p Igor -bn Athena \
    -mt 128 -vms 15000 \
    --vad-last-ms 600

# If talk-llama exits on its own, cleanup runs via the trap
