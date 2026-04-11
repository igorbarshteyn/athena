#!/usr/bin/env bash
# speak-daemon.sh — talk-llama --speak wrapper for orpheus-speak --watch mode
#
# Setup (two terminals):
#
#   Terminal 2 (daemon — start second, leave running):
#     /home/user/Orpheus/build/orpheus-speak --watch /tmp/speak_tts.txt --snac snac24_dynamic_fp16.onnx --play "aplay -q" -v -o /home/user/output.wav
#
#   Terminal 3 (talk-llama):
#     /home/user/whisper.cpp/build/bin/whisper-talk-llama -ml /home/user/models/QWEN-3.5-9B/Qwen3.5-9B-UD-Q6_K_XL.gguf -mw /home/user/models/Whisper/ggml-small.bin --temp 1.0 --top-p 0.8 --top-k 20 --min-p 0.00 -t 4 -ngl 99 -s /home/user/models/Whisper/speak-daemon.sh -sf /home/user/speakfile.temp -p Igor -bn Qwen -mt 64 -vms 15000 --presence-penalty 1.5 --repeat-penalty 1.0 -ctk q8_0 -ctv q8_0; rm /home/user/speakfile.temp
#
#   Terminal 1 (llama-server - start first, leave running):
#      /home/user/llama.cpp/build/bin/llama-server -m /home/user/models/Orpheus/orpheus-3b-0.1-ft-Q8_0.gguf -c 8192 -ngl 99 --host 127.0.0.1 --port 8080 --cache-type-k q8_0 --cache-type-v q8_0 -fa on
#
# MUCH faster than speak.sh because the ONNX session stays warm.

SPEAK_FILE="/home/user/speakfile.temp"
TRIGGER_FILE="/home/user/speak_tts.txt"
DONE_FILE="/home/user/speak_tts.done"

if [ ! -f "$SPEAK_FILE" ]; then
    exit 0
fi

# Remove any stale done flag
rm -f "$DONE_FILE"

# Sanitize and write to trigger file (atomic via temp + mv)
TMPF=$(mktemp)
sed -E \
    -e 's/\*\*([^*]*)\*\*/\1/g' \
    -e 's/\*([^*]*)\*/\1/g' \
    -e 's/`[^`]*`//g' \
    -e 's/^#{1,6} //g' \
    -e 's/https?:\/\/[^ ]*//g' \
    -e 's/\[([^]]*)\]\([^)]*\)/\1/g' \
    -e 's/<\|[a-z_]*\|>//g' \
    "$SPEAK_FILE" > "$TMPF"

if [ ! -s "$TMPF" ]; then
    rm -f "$TMPF"
    exit 0
fi

mv -f "$TMPF" "$TRIGGER_FILE"

# Wait for daemon to finish (it creates .done after playback)
# Timeout after 1800s (30 min) to avoid hanging forever (long stories need time)
for i in $(seq 1 36000); do
    if [ -f "$DONE_FILE" ]; then
        rm -f "$DONE_FILE"
        exit 0
    fi
    sleep 0.05
done

echo "[speak-daemon.sh] WARNING: daemon did not signal completion" >&2
