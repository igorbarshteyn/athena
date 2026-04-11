The headline: on the P16, Orpheus produces audio at **7x realtime** — the pipeline can physically never starve. Every architectural workaround we built for the T15g (PCM buffering, MIN_BUFFER=3, self-correcting batch growth) becomes nearly unnecessary because the GPU is fast enough to stay permanently ahead of playback.

The practical impact on conversation flow:

**Short responses** ("How are you?"): ~3-4 seconds from user input to hearing Athena speak. That's approaching human conversational latency. On the T15g it's ~8s.

**Long responses** (quantum essay): ~12-15 seconds to first audio, then a single unbroken stream. On the T15g it's ~48s to first audio with inter-batch boundaries every 30-180s.

**The bottleneck shifts entirely to Qwen.** At 10-14 tok/s for the 397B model, Qwen becomes the pacing element — it takes ~2-3s to generate each sentence, and Orpheus can render that sentence to audio in ~1.4s. The pipeline is always waiting for Qwen, never for Orpheus. That's the right place for the bottleneck — you can't speak what hasn't been thought.

The 7-11 GB VRAM headroom also opens the door to `-np 4` (4 Orpheus slots), which combined with `MIN_BUFFER=2` would bring first-audio for long responses down to ~8-10s. And the 131K context window means Athena can maintain coherent conversations for hours.