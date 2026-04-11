// orpheus-speak.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Minimal C++ CLI that turns text into speech via:
//   1. llama-server  (Orpheus 3B GGUF on GPU)  → SNAC audio tokens
//   2. ONNX Runtime  (SNAC 24 kHz decoder)      → 24 kHz PCM → WAV file
//
// Zero Python dependencies.  Requires libcurl + onnxruntime C API.
//
// Usage:
//   orpheus-speak [options] "Text to speak"
//   orpheus-speak [options] -f input.txt
//   orpheus-speak [options] -f input.txt -o output.wav
//
// Designed to be used as the --speak command for whisper.cpp/talk-llama.
// ─────────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <libgen.h>
#include <sys/inotify.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <curl/curl.h>
#include <onnxruntime_c_api.h>

// ─────────────────────────────────────────────────────────────────────────────
// Configuration defaults (override via CLI flags)
// ─────────────────────────────────────────────────────────────────────────────
struct Config {
    std::string api_url       = "http://127.0.0.1:8080/completion";
    std::string snac_model    = "snac24_decoder.onnx";  // SNAC ONNX decoder path
    std::string voice         = "tara";
    std::string output_wav    = "/dev/shm/orpheus_tts.wav"; // RAM-backed tmpfs — no disk IO
    std::string input_file;                              // -f <file>
    std::string text;                                    // positional text
    std::string play_cmd      = "";                      // e.g. "aplay" or "ffplay -nodisp -autoexit"
    std::string watch_file;                              // --watch <file>: persistent daemon mode
    float       temperature   = 0.6f;
    float       top_p         = 0.9f;
    float       rep_penalty   = 1.1f;
    int         max_tokens    = 2500;
    int         sample_rate   = 24000;
    bool        verbose       = false;
    bool        snac_cpu      = false;  // force SNAC decoder to CPU (saves ~1.7 GB VRAM)
};

// ─────────────────────────────────────────────────────────────────────────────
// Orpheus special token IDs
// ─────────────────────────────────────────────────────────────────────────────
// The Orpheus tokenizer maps audio codes to custom_token_N where
// N = AUDIO_OFFSET + codebook_layer * 4096 + code_index
// custom_token_0 through custom_token_9 are special (SOH, EOT, etc.)
// Audio codes start at custom_token_10.
// See: https://github.com/canopyai/Orpheus-TTS
static constexpr int ORPHEUS_TOKEN_BASE   = 128266;  // vocab ID of first audio token
static constexpr int ORPHEUS_CODEBOOK_SZ  = 4096;
static constexpr int ORPHEUS_FRAME_TOKENS = 7;       // tokens per SNAC frame
static constexpr int ORPHEUS_AUDIO_OFFSET = 10;      // audio starts at <custom_token_10>

// Special framing tokens (not audio)
static constexpr int TOKEN_SOH = 128259;  // start-of-human
static constexpr int TOKEN_EOT = 128009;  // end-of-text
static constexpr int TOKEN_EOH = 128260;  // end-of-human
static constexpr int TOKEN_SOA = 128261;  // start-of-audio  (custom_token_5 sometimes)

// ─────────────────────────────────────────────────────────────────────────────
// WAV writer  (16-bit PCM, mono)
// ─────────────────────────────────────────────────────────────────────────────
static bool write_wav(const std::string &path, const std::vector<float> &pcm, int sr) {
    const int num_samples  = (int)pcm.size();
    const int bits         = 16;
    const int byte_rate    = sr * bits / 8;
    const int data_bytes   = num_samples * (bits / 8);
    const int file_size    = 36 + data_bytes;

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    // RIFF header
    f.write("RIFF", 4);
    auto write32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto write16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };

    write32(file_size);
    f.write("WAVE", 4);

    // fmt chunk
    f.write("fmt ", 4);
    write32(16);           // chunk size
    write16(1);            // PCM
    write16(1);            // mono
    write32(sr);           // sample rate
    write32(byte_rate);    // byte rate
    write16(bits / 8);     // block align
    write16(bits);         // bits per sample

    // data chunk
    f.write("data", 4);
    write32(data_bytes);

    for (float s : pcm) {
        float clamped = std::max(-1.0f, std::min(1.0f, s));
        int16_t i16   = static_cast<int16_t>(clamped * 32767.0f);
        f.write(reinterpret_cast<const char*>(&i16), 2);
    }

    return f.good();
}

// ─────────────────────────────────────────────────────────────────────────────
// libcurl helpers
// ─────────────────────────────────────────────────────────────────────────────
static size_t curl_write_cb(char *data, size_t size, size_t nmemb, void *userp) {
    auto *buf = static_cast<std::string *>(userp);
    buf->append(data, size * nmemb);
    return size * nmemb;
}

// Build the Orpheus prompt format:
//   <custom_token_3><|begin_of_text|>{voice}: {text}<|eot_id|>
//   <custom_token_4><custom_token_5><custom_token_1>
//
// These map to token IDs in the Orpheus/Llama tokenizer:
//   <custom_token_3> = 128259 (start-of-human)
//   <|begin_of_text|> = standard Llama BOS
//   <|eot_id|>       = 128009 (end-of-text)
//   <custom_token_4> = 128260 (end-of-human)
//   <custom_token_5> = 128261 (start-of-audio)
//   <custom_token_1> = 128257 (audio generation marker)
static std::string build_orpheus_prompt(const std::string &voice, const std::string &text) {
    return "<custom_token_3><|begin_of_text|>" + voice + ": " + text +
           "<|eot_id|><custom_token_4><custom_token_5><custom_token_1>";
}

// Build the JSON body for /completion
static std::string build_request_json(const Config &cfg, const std::string &prompt) {
    // Manual JSON construction to avoid external dependency.
    // Escape any quotes/backslashes in the prompt.
    std::string escaped;
    for (char c : prompt) {
        if (c == '"')       escaped += "\\\"";
        else if (c == '\\') escaped += "\\\\";
        else if (c == '\n') escaped += "\\n";
        else if (c == '\r') escaped += "\\r";
        else if (c == '\t') escaped += "\\t";
        else                escaped += c;
    }

    std::ostringstream js;
    js << "{"
       << "\"prompt\":\"" << escaped << "\","
       << "\"n_predict\":" << cfg.max_tokens << ","
       << "\"temperature\":" << cfg.temperature << ","
       << "\"top_p\":" << cfg.top_p << ","
       << "\"repeat_penalty\":" << cfg.rep_penalty << ","
       << "\"stop\":[\"<|eot_id|>\"],"
       << "\"parse_special\":true,"
       << "\"stream\":false"
       << "}";
    return js.str();
}

// POST to llama-server and return the raw response body.
// Uses thread_local CURL handle — allocated once per thread, reused across
// calls.  Keeps TCP connection alive to llama-server and avoids per-request
// handle allocation (~0.1-0.2ms savings per call, compounds over 10+ chunks).
static bool http_post(const std::string &url, const std::string &body,
                      std::string &response, bool verbose) {
    thread_local CURL *curl = nullptr;
    thread_local struct curl_slist *headers = nullptr;

    if (!curl) {
        curl = curl_easy_init();
        if (!curl) { std::cerr << "[orpheus-speak] curl_easy_init failed\n"; return false; }
        headers = curl_slist_append(nullptr, "Content-Type: application/json");
        headers = curl_slist_append(headers, "Connection: keep-alive");
    } else {
        curl_easy_reset(curl);
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
    curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);      // disable Nagle — lower latency
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);     // keep TCP connection alive

    if (verbose) {
        std::cerr << "[orpheus-speak] POST " << url << "\n";
        std::cerr << "[orpheus-speak] body: " << body.substr(0, 200) << "...\n";
    }

    CURLcode res = curl_easy_perform(curl);

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    if (res != CURLE_OK) {
        std::cerr << "[orpheus-speak] curl error: " << curl_easy_strerror(res) << "\n";
        return false;
    }

    if (http_code != 200) {
        if (verbose)
            std::cerr << "[orpheus-speak] HTTP " << http_code
                      << " (response: " << response.substr(0, 120) << ")\n";
        response.clear();  // treat non-200 as empty
        return true;        // curl succeeded, but server rejected
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Async HTTP generation (thread-safe — no SNAC, no shared state)
// ─────────────────────────────────────────────────────────────────────────────
struct HttpResult {
    std::string response;
    float       server_ms;
    bool        ok;
};

static HttpResult http_generate(const std::string &api_url,
                                const std::string &voice,
                                const std::string &chunk_text,
                                const Config      &cfg,
                                bool               verbose) {
    std::string prompt = build_orpheus_prompt(voice, chunk_text);
    std::string body   = build_request_json(cfg, prompt);

    // Retry loop — server may return 503 when slots are momentarily busy
    // during concurrent submission.  Backoff gives the server time to
    // finish a prior request and free a slot.
    constexpr int max_retries = 3;

    for (int attempt = 0; attempt < max_retries; attempt++) {
        std::string response;

        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok  = http_post(api_url, body, response, verbose && attempt == 0);
        auto t1 = std::chrono::high_resolution_clock::now();

        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        if (ok && !response.empty()) {
            return { std::move(response), ms, true };
        }

        // Failed — retry with backoff (500ms, 1000ms)
        if (attempt + 1 < max_retries) {
            int backoff = 500 * (attempt + 1);
            std::cerr << "[orpheus-speak] slot busy, retry " << attempt + 1
                      << "/" << max_retries << " in " << backoff << "ms\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(backoff));
        }
    }

    return { "", 0, false };
}

// ─────────────────────────────────────────────────────────────────────────────
// Token extraction
// ─────────────────────────────────────────────────────────────────────────────
// Extract custom_token IDs from the LLM response text.
// Orpheus outputs tokens like: <custom_token_28631> <custom_token_5043> ...
// We extract the integer N from each <custom_token_N>.
static std::vector<int> extract_audio_tokens(const std::string &text, bool verbose) {
    std::vector<int> tokens;
    tokens.reserve(512);  // typical response size

    // Hand-rolled scanner — much faster than std::regex for large responses.
    // Matches: <custom_token_DIGITS>
    static const char prefix[] = "<custom_token_";
    static const size_t plen = sizeof(prefix) - 1;

    size_t pos = 0;
    while (pos < text.size()) {
        pos = text.find(prefix, pos);
        if (pos == std::string::npos) break;
        pos += plen;

        // Parse digits
        int tok = 0;
        bool has_digits = false;
        while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
            tok = tok * 10 + (text[pos] - '0');
            has_digits = true;
            pos++;
        }

        // Must end with '>'
        if (!has_digits || pos >= text.size() || text[pos] != '>') continue;
        pos++; // skip '>'

        // Audio tokens start at <custom_token_10> (tokens 0-9 are special)
        if (tok >= ORPHEUS_AUDIO_OFFSET) {
            int code = tok - ORPHEUS_AUDIO_OFFSET;
            if (code < ORPHEUS_FRAME_TOKENS * ORPHEUS_CODEBOOK_SZ) {
                tokens.push_back(code);
            }
        }
    }

    if (verbose) {
        std::cerr << "[orpheus-speak] extracted " << tokens.size()
                  << " audio tokens from response\n";
    }
    return tokens;
}

// Extract the "content" or "text" field value from a JSON response.
// Handles both native /completion format: {"content":"..."}
// and OAI-compatible format: {"choices":[{"text":"..."}]}
static std::string extract_text_from_json(const std::string &json) {
    // Find "content": first (native /completion endpoint), then "text": (OAI compat)
    auto pos = json.find("\"content\":");
    if (pos == std::string::npos) {
        pos = json.find("\"text\":");
        if (pos == std::string::npos) return "";
    }

    // Find the colon after the key, then the opening quote of the value
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    pos++; // skip opening quote

    // Read until unescaped closing quote
    std::string result;
    for (size_t i = pos; i < json.size(); i++) {
        if (json[i] == '\\' && i + 1 < json.size()) {
            i++;
            switch (json[i]) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                default:   result += '\\'; result += json[i]; break;
            }
        } else if (json[i] == '"') {
            break;
        } else {
            result += json[i];
        }
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sentence splitter for chunked TTS
// ─────────────────────────────────────────────────────────────────────────────
// Split text at sentence boundaries so each sentence can be sent to Orpheus
// independently. This avoids attention skip issues on long text and keeps
// each request within the context window.
static std::vector<std::string> split_sentences(const std::string &text) {
    std::vector<std::string> sentences;
    std::string current;

    for (size_t i = 0; i < text.size(); i++) {
        current += text[i];

        bool is_terminal = (text[i] == '.' || text[i] == '?' || text[i] == '!');
        bool at_end      = (i + 1 >= text.size());
        bool next_space  = (!at_end && (text[i + 1] == ' ' || text[i + 1] == '\n'));
        bool at_boundary = is_terminal && (at_end || next_space);

        // Don't split fragments shorter than ~3 words
        if (at_boundary && current.size() > 15) {
            size_t start = current.find_first_not_of(" \n\r\t");
            if (start != std::string::npos) {
                sentences.push_back(current.substr(start));
            }
            current.clear();
        }
    }

    // Remaining text
    if (!current.empty()) {
        size_t start = current.find_first_not_of(" \n\r\t");
        if (start != std::string::npos) {
            std::string remainder = current.substr(start);
            if (!sentences.empty() && remainder.size() < 15) {
                sentences.back() += " " + remainder;
            } else {
                sentences.push_back(remainder);
            }
        }
    }

    if (sentences.empty() && !text.empty()) {
        sentences.push_back(text);
    }

    return sentences;
}

// ─────────────────────────────────────────────────────────────────────────────
// SNAC token deinterleaving
// ─────────────────────────────────────────────────────────────────────────────
// Orpheus produces 7 tokens per SNAC frame in this interleaved order:
//   [L0, L1a, L2a, L2b, L1b, L2c, L2d]
//
// We deinterleave into 3 codebook layers:
//   codes0 (L0):  1 code  per frame  → length = num_frames
//   codes1 (L1):  2 codes per frame  → length = num_frames * 2
//   codes2 (L2):  4 codes per frame  → length = num_frames * 4
//
// Each code value = raw_token_id % 4096  (the codebook index)
// The layer is encoded as raw_token_id / 4096.
struct SnacCodes {
    std::vector<int64_t> codes0;  // [num_frames]
    std::vector<int64_t> codes1;  // [num_frames * 2]
    std::vector<int64_t> codes2;  // [num_frames * 4]
};

static SnacCodes deinterleave_tokens(const std::vector<int> &tokens, bool verbose) {
    SnacCodes codes;
    int num_frames = (int)tokens.size() / ORPHEUS_FRAME_TOKENS;

    if (verbose) {
        std::cerr << "[orpheus-speak] " << tokens.size() << " tokens → "
                  << num_frames << " SNAC frames ("
                  << (float)num_frames / 46.875f << "s audio)\n";
    }

    for (int i = 0; i < num_frames; i++) {
        int base = i * ORPHEUS_FRAME_TOKENS;

        // Layer 0: 1 code per frame
        codes.codes0.push_back(tokens[base + 0] % ORPHEUS_CODEBOOK_SZ);

        // Layer 1: 2 codes per frame (positions 1, 4)
        codes.codes1.push_back((tokens[base + 1] - 1 * ORPHEUS_CODEBOOK_SZ) % ORPHEUS_CODEBOOK_SZ);
        codes.codes1.push_back((tokens[base + 4] - 4 * ORPHEUS_CODEBOOK_SZ) % ORPHEUS_CODEBOOK_SZ);

        // Layer 2: 4 codes per frame (positions 2, 3, 5, 6)
        codes.codes2.push_back((tokens[base + 2] - 2 * ORPHEUS_CODEBOOK_SZ) % ORPHEUS_CODEBOOK_SZ);
        codes.codes2.push_back((tokens[base + 3] - 3 * ORPHEUS_CODEBOOK_SZ) % ORPHEUS_CODEBOOK_SZ);
        codes.codes2.push_back((tokens[base + 5] - 5 * ORPHEUS_CODEBOOK_SZ) % ORPHEUS_CODEBOOK_SZ);
        codes.codes2.push_back((tokens[base + 6] - 6 * ORPHEUS_CODEBOOK_SZ) % ORPHEUS_CODEBOOK_SZ);
    }

    return codes;
}

// ─────────────────────────────────────────────────────────────────────────────
// ONNX Runtime: SNAC decode  (codes → PCM float32)
// ─────────────────────────────────────────────────────────────────────────────
#define ORT_CHECK(expr) do {                                             \
    OrtStatus *_s = (expr);                                              \
    if (_s) {                                                            \
        const char *msg = g_ort->GetErrorMessage(_s);                    \
        std::cerr << "[orpheus-speak] ORT error: " << msg << "\n";       \
        g_ort->ReleaseStatus(_s);                                        \
        return {};                                                       \
    }                                                                    \
} while(0)

// ─────────────────────────────────────────────────────────────────────────────
// Persistent SNAC decoder — init once, decode many times
// ─────────────────────────────────────────────────────────────────────────────
struct SnacDecoder {
    const OrtApi *g_ort = nullptr;
    OrtEnv *env = nullptr;
    OrtSessionOptions *opts = nullptr;
    OrtSession *session = nullptr;
    OrtMemoryInfo *mem_info = nullptr;
    OrtAllocator *allocator = nullptr;
    OrtRunOptions *run_opts = nullptr;      // Fix 1: arena shrinkage per-Run

    const char *in_names[3] = {};
    const char *out_names[1] = {};
    bool is_dynamic = false;
    bool has_cuda_ep = false;
    bool ok = false;

    // Fix 3: pre-allocated padded buffers for dynamic model.
    // Each SNAC codes0 frame produces 2048 audio samples at 24 kHz.
    // MAX_C0=1024 covers ~42 seconds of audio per decode call — well beyond
    // any chunked utterance.  If exceeded, falls back to fresh allocation.
    static constexpr int64_t MAX_C0 = 1024;
    static constexpr int64_t MAX_C1 = MAX_C0 * 2;
    static constexpr int64_t MAX_C2 = MAX_C0 * 4;
    static constexpr int SAMPLES_PER_FRAME = 2048;
    std::vector<int64_t> buf_c0, buf_c1, buf_c2;

    void init(const Config &cfg) {
        g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

        if (g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "orpheus-speak", &env)) return;
        if (g_ort->CreateSessionOptions(&opts)) return;

        g_ort->SetIntraOpNumThreads(opts, 0);
        g_ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);

        // Fix 2: disable CPU memory arena — prevents unbounded growth from
        // dynamic input shapes.  Uses raw malloc/free per tensor instead.
        // The SNAC model is small (~50 MB) and tensors are tiny, so the
        // per-call malloc overhead is negligible vs. compute time.
        g_ort->DisableCpuMemArena(opts);

        // Cache the optimized graph to disk
        std::string opt_path = cfg.snac_model + ".optimized";
        g_ort->SetOptimizedModelFilePath(opts, opt_path.c_str());

        // CUDA EP (skip if --snac-cpu requested)
        if (!cfg.snac_cpu) {
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = 0;
            // kSameAsRequested (1) prevents power-of-two over-allocation.
            // Combined with Fix 3 (fixed shapes), the GPU arena stabilizes
            // after the first Run() and never grows.
            cuda_opts.arena_extend_strategy = 1;
            cuda_opts.do_copy_in_default_stream = 1;
            OrtStatus *cs = g_ort->SessionOptionsAppendExecutionProvider_CUDA(opts, &cuda_opts);
            if (cs) {
                if (cfg.verbose) std::cerr << "[orpheus-speak] CUDA EP unavailable, using CPU\n";
                g_ort->ReleaseStatus(cs);
            } else {
                has_cuda_ep = true;
                if (cfg.verbose) std::cerr << "[orpheus-speak] SNAC using CUDA EP\n";
            }
        } else {
            std::cerr << "[orpheus-speak] SNAC using CPU (--snac-cpu)\n";
        }

        if (g_ort->CreateSession(env, cfg.snac_model.c_str(), opts, &session)) return;

        // Fix 2: use OrtDeviceAllocator (raw malloc/free) instead of
        // OrtArenaAllocator which grows monotonically with dynamic shapes
        if (g_ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &mem_info)) return;
        if (g_ort->GetAllocatorWithDefaultOptions(&allocator)) return;

        // Fix 1: create RunOptions with arena shrinkage for any active arena.
        // CPU arena is disabled by Fix 2, so only request GPU shrinkage when
        // CUDA EP is active.  Requesting shrinkage on a non-existent arena
        // causes ORT to error on every Run() call.
        if (g_ort->CreateRunOptions(&run_opts)) return;
        if (has_cuda_ep) {
            g_ort->AddRunConfigEntry(run_opts,
                "memory.enable_memory_arena_shrinkage", "gpu:0");
        }

        // Read input/output names from model
        char *n0, *n1, *n2, *no;
        g_ort->SessionGetInputName(session, 0, allocator, &n0);
        g_ort->SessionGetInputName(session, 1, allocator, &n1);
        g_ort->SessionGetInputName(session, 2, allocator, &n2);
        g_ort->SessionGetOutputName(session, 0, allocator, &no);
        in_names[0] = n0; in_names[1] = n1; in_names[2] = n2;
        out_names[0] = no;

        // Detect static vs dynamic
        OrtTypeInfo *ti = nullptr;
        g_ort->SessionGetInputTypeInfo(session, 0, &ti);
        const OrtTensorTypeAndShapeInfo *si = nullptr;
        g_ort->CastTypeInfoToTensorInfo(ti, &si);
        size_t dc = 0; g_ort->GetDimensionsCount(si, &dc);
        std::vector<int64_t> dims(dc);
        g_ort->GetDimensions(si, dims.data(), dc);
        g_ort->ReleaseTypeInfo(ti);
        is_dynamic = (dc < 2 || dims[1] <= 0);

        // Fix 3: pre-allocate padded buffers for dynamic model
        if (is_dynamic) {
            buf_c0.resize(MAX_C0, 0);
            buf_c1.resize(MAX_C1, 0);
            buf_c2.resize(MAX_C2, 0);
        }

        if (cfg.verbose) {
            std::cerr << "[orpheus-speak] SNAC model: "
                      << (is_dynamic ? "dynamic" : "static")
                      << ", inputs: " << n0 << ", " << n1 << ", " << n2 << "\n";
        }

        ok = true;
    }

    std::vector<float> decode(const Config &cfg, const SnacCodes &codes) {
        if (!ok) return {};
        std::vector<float> all_pcm;

        auto run_one = [&](int64_t *d0, int64_t n0, int64_t *d1, int64_t n1,
                           int64_t *d2, int64_t n2) {
            int64_t s0[] = {1, n0}, s1[] = {1, n1}, s2[] = {1, n2};
            OrtValue *inputs[3] = {};
            g_ort->CreateTensorWithDataAsOrtValue(mem_info, d0, n0*8, s0, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[0]);
            g_ort->CreateTensorWithDataAsOrtValue(mem_info, d1, n1*8, s1, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[1]);
            g_ort->CreateTensorWithDataAsOrtValue(mem_info, d2, n2*8, s2, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[2]);

            OrtValue *output = nullptr;
            // Fix 1: pass run_opts (with arena shrinkage) instead of nullptr
            OrtStatus *rs = g_ort->Run(session, run_opts, in_names, (const OrtValue *const *)inputs, 3, out_names, 1, &output);
            if (rs) {
                if (cfg.verbose) std::cerr << "[orpheus-speak] ORT Run error: " << g_ort->GetErrorMessage(rs) << "\n";
                g_ort->ReleaseStatus(rs);
                for (auto &v : inputs) if (v) g_ort->ReleaseValue(v);
                return;
            }

            float *pcm = nullptr;
            g_ort->GetTensorMutableData(output, (void**)&pcm);
            OrtTensorTypeAndShapeInfo *info = nullptr;
            g_ort->GetTensorTypeAndShape(output, &info);
            size_t ne = 0; g_ort->GetTensorShapeElementCount(info, &ne);
            g_ort->ReleaseTensorTypeAndShapeInfo(info);

            all_pcm.insert(all_pcm.end(), pcm, pcm + ne);
            g_ort->ReleaseValue(output);
            for (auto &v : inputs) g_ort->ReleaseValue(v);
        };

        if (is_dynamic) {
            int64_t actual_n0 = (int64_t)codes.codes0.size();
            int64_t actual_n1 = (int64_t)codes.codes1.size();
            int64_t actual_n2 = (int64_t)codes.codes2.size();

            if (actual_n0 <= MAX_C0) {
                // Fix 3: use pre-allocated padded buffers — the arena sees
                // the same tensor shape [1, MAX_C0/C1/C2] every time, so no
                // new allocations and no growth.  Zero-pad unused slots;
                // truncate output PCM to match actual input length.
                std::memset(buf_c0.data(), 0, MAX_C0 * sizeof(int64_t));
                std::memset(buf_c1.data(), 0, MAX_C1 * sizeof(int64_t));
                std::memset(buf_c2.data(), 0, MAX_C2 * sizeof(int64_t));
                std::memcpy(buf_c0.data(), codes.codes0.data(), actual_n0 * sizeof(int64_t));
                std::memcpy(buf_c1.data(), codes.codes1.data(), actual_n1 * sizeof(int64_t));
                std::memcpy(buf_c2.data(), codes.codes2.data(), actual_n2 * sizeof(int64_t));
                run_one(buf_c0.data(), MAX_C0, buf_c1.data(), MAX_C1, buf_c2.data(), MAX_C2);
                // Truncate: SNAC produces SAMPLES_PER_FRAME samples per codes0 frame
                int64_t actual_samples = actual_n0 * SAMPLES_PER_FRAME;
                if ((int64_t)all_pcm.size() > actual_samples)
                    all_pcm.resize(actual_samples);
            } else {
                // Fallback for extremely long audio (>42s per chunk)
                std::vector<int64_t> c0(codes.codes0), c1(codes.codes1), c2(codes.codes2);
                run_one(c0.data(), c0.size(), c1.data(), c1.size(), c2.data(), c2.size());
            }
        } else {
            static constexpr int W0 = 12, W1 = 24, W2 = 48, SPW = 24576;
            int total = (int)codes.codes0.size();
            int nwin = (total + W0 - 1) / W0;
            all_pcm.reserve(nwin * SPW);
            for (int w = 0; w < nwin; w++) {
                std::vector<int64_t> c0(W0, 0), c1(W1, 0), c2(W2, 0);
                int o0 = w*W0, o1 = w*W1, o2 = w*W2;
                int n0 = std::min(W0, total - o0);
                int n1 = std::min(W1, (int)codes.codes1.size() - o1);
                int n2 = std::min(W2, (int)codes.codes2.size() - o2);
                for (int i = 0; i < n0; i++) c0[i] = codes.codes0[o0+i];
                for (int i = 0; i < n1; i++) c1[i] = codes.codes1[o1+i];
                for (int i = 0; i < n2; i++) c2[i] = codes.codes2[o2+i];
                size_t prev = all_pcm.size();
                run_one(c0.data(), W0, c1.data(), W1, c2.data(), W2);
                if (w == nwin - 1 && n0 < W0)
                    all_pcm.resize(prev + (int)((float)n0 / W0 * SPW));
            }
        }

        if (cfg.verbose) {
            std::cerr << "[orpheus-speak] SNAC decoded " << all_pcm.size()
                      << " samples (" << (float)all_pcm.size() / cfg.sample_rate
                      << "s)" << (is_dynamic ? " in 1 pass" : "") << "\n";
        }
        return all_pcm;
    }

    void cleanup() {
        if (!g_ort) return;
        if (allocator) {
            for (auto n : in_names) if (n) g_ort->AllocatorFree(allocator, (void*)n);
            if (out_names[0]) g_ort->AllocatorFree(allocator, (void*)out_names[0]);
        }
        if (run_opts) g_ort->ReleaseRunOptions(run_opts);
        if (mem_info) g_ort->ReleaseMemoryInfo(mem_info);
        if (session)  g_ort->ReleaseSession(session);
        if (opts)     g_ort->ReleaseSessionOptions(opts);
        if (env)      g_ort->ReleaseEnv(env);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────
static void usage(const char *argv0) {
    std::cerr
        << "Usage: " << argv0 << " [options] \"text to speak\"\n"
        << "       " << argv0 << " [options] -f input.txt\n"
        << "       " << argv0 << " [options] --watch /tmp/speak.txt  (daemon mode)\n\n"
        << "Options:\n"
        << "  -f FILE        Read input text from FILE\n"
        << "  -o FILE        Output WAV path        [/dev/shm/orpheus_tts.wav]\n"
        << "  --voice NAME   Orpheus voice           [tara]\n"
        << "  --api URL      llama-server URL        [http://127.0.0.1:8080/completion]\n"
        << "  --snac PATH    SNAC ONNX decoder path  [snac24_decoder.onnx]\n"
        << "  --play CMD     Play command after gen   (e.g. 'aplay')\n"
        << "  --watch FILE   Daemon mode: watch FILE for changes, speak each update\n"
        << "                 SNAC session stays alive — much faster after first utterance\n"
        << "  --temp F       Temperature              [0.6]\n"
        << "  --top-p F      Top-p                    [0.9]\n"
        << "  --rep-pen F    Repetition penalty        [1.1]\n"
        << "  --max-tokens N Max tokens to generate   [2500]\n"
        << "  --snac-cpu     Force SNAC decoder to CPU (frees ~1.7 GB VRAM)\n"
        << "  -v, --verbose  Verbose logging\n"
        << "  -h, --help     This message\n\n"
        << "Voices: tara, leah, jess, leo, dan, mia, zac, zoe\n"
        << "Emotion tags: <laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>\n";
}

static Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if ((a == "-f") && i + 1 < argc)             cfg.input_file = argv[++i];
        else if ((a == "-o") && i + 1 < argc)        cfg.output_wav = argv[++i];
        else if ((a == "--voice") && i + 1 < argc)    cfg.voice      = argv[++i];
        else if ((a == "--api") && i + 1 < argc)      cfg.api_url    = argv[++i];
        else if ((a == "--snac") && i + 1 < argc)     cfg.snac_model = argv[++i];
        else if ((a == "--play") && i + 1 < argc)     cfg.play_cmd   = argv[++i];
        else if ((a == "--watch") && i + 1 < argc)    cfg.watch_file = argv[++i];
        else if ((a == "--temp") && i + 1 < argc)     cfg.temperature= std::stof(argv[++i]);
        else if ((a == "--top-p") && i + 1 < argc)    cfg.top_p      = std::stof(argv[++i]);
        else if ((a == "--rep-pen") && i + 1 < argc)  cfg.rep_penalty= std::stof(argv[++i]);
        else if ((a == "--max-tokens") && i + 1 < argc) cfg.max_tokens= std::stoi(argv[++i]);
        else if (a == "-v" || a == "--verbose")       cfg.verbose     = true;
        else if (a == "--snac-cpu")                   cfg.snac_cpu    = true;
        else if (a == "-h" || a == "--help")          { usage(argv[0]); exit(0); }
        else if (a[0] != '-')                         cfg.text        = a;
        else { std::cerr << "Unknown option: " << a << "\n"; usage(argv[0]); exit(1); }
    }
    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio playback via fork/exec (replaces system() to avoid shell overhead)
// ─────────────────────────────────────────────────────────────────────────────
// Splits play_cmd into argv, appends wav_path, forks and execs directly.
// Saves ~5ms per chunk vs system() by avoiding shell process spawning.
// Redirects child stderr to /dev/null (equivalent to 2>/dev/null).
static int play_audio(const std::string &play_cmd, const std::string &wav_path) {
    // Tokenize play_cmd (e.g. "aplay -q -D pulse" → ["aplay", "-q", "-D", "pulse"])
    std::vector<std::string> args;
    std::istringstream iss(play_cmd);
    std::string tok;
    while (iss >> tok) args.push_back(tok);
    args.push_back(wav_path);

    // Build null-terminated argv array for execvp
    std::vector<char *> argv;
    for (auto &a : args) argv.push_back(&a[0]);
    argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid == 0) {
        // Child: redirect stderr to /dev/null, exec player
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
        execvp(argv[0], argv.data());
        _exit(127);  // exec failed
    } else if (pid > 0) {
        // Parent: wait for playback to finish
        int status;
        waitpid(pid, &status, 0);
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
    return -1;  // fork failed
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline: text → (chunked) API → tokens → SNAC → WAV → play
// ─────────────────────────────────────────────────────────────────────────────
// For short text (1-2 sentences): single Orpheus request, same as before.
// For long text (3+ sentences): parallel chunked processing:
//   - All-ahead submission: all chunks submitted as staggered concurrent HTTP requests
//   - Retry with backoff on HTTP 503 (slot busy)
//   - llama-server continuous batching fills up to -np slots in parallel
//   - Background playback: play each chunk via double-buffered WAV as it arrives
// This avoids Orpheus attention-skip on long text and stays within context.
static bool speak_text(const Config &cfg, SnacDecoder &snac, const std::string &text) {
    if (text.empty()) return false;

    auto sentences = split_sentences(text);
    bool use_chunked = (int)text.size() > 300;

    if (cfg.verbose) {
        std::cerr << "[orpheus-speak] voice=" << cfg.voice
                  << " text=\"" << text.substr(0, 80) << (text.size() > 80 ? "..." : "")
                  << "\"" << (use_chunked ? " [chunked: " + std::to_string(sentences.size()) + " sentences]" : "")
                  << "\n";
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<float> all_pcm;

    if (use_chunked) {
        // ── Chunked mode: character-based grouping ─────────────────────
        // Orpheus-3B reliably vocalizes up to ~430 chars per request.
        // Beyond that it probabilistically hits <eot_id> before finishing.
        // Group sentences until adding the next would exceed the limit.
        static constexpr int CHUNK_CHAR_LIMIT = 300;  // safe for 2304 tokens/slot

        std::vector<std::string> chunks;
        std::string current;
        for (const auto &s : sentences) {
            if (!current.empty()
                && (int)(current.size() + 1 + s.size()) > CHUNK_CHAR_LIMIT) {
                chunks.push_back(current);
                current.clear();
            }
            if (!current.empty()) current += " ";
            current += s;
        }
        if (!current.empty()) chunks.push_back(current);

        // ── Parallel generation + pipelined playback ─────────────────
        //
        // Architecture:
        //   - All-ahead submission: submit ALL chunks as concurrent HTTP
        //     requests with a small stagger (200ms) between each.  With
        //     -np N, llama-server's continuous batching processes up to N
        //     in parallel on the GPU.
        //   - Retry with backoff: if a slot is momentarily busy (HTTP 503),
        //     http_generate retries up to 3 times with 500ms/1000ms backoff.
        //   - Background playback: play each chunk individually via
        //     double-buffered WAV as soon as its SNAC decode completes.

        std::string wav_buf[2] = { cfg.output_wav, cfg.output_wav + ".buf" };
        int buf_idx = 0;

        float total_server_ms = 0, total_decode_ms = 0;
        int   total_tokens    = 0;
        size_t total_pcm_samples = 0;

        // Sliding window submission — keep at most max_inflight HTTP
        // requests in flight at once.  With -np 3, this means 3 slots
        // actively generating + 2 queued and ready, so the GPU never
        // idles between chunks.  Prevents overwhelming llama-server
        // with dozens of concurrent connections on long responses
        // (which exhausts the retry budget for late chunks).
        //
        // For short responses (≤ max_inflight chunks), this behaves
        // identically to all-ahead submission.
        static constexpr int max_inflight = 5;  // np=3 active + 2 queued

        std::vector<std::future<HttpResult>> http_futures(chunks.size());
        size_t next_submit = 0;

        // Submit initial window with stagger
        for (; next_submit < std::min(chunks.size(), (size_t)max_inflight); next_submit++) {
            if (next_submit > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            http_futures[next_submit] = std::async(std::launch::async,
                http_generate, cfg.api_url, cfg.voice, chunks[next_submit],
                std::cref(cfg), (next_submit == 0) && cfg.verbose);
        }

        std::thread play_thread;
        auto play_end_time = std::chrono::high_resolution_clock::now();

        // Collect results in order; submit next chunk as each completes
        for (size_t i = 0; i < chunks.size(); i++) {
            if (cfg.verbose) {
                std::cerr << "[orpheus-speak]   chunk " << i + 1 << "/" << chunks.size()
                          << ": \"" << chunks[i].substr(0, 60)
                          << (chunks[i].size() > 60 ? "..." : "") << "\"\n";
            }

            // 1. Wait for this chunk's HTTP response (may already be done)
            HttpResult hr = http_futures[i].get();

            // Submit next chunk now that one result has been collected.
            // The slot that just finished (or is about to finish) will be
            // available for this new request.  No stagger needed — the
            // server is no longer at the initial submission burst.
            if (next_submit < chunks.size()) {
                http_futures[next_submit] = std::async(std::launch::async,
                    http_generate, cfg.api_url, cfg.voice, chunks[next_submit],
                    std::cref(cfg), false);
                next_submit++;
            }

            if (!hr.ok) {
                std::cerr << "[orpheus-speak]   chunk " << i + 1
                          << " HTTP failed after retries, skipping\n";
                continue;
            }

            total_server_ms += hr.server_ms;

            // 2. Parse tokens + SNAC decode (main thread — SNAC not thread-safe)
            std::string gen_text = extract_text_from_json(hr.response);
            if (gen_text.empty()) gen_text = hr.response;

            std::vector<int> audio_tokens = extract_audio_tokens(gen_text, false);
            if (audio_tokens.empty()) continue;

            int usable = ((int)audio_tokens.size() / ORPHEUS_FRAME_TOKENS)
                       * ORPHEUS_FRAME_TOKENS;
            audio_tokens.resize(usable);
            total_tokens += usable;

            SnacCodes codes = deinterleave_tokens(audio_tokens, false);

            auto t_s0 = std::chrono::high_resolution_clock::now();
            std::vector<float> pcm = snac.decode(cfg, codes);
            auto t_s1 = std::chrono::high_resolution_clock::now();
            total_decode_ms += std::chrono::duration<float, std::milli>(t_s1 - t_s0).count();

            if (pcm.empty()) continue;
            total_pcm_samples += pcm.size();

            float audio_ms = (float)pcm.size() / cfg.sample_rate * 1000.0f;

            // 3. Wait for PREVIOUS chunk's playback to finish
            if (play_thread.joinable()) play_thread.join();

            // Drain guard: aplay -D pulse may return before PulseAudio
            // finishes playing the audio.  Sleep until expected end time.
            {
                auto now = std::chrono::high_resolution_clock::now();
                if (now < play_end_time)
                    std::this_thread::sleep_until(play_end_time);
            }

            // 4. Write WAV to current buffer and start background playback
            if (!write_wav(wav_buf[buf_idx], pcm, cfg.sample_rate)) {
                std::cerr << "[orpheus-speak] failed to write: " << wav_buf[buf_idx] << "\n";
                continue;
            }

            if (!cfg.play_cmd.empty()) {
                play_end_time = std::chrono::high_resolution_clock::now()
                              + std::chrono::milliseconds((int)audio_ms + 100);

                std::string wav = wav_buf[buf_idx];
                std::string pcmd = cfg.play_cmd;
                play_thread = std::thread([pcmd, wav]() { play_audio(pcmd, wav); });
            }

            buf_idx = 1 - buf_idx;  // toggle double buffer
        }

        // Wait for final playback to finish
        if (play_thread.joinable()) play_thread.join();
        {
            auto now = std::chrono::high_resolution_clock::now();
            if (now < play_end_time)
                std::this_thread::sleep_until(play_end_time);
        }

        // Clean up secondary WAV buffer
        std::remove(wav_buf[1].c_str());

        auto t_end = std::chrono::high_resolution_clock::now();

        if (cfg.verbose) {
            auto total_ms = std::chrono::duration<float, std::milli>(t_end - t0).count();
            float audio_sec = (float)total_pcm_samples / cfg.sample_rate;
            std::cerr << "[orpheus-speak] pipelined timing:\n"
                      << "  llama-server:  " << total_server_ms << " ms (" << total_tokens << " tokens)\n"
                      << "  SNAC decode:   " << total_decode_ms << " ms\n"
                      << "  total wall:    " << total_ms << " ms\n"
                      << "  audio length:  " << audio_sec * 1000.0f << " ms\n"
                      << "  chunks:        " << chunks.size()
                      << " (from " << sentences.size() << " sentences)\n"
                      << "  pipeline:      sliding window (max " << max_inflight << " in-flight) + play-while-decode\n";
        }

    } else {
        // ── Single-shot mode: one Orpheus request for the whole text ─────
        std::string prompt = build_orpheus_prompt(cfg.voice, text);
        std::string body   = build_request_json(cfg, prompt);
        std::string response;

        if (!http_post(cfg.api_url, body, response, cfg.verbose) || response.empty()) {
            std::cerr << "[orpheus-speak] API call failed\n";
            return false;
        }

        auto t1 = std::chrono::high_resolution_clock::now();

        std::string gen_text = extract_text_from_json(response);
        if (gen_text.empty()) gen_text = response;

        std::vector<int> audio_tokens = extract_audio_tokens(gen_text, cfg.verbose);
        if (audio_tokens.empty()) {
            std::cerr << "[orpheus-speak] no audio tokens in response\n";
            return false;
        }

        int usable = ((int)audio_tokens.size() / ORPHEUS_FRAME_TOKENS) * ORPHEUS_FRAME_TOKENS;
        audio_tokens.resize(usable);

        SnacCodes codes = deinterleave_tokens(audio_tokens, cfg.verbose);
        if (codes.codes0.empty()) return false;

        auto t2 = std::chrono::high_resolution_clock::now();

        all_pcm = snac.decode(cfg, codes);
        if (all_pcm.empty()) return false;

        auto t3 = std::chrono::high_resolution_clock::now();

        if (!write_wav(cfg.output_wav, all_pcm, cfg.sample_rate)) {
            std::cerr << "[orpheus-speak] failed to write: " << cfg.output_wav << "\n";
            return false;
        }

        auto t4 = std::chrono::high_resolution_clock::now();

        if (!cfg.play_cmd.empty()) {
            play_audio(cfg.play_cmd, cfg.output_wav);
        }

        auto t5 = std::chrono::high_resolution_clock::now();

        if (cfg.verbose) {
            auto ms = [](auto a, auto b) {
                return std::chrono::duration<float, std::milli>(b - a).count();
            };
            float audio_sec = (float)all_pcm.size() / cfg.sample_rate;
            std::cerr << "[orpheus-speak] timing breakdown:\n"
                      << "  llama-server:  " << ms(t0, t1) << " ms\n"
                      << "  token parse:   " << ms(t1, t2) << " ms\n"
                      << "  SNAC decode:   " << ms(t2, t3) << " ms\n"
                      << "  WAV write:     " << ms(t3, t4) << " ms\n"
                      << "  playback:      " << ms(t4, t5) << " ms\n"
                      << "  total:         " << ms(t0, t5) << " ms\n"
                      << "  audio length:  " << audio_sec * 1000.0f << " ms\n"
                      << "  RTF:           " << ms(t0, t4) / (audio_sec * 1000.0f) << "x\n";
        }
    }

    return true;
}

// Read and trim a text file
static std::string read_text_file(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs) return "";
    std::ostringstream ss;
    ss << ifs.rdbuf();
    std::string text = ss.str();
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' '))
        text.pop_back();
    return text;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

    curl_global_init(CURL_GLOBAL_DEFAULT);

    // Initialize SNAC decoder once
    SnacDecoder snac;
    snac.init(cfg);
    if (!snac.ok) {
        std::cerr << "[orpheus-speak] SNAC decoder init failed\n";
        return 1;
    }

    if (!cfg.watch_file.empty()) {
        // ── Watch mode: unified streaming protocol ──────────────────────
        // All speech (generation, goodbye, heard_ok) arrives as lines in
        // the trigger file, terminated by a ---END--- sentinel.
        //
        // Protocol:
        //   talk-llama appends "sentence\n" lines during token generation
        //   talk-llama appends "---END---\n" when done
        //   orpheus-speak reads lines incrementally, batches all available
        //   sentences into speak_text() for sliding-window pipelined playback
        //   orpheus-speak signals .done after END, deletes trigger file,
        //   and drains stale inotify events before returning to idle.
        //
        // Robustness guarantees:
        //   - Drain after session prevents stale events from causing
        //     duplicate processing or cross-session contamination
        //   - File deletion ensures clean state for next session
        //   - Single protocol eliminates batch/stream classification bugs
        std::cerr << "[orpheus-speak] watching " << cfg.watch_file
                  << " (SNAC session warm, Ctrl+C to quit)\n";

        // Derive .done path: /home/user/speak_tts.txt → /home/user/speak_tts.done
        std::string done_file = cfg.watch_file;
        auto dot = done_file.rfind('.');
        if (dot != std::string::npos) done_file = done_file.substr(0, dot);
        done_file += ".done";

        // Extract directory and basename for inotify directory watch
        std::string watch_dir, watch_base;
        {
            std::vector<char> dp(cfg.watch_file.begin(), cfg.watch_file.end());
            std::vector<char> bp(cfg.watch_file.begin(), cfg.watch_file.end());
            dp.push_back('\0'); bp.push_back('\0');
            watch_dir  = dirname(dp.data());
            watch_base = basename(bp.data());
        }

        int ifd = inotify_init();
        if (ifd < 0) {
            perror("[orpheus-speak] inotify_init");
            return 1;
        }

        int wd = inotify_add_watch(ifd, watch_dir.c_str(),
                                   IN_MOVED_TO | IN_CLOSE_WRITE);
        if (wd < 0) {
            perror("[orpheus-speak] inotify_add_watch");
            close(ifd);
            return 1;
        }

        char evbuf[4096] __attribute__((aligned(__alignof__(struct inotify_event))));

        while (true) {
            // ── Idle: block until trigger file activity ──────────────────
            int len = read(ifd, evbuf, sizeof(evbuf));
            if (len <= 0) break;

            // Check if the event is for our trigger file, and classify type.
            // IN_MOVED_TO: speak-daemon.sh did atomic mv → batch mode (no END sentinel)
            // IN_CLOSE_WRITE: talk-llama streaming append → streaming mode (expects END)
            bool batch_trigger = false;
            bool stream_trigger = false;
            int pos = 0;
            while (pos < len) {
                auto *ev = reinterpret_cast<struct inotify_event *>(&evbuf[pos]);
                if (ev->len > 0 && watch_base == ev->name) {
                    if (ev->mask & IN_MOVED_TO)    batch_trigger = true;
                    if (ev->mask & IN_CLOSE_WRITE) stream_trigger = true;
                }
                pos += sizeof(struct inotify_event) + ev->len;
            }
            if (!batch_trigger && !stream_trigger) continue;

            // Verify file actually exists (could be stale event after drain)
            {
                struct stat st;
                if (stat(cfg.watch_file.c_str(), &st) != 0) continue;
            }

            if (batch_trigger) {
                // ── Batch mode: speak-daemon.sh wrote complete file via mv ──
                // No ---END--- sentinel — read entire file, process, done.
                // This is the backward-compatible fallback for when
                // --stream-file is not set on talk-llama.
                if (cfg.verbose) {
                    std::cerr << "[orpheus-speak] batch session (speak-daemon.sh)\n";
                }
                std::string text = read_text_file(cfg.watch_file);
                if (!text.empty()) {
                    speak_text(cfg, snac, text);
                }
            } else {

            // ── Streaming: continuous pipeline until ---END--- ──────────
            // Unlike the batch approach (read → speak_text → read → speak_text),
            // this maintains a SINGLE sliding window across the entire session.
            // New sentences are read and chunked into the pipeline between each
            // collect/play cycle, so Orpheus generation for the next chunk
            // overlaps with playback of the current chunk — no gaps.
            if (cfg.verbose) {
                std::cerr << "[orpheus-speak] session started\n";
            }

            static constexpr int max_inflight = 5;
            static constexpr int CHUNK_CHAR_LIMIT = 300;

            // Pipeline state — persists across sentence arrivals
            std::vector<std::string> chunks;
            std::vector<std::future<HttpResult>> http_futures;
            size_t next_submit = 0;
            size_t next_collect = 0;

            // Sentence accumulator for chunking
            std::string pending_chunk;

            // File reading state
            size_t file_pos = 0;
            bool end_received = false;

            // Playback state
            std::string wav_buf[2] = { cfg.output_wav, cfg.output_wav + ".buf" };
            int buf_idx = 0;
            std::thread play_thread;
            auto play_end_time = std::chrono::high_resolution_clock::now();

            // Stats
            float total_server_ms = 0, total_decode_ms = 0;
            int total_tokens = 0;
            size_t total_pcm_samples = 0;
            auto t0 = std::chrono::high_resolution_clock::now();

            // ── Helper: flush pending_chunk into chunks vector ──
            auto flush_pending = [&]() {
                if (!pending_chunk.empty()) {
                    chunks.push_back(pending_chunk);
                    http_futures.resize(chunks.size());
                    pending_chunk.clear();
                }
            };

            // ── Helper: read available lines, chunk sentences ──
            auto read_new_lines = [&]() {
                if (end_received) return;
                std::ifstream ifs(cfg.watch_file);
                if (!ifs) return;
                ifs.seekg(file_pos);
                std::string line;
                while (std::getline(ifs, line)) {
                    file_pos = ifs.tellg();
                    while (!line.empty() && line.back() == '\r') line.pop_back();
                    if (line.empty()) continue;
                    if (line == "---END---") {
                        flush_pending();
                        end_received = true;
                        return;
                    }
                    // Group sentences into chunks ≤ CHUNK_CHAR_LIMIT
                    if (!pending_chunk.empty() &&
                        (int)(pending_chunk.size() + 1 + line.size()) > CHUNK_CHAR_LIMIT) {
                        chunks.push_back(pending_chunk);
                        http_futures.resize(chunks.size());
                        pending_chunk.clear();
                    }
                    if (!pending_chunk.empty()) pending_chunk += " ";
                    pending_chunk += line;
                }
            };

            // ── Helper: submit chunks up to sliding window limit ──
            auto submit_available = [&]() {
                while (next_submit < chunks.size() &&
                       (next_submit - next_collect) < (size_t)max_inflight) {
                    http_futures[next_submit] = std::async(std::launch::async,
                        http_generate, cfg.api_url, cfg.voice,
                        chunks[next_submit], std::cref(cfg),
                        (next_submit == 0) && cfg.verbose);
                    next_submit++;
                }
            };

            // ── Collect-and-decode helper (appends PCM to accumulator) ──
            std::vector<float> pcm_accum;
            float accum_audio_ms = 0;
            int accum_chunks = 0;

            auto collect_one = [&]() -> bool {
                if (next_collect >= next_submit) return false;

                if (cfg.verbose) {
                    std::cerr << "[orpheus-speak]   chunk " << next_collect + 1
                              << ": \"" << chunks[next_collect].substr(0, 60)
                              << (chunks[next_collect].size() > 60 ? "..." : "")
                              << "\"\n";
                }

                HttpResult hr = http_futures[next_collect].get();
                next_collect++;

                // Top up pipeline while we process this result
                read_new_lines();
                if (next_collect >= chunks.size() && !pending_chunk.empty())
                    flush_pending();
                submit_available();

                if (!hr.ok) {
                    std::cerr << "[orpheus-speak]   chunk " << next_collect
                              << " HTTP failed, skipping\n";
                    return false;
                }
                total_server_ms += hr.server_ms;

                std::string gen_text = extract_text_from_json(hr.response);
                if (gen_text.empty()) gen_text = hr.response;
                std::vector<int> audio_tokens = extract_audio_tokens(gen_text, false);
                if (audio_tokens.empty()) return false;
                int usable = ((int)audio_tokens.size() / ORPHEUS_FRAME_TOKENS)
                           * ORPHEUS_FRAME_TOKENS;
                audio_tokens.resize(usable);
                total_tokens += usable;
                SnacCodes codes = deinterleave_tokens(audio_tokens, false);
                auto t_s0 = std::chrono::high_resolution_clock::now();
                std::vector<float> pcm = snac.decode(cfg, codes);
                auto t_s1 = std::chrono::high_resolution_clock::now();
                total_decode_ms += std::chrono::duration<float, std::milli>(t_s1 - t_s0).count();
                if (pcm.empty()) return false;
                total_pcm_samples += pcm.size();

                // Append to accumulator — multiple chunks become one
                // seamless WAV with zero inter-chunk gaps.
                float audio_ms = (float)pcm.size() / cfg.sample_rate * 1000.0f;
                pcm_accum.insert(pcm_accum.end(), pcm.begin(), pcm.end());
                accum_audio_ms += audio_ms;
                accum_chunks++;
                return true;
            };

            // Buffer strategy: wait for MIN_BUFFER_INITIAL chunks before
            // FIRST playback to build a cushion, then switch to playing
            // whatever is available (MIN_BUFFER_SUBSEQUENT=1).  This avoids
            // the 10-30s silence gaps that occur when FILL demands 3 chunks
            // between every batch — after the initial buffer is built, the
            // REFILL phase keeps the pipeline fed during playback.
            static constexpr int MIN_BUFFER_INITIAL    = 3;
            static constexpr int MIN_BUFFER_SUBSEQUENT = 1;
            int min_buffer_now = MIN_BUFFER_INITIAL;

            while (true) {
                // ── FILL: collect + decode until buffer ready ──────────
                while (accum_chunks < min_buffer_now) {
                    read_new_lines();
                    if (next_collect >= chunks.size() && !pending_chunk.empty())
                        flush_pending();
                    submit_available();

                    if (next_collect < next_submit) {
                        collect_one();
                    } else {
                        // Nothing in flight — play what we have or wait
                        if (accum_chunks > 0 || end_received) break;
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    }
                }

                // Nothing decoded and session over → done
                if (pcm_accum.empty()) {
                    if (end_received && next_collect >= chunks.size()) break;
                    continue;
                }

                // ── PLAY: concatenated PCM as one seamless WAV ────────
                if (play_thread.joinable()) play_thread.join();
                {
                    auto now = std::chrono::high_resolution_clock::now();
                    if (now < play_end_time)
                        std::this_thread::sleep_until(play_end_time);
                }

                if (cfg.verbose && accum_chunks > 1) {
                    std::cerr << "[orpheus-speak]   playing " << accum_chunks
                              << " chunks combined (" << (int)accum_audio_ms
                              << " ms audio)\n";
                }

                if (!write_wav(wav_buf[buf_idx], pcm_accum, cfg.sample_rate)) {
                    pcm_accum.clear();
                    accum_audio_ms = 0;
                    accum_chunks = 0;
                    continue;
                }
                if (!cfg.play_cmd.empty()) {
                    play_end_time = std::chrono::high_resolution_clock::now()
                                  + std::chrono::milliseconds((int)accum_audio_ms + 100);
                    std::string wav = wav_buf[buf_idx];
                    std::string pcmd = cfg.play_cmd;
                    play_thread = std::thread([pcmd, wav]() { play_audio(pcmd, wav); });
                }
                buf_idx = 1 - buf_idx;

                pcm_accum.clear();
                accum_audio_ms = 0;
                accum_chunks = 0;

                // After first batch, switch to "play whatever is ready"
                // mode — the pipeline is primed, no need to re-buffer.
                min_buffer_now = MIN_BUFFER_SUBSEQUENT;

                // ── REFILL: collect more while current batch plays ────
                // This is the key to gapless audio: while the combined
                // WAV plays (~10-20s), we decode the next batch of chunks.
                // By the time playback ends, the next batch is ready.
                {
                    auto now = std::chrono::high_resolution_clock::now();
                    while (now < play_end_time) {
                        read_new_lines();
                        if (next_collect >= chunks.size() && !pending_chunk.empty())
                            flush_pending();
                        submit_available();

                        if (next_collect < next_submit) {
                            collect_one();
                        } else if (end_received) {
                            break;
                        } else {
                            std::this_thread::sleep_for(std::chrono::milliseconds(50));
                        }
                        now = std::chrono::high_resolution_clock::now();
                    }
                }

                // Check if we're completely done
                if (end_received && next_collect >= chunks.size()
                    && pcm_accum.empty()) break;
            }

            // Wait for final playback
            if (play_thread.joinable()) play_thread.join();
            {
                auto now = std::chrono::high_resolution_clock::now();
                if (now < play_end_time)
                    std::this_thread::sleep_until(play_end_time);
            }
            std::remove(wav_buf[1].c_str());

            // Print session timing
            if (cfg.verbose && total_pcm_samples > 0) {
                auto t_end = std::chrono::high_resolution_clock::now();
                auto total_ms = std::chrono::duration<float, std::milli>(t_end - t0).count();
                float audio_sec = (float)total_pcm_samples / cfg.sample_rate;
                std::cerr << "[orpheus-speak] streaming timing:\n"
                          << "  llama-server:  " << total_server_ms
                          << " ms (" << total_tokens << " tokens)\n"
                          << "  SNAC decode:   " << total_decode_ms << " ms\n"
                          << "  total wall:    " << total_ms << " ms\n"
                          << "  audio length:  " << audio_sec * 1000.0f << " ms\n"
                          << "  chunks:        " << chunks.size() << "\n"
                          << "  pipeline:      continuous sliding window (max "
                          << max_inflight << " in-flight)\n";
            }

            } // end streaming mode (else branch)

            // ── Cleanup (shared by batch + streaming) ────────────────────
            std::ofstream(done_file).put('1');
            std::remove(cfg.watch_file.c_str());

            // Drain ALL pending inotify events.  Each sentence append
            // triggered IN_CLOSE_WRITE; those events are now stale.
            // Without this drain, stale events would trigger duplicate
            // sessions that re-process deleted/recreated files.
            {
                int flags = fcntl(ifd, F_GETFL, 0);
                fcntl(ifd, F_SETFL, flags | O_NONBLOCK);
                while (read(ifd, evbuf, sizeof(evbuf)) > 0) { /* discard */ }
                fcntl(ifd, F_SETFL, flags);  // restore blocking
            }

            if (cfg.verbose) {
                std::cerr << "[orpheus-speak] session complete\n";
            }
        }

        close(ifd);

    } else {
        // ── Single-shot mode ─────────────────────────────────────────────
        std::string text;
        if (!cfg.input_file.empty()) {
            text = read_text_file(cfg.input_file);
        } else {
            text = cfg.text;
        }

        // Trim
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' '))
            text.pop_back();

        if (text.empty()) {
            std::cerr << "[orpheus-speak] no input text\n";
            usage(argv[0]);
            snac.cleanup();
            return 1;
        }

        bool ok = speak_text(cfg, snac, text);
        snac.cleanup();
        curl_global_cleanup();
        return ok ? 0 : 1;
    }
}
