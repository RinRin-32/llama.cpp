// tools/tts/qwen3tts.cpp
// Qwen3-TTS CLI: text + reference audio → speech via llama.cpp
//
// Pipeline:
//   1. Speaker encoder (ECAPA-TDNN) extracts speaker embedding from reference WAV
//   2. Talker model generates codec token IDs (codebook 0) with MRoPE
//   3. Code predictor generates remaining 15 codebook tokens per frame
//   4. Vocoder decodes all 16 codebooks into audio waveform
//
// Usage:
//   llama-qwen3tts \
//     --model-talker talker.gguf \
//     --model-cp code-predictor.gguf \
//     --model-vocoder tokenizer.gguf \
//     --ref-audio reference.wav \
//     --text "Hello world" \
//     --output output.wav

#include "llama.h"
#include "common.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif
#include <map>
#include <numeric>
#include <chrono>
#include <fstream>
#include <random>
#include <sstream>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <thread>
#include <future>
#include <memory>

#define CPPHTTPLIB_NO_DEFAULT_USER_AGENT
#include "httplib.h"
#include "nlohmann/json.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ═══════════════════════════════════════════════════════════════════════════════
//  WAV I/O helpers
// ═══════════════════════════════════════════════════════════════════════════════

// Resample mono float audio to dst_sr. Linear interpolation, with a simple moving-average
// low-pass before downsampling to curb aliasing. Good enough for the speaker encoder and the
// speech tokenizer, both of which expect 24 kHz input.
static std::vector<float> resample_linear(const std::vector<float> & in, int src_sr, int dst_sr) {
    if (src_sr == dst_sr || in.empty() || src_sr <= 0 || dst_sr <= 0) return in;
    std::vector<float> x = in;
    if (dst_sr < src_sr) {
        int k = src_sr / dst_sr; // anti-alias window, e.g. 2 for 48k->24k
        if (k > 1) {
            std::vector<float> lp(in.size());
            int half = k / 2;
            for (int i = 0; i < (int)in.size(); i++) {
                float acc = 0.0f; int cnt = 0;
                for (int j = -half; j <= half; j++) {
                    int idx = i + j;
                    if (idx >= 0 && idx < (int)in.size()) { acc += in[idx]; cnt++; }
                }
                lp[i] = acc / cnt;
            }
            x = std::move(lp);
        }
    }
    double ratio = (double)dst_sr / (double)src_sr;
    size_t n_out = (size_t)(x.size() * ratio);
    std::vector<float> out(n_out);
    for (size_t i = 0; i < n_out; i++) {
        double pos = (double)i / ratio;
        size_t i0 = (size_t)pos;
        double frac = pos - (double)i0;
        float a = x[i0];
        float b = (i0 + 1 < x.size()) ? x[i0 + 1] : a;
        out[i] = (float)(a * (1.0 - frac) + b * frac);
    }
    return out;
}

static bool read_wav(const char * path, std::vector<float> & out, int target_sr = 24000) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return false; }

    char riff[4]; fread(riff, 1, 4, f);
    if (memcmp(riff, "RIFF", 4) != 0) { fclose(f); fprintf(stderr, "ERROR: not RIFF\n"); return false; }
    fseek(f, 4, SEEK_CUR);
    char wave[4]; fread(wave, 1, 4, f);
    if (memcmp(wave, "WAVE", 4) != 0) { fclose(f); fprintf(stderr, "ERROR: not WAVE\n"); return false; }

    int sr = 0, n_ch = 0, bps = 0; bool have_fmt = false;
    while (true) {
        char id[4]; uint32_t sz;
        if (fread(id, 1, 4, f) != 4 || fread(&sz, 4, 1, f) != 1) break;
        if (memcmp(id, "fmt ", 4) == 0) {
            uint16_t afmt, ch; uint32_t srate; uint16_t bps16;
            fread(&afmt, 2, 1, f); fread(&ch, 2, 1, f); fread(&srate, 4, 1, f);
            fseek(f, 4, SEEK_CUR); fseek(f, 2, SEEK_CUR); fread(&bps16, 2, 1, f);
            if (sz > 16) fseek(f, sz - 16, SEEK_CUR);
            if (afmt != 1) { fclose(f); fprintf(stderr, "ERROR: not PCM\n"); return false; }
            sr = (int)srate; n_ch = (int)ch; bps = (int)bps16; have_fmt = true;
        } else if (memcmp(id, "data", 4) == 0) {
            if (!have_fmt) { fclose(f); return false; }
            int n = (int)(sz / (bps / 8) / n_ch);
            out.resize(n);
            for (int i = 0; i < n; i++) {
                int16_t s16 = 0;
                for (int c = 0; c < n_ch; c++) {
                    int16_t cs; fread(&cs, 2, 1, f);
                    if (c == 0) s16 = cs;
                }
                out[i] = s16 / 32767.0f;
            }
            break;
        } else { fseek(f, sz, SEEK_CUR); }
    }
    fclose(f);
    if (sr != target_sr && !out.empty()) {
        int n_before = (int)out.size();
        out = resample_linear(out, sr, target_sr);
        printf("Resampled %s: %d Hz -> %d Hz (%d -> %d samples)\n",
               path, sr, target_sr, n_before, (int)out.size());
        sr = target_sr;
    }
    printf("Read %s  (%d samples, %.2fs)\n", path, (int)out.size(), (float)out.size() / sr);
    return !out.empty();
}

static void write_wav(const char * path, const float * samples, int n, int sr = 24000) {
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", path); return; }
    const int32_t dsz = n * 2, csz = 36 + dsz, brate = sr * 2;
    const int16_t n_ch = 1, bps = 16, balign = 2, pcm = 1;
    fwrite("RIFF", 1, 4, f); fwrite(&csz, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    const int32_t fmt = 16; fwrite(&fmt, 4, 1, f);
    fwrite(&pcm, 2, 1, f); fwrite(&n_ch, 2, 1, f); fwrite(&sr, 4, 1, f);
    fwrite(&brate, 4, 1, f); fwrite(&balign, 2, 1, f); fwrite(&bps, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&dsz, 4, 1, f);
    for (int i = 0; i < n; i++) {
        float v = fmaxf(-1.0f, fminf(1.0f, samples[i]));
        int16_t s16 = (int16_t)(v * 32767.0f);
        fwrite(&s16, 2, 1, f);
    }
    fclose(f);
    printf("Wrote %s  (%d samples, %.2fs)\n", path, n, (float)n / sr);
}

// Encode mono f32 [-1,1] samples to an in-memory 16-bit PCM WAV (for HTTP responses).
static std::string wav_bytes(const float * samples, int n, int sr = 24000) {
    auto p32 = [](std::string & s, uint32_t v) { for (int i = 0; i < 4; i++) s.push_back((char)((v >> (8*i)) & 0xff)); };
    auto p16 = [](std::string & s, uint16_t v) { for (int i = 0; i < 2; i++) s.push_back((char)((v >> (8*i)) & 0xff)); };
    std::string s; uint32_t dsz = (uint32_t)n * 2;
    s += "RIFF"; p32(s, 36 + dsz); s += "WAVE"; s += "fmt "; p32(s, 16);
    p16(s, 1); p16(s, 1); p32(s, (uint32_t)sr); p32(s, (uint32_t)sr * 2); p16(s, 2); p16(s, 16);
    s += "data"; p32(s, dsz);
    for (int i = 0; i < n; i++) {
        float v = fmaxf(-1.0f, fminf(1.0f, samples[i]));
        int16_t q = (int16_t)(v * 32767.0f);
        s.push_back((char)(q & 0xff)); s.push_back((char)((q >> 8) & 0xff));
    }
    return s;
}

// Find the zero-crossing nearest to `cut` within +/- window samples (sglang-omni style).
// Used to place a clean cut when trimming, avoiding clicks from cutting mid-waveform.
static int find_zero_crossing(const std::vector<float> & wav, int cut, int window = 480) {
    int n = (int)wav.size();
    int hw = std::min(std::min(window, cut), n - cut - 1);
    if (hw <= 0) return cut;
    int best = cut, best_dist = hw + 1;
    for (int i = cut - hw; i < cut + hw; i++) {
        if ((wav[i] >= 0.0f) != (wav[i + 1] >= 0.0f)) {
            int d = std::abs(i - cut);
            if (d < best_dist) { best_dist = d; best = i + 1; }
        }
    }
    return best;
}

// Linear fade-in over the first `samples` samples, to smooth a hard cut (sglang-omni style).
static void fade_in(std::vector<float> & wav, int samples = 240) {
    int n = std::min(samples, (int)wav.size());
    for (int i = 0; i < n && n > 1; i++) wav[i] *= (float)i / (float)(n - 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Speaker Encoder (ECAPA-TDNN)
//  Extracts a 1024-dim speaker embedding from reference audio.
//  Weights are read from the Talker GGUF under the spk_enc.* namespace.
// ═══════════════════════════════════════════════════════════════════════════════

static constexpr int SPK_N_MELS   = 128;
static constexpr int SPK_N_FFT    = 1024;
static constexpr int SPK_HOP      = 256;
static constexpr int SPK_WIN      = 1024;
static constexpr int SPK_EMB_DIM  = 1024;
static constexpr int SPK_HIDDEN   = 512;
static constexpr int SPK_SCALE    = 8;
static constexpr int SPK_BRANCH   = SPK_HIDDEN / SPK_SCALE; // 64
#define SPK_MAX_NODES 16384

struct spk_res2net_block {
    ggml_tensor * tdnn1_w = nullptr;
    ggml_tensor * tdnn1_b = nullptr;
    ggml_tensor * res2net_w[7] = {};
    ggml_tensor * res2net_b[7] = {};
    ggml_tensor * tdnn2_w = nullptr;
    ggml_tensor * tdnn2_b = nullptr;
    ggml_tensor * se_conv1_w = nullptr;
    ggml_tensor * se_conv1_b = nullptr;
    ggml_tensor * se_conv2_w = nullptr;
    ggml_tensor * se_conv2_b = nullptr;
};

struct spk_encoder_model {
    ggml_tensor * conv0_w = nullptr;
    ggml_tensor * conv0_b = nullptr;
    spk_res2net_block blocks[3];
    ggml_tensor * mfa_w   = nullptr;
    ggml_tensor * mfa_b   = nullptr;
    ggml_tensor * asp_conv_w = nullptr;
    ggml_tensor * asp_conv_b = nullptr;
    ggml_tensor * asp_tdnn_w = nullptr;
    ggml_tensor * asp_tdnn_b = nullptr;
    ggml_tensor * fc_w    = nullptr;
    ggml_tensor * fc_b    = nullptr;
};

// Slaney mel filterbank (matches librosa norm='slaney')
static void compute_mel_filterbank(float * fb, int n_mels, int n_fft, int sr, float fmin, float fmax) {
    auto hz2mel = [](float hz) -> float {
        const float f_sp = 200.0f / 3.0f;
        if (hz < 1000.0f) return hz / f_sp;
        return 1000.0f / f_sp + logf(hz / 1000.0f) / (logf(6.4f) / 27.0f);
    };
    auto mel2hz = [](float mel) -> float {
        const float f_sp = 200.0f / 3.0f;
        const float min_log_mel = 1000.0f / f_sp;
        if (mel < min_log_mel) return f_sp * mel;
        return 1000.0f * expf((logf(6.4f) / 27.0f) * (mel - min_log_mel));
    };
    int bins = n_fft / 2 + 1;
    float mel_min = hz2mel(fmin), mel_max = hz2mel(fmax);
    std::vector<float> mel_pts(n_mels + 2), hz_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        mel_pts[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
        hz_pts[i] = mel2hz(mel_pts[i]);
    }
    memset(fb, 0, n_mels * bins * sizeof(float));
    for (int m = 0; m < n_mels; m++) {
        float fl = hz_pts[m], fc = hz_pts[m + 1], fr = hz_pts[m + 2];
        float enorm = 2.0f / (fr - fl);
        for (int k = 0; k < bins; k++) {
            float freq = (float)k * sr / n_fft;
            if (freq >= fl && freq <= fc && fc > fl)
                fb[m * bins + k] = enorm * (freq - fl) / (fc - fl);
            else if (freq > fc && freq <= fr && fr > fc)
                fb[m * bins + k] = enorm * (fr - freq) / (fr - fc);
        }
    }
}

static void compute_dft(const float * input, float * real, float * imag, int n) {
    for (int k = 0; k < n; k++) {
        real[k] = 0.0f; imag[k] = 0.0f;
        for (int t = 0; t < n; t++) {
            float angle = -2.0f * (float)M_PI * k * t / n;
            real[k] += input[t] * cosf(angle);
            imag[k] += input[t] * sinf(angle);
        }
    }
}

static bool compute_mel_spectrogram(const float * samples, int n_samples,
                                     std::vector<float> & mel, int & n_frames) {
    int padding = (SPK_N_FFT - SPK_HOP) / 2;
    int padded_len = n_samples + 2 * padding;

    std::vector<float> padded(padded_len);
    for (int i = 0; i < padded_len; i++) {
        int src;
        if (i < padding)                   src = padding - i;
        else if (i >= padding + n_samples) src = 2 * n_samples - (i - padding) - 2;
        else                               src = i - padding;
        padded[i] = samples[std::max(0, std::min(n_samples - 1, src))];
    }

    n_frames = (padded_len - SPK_N_FFT) / SPK_HOP + 1;
    if (n_frames <= 0) return false;

    int bins = SPK_N_FFT / 2 + 1;
    std::vector<float> filterbank(SPK_N_MELS * bins);
    compute_mel_filterbank(filterbank.data(), SPK_N_MELS, SPK_N_FFT, 24000, 0.0f, 12000.0f);

    std::vector<float> window(SPK_N_FFT, 0.0f);
    int offset = (SPK_N_FFT - SPK_WIN) / 2;
    for (int i = 0; i < SPK_WIN; i++)
        window[offset + i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / SPK_WIN));

    mel.resize(SPK_N_MELS * n_frames);
    std::vector<float> frame(SPK_N_FFT), fft_re(SPK_N_FFT), fft_im(SPK_N_FFT);

    for (int fr = 0; fr < n_frames; fr++) {
        int start = fr * SPK_HOP;
        for (int i = 0; i < SPK_N_FFT; i++) frame[i] = padded[start + i] * window[i];
        compute_dft(frame.data(), fft_re.data(), fft_im.data(), SPK_N_FFT);
        for (int m = 0; m < SPK_N_MELS; m++) {
            float sum = 0.0f;
            for (int k = 0; k < bins; k++) {
                float mag = sqrtf(fft_re[k] * fft_re[k] + fft_im[k] * fft_im[k] + 1e-9f);
                sum += filterbank[m * bins + k] * mag;
            }
            mel[m * n_frames + fr] = logf(std::max(sum, 1e-5f));
        }
    }
    return true;
}

static ggml_tensor * spk_conv1d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                                 ggml_tensor * x, int stride, int pad, int dilation) {
    if (w->type != GGML_TYPE_F16) {
        w = ggml_cast(ctx, w, GGML_TYPE_F16);
    }
    ggml_tensor * y = ggml_conv_1d(ctx, w, x, stride, pad, dilation);
    if (b) {
        int64_t oc = y->ne[1];
        y = ggml_add(ctx, y, ggml_reshape_3d(ctx, b, 1, oc, 1));
    }
    return y;
}

struct gguf_tensor_loader {
    ggml_context * ctx = nullptr;
    struct gguf_context * guf = nullptr;
    std::map<std::string, ggml_tensor *> tensors;

    ~gguf_tensor_loader() {
        if (guf) gguf_free(guf);
        if (ctx) ggml_free(ctx);
    }

    bool load(const char * path, const char * prefix) {
        struct gguf_init_params params;
        params.no_alloc = false;
        params.ctx = &ctx;
        guf = gguf_init_from_file(path, params);
        if (!guf) {
            fprintf(stderr, "ERROR: cannot open GGUF: %s\n", path);
            return false;
        }
        int64_t n = gguf_get_n_tensors(guf);
        int loaded = 0;
        size_t prefix_len = strlen(prefix);
        for (int64_t i = 0; i < n; i++) {
            const char * name = gguf_get_tensor_name(guf, i);
            if (strncmp(name, prefix, prefix_len) == 0) {
                ggml_tensor * t = ggml_get_tensor(ctx, name);
                if (t) {
                    tensors[name] = t;
                    loaded++;
                }
            }
        }
        printf("Loaded %d tensors with prefix '%s' from %s\n", loaded, prefix, path);
        return loaded > 0;
    }

    ggml_tensor * get(const char * name) const {
        auto it = tensors.find(name);
        return (it != tensors.end()) ? it->second : nullptr;
    }
};

static bool spk_load_weights(gguf_tensor_loader & loader, spk_encoder_model & spk) {
    auto get = [&](const char * name) -> ggml_tensor * {
        ggml_tensor * t = loader.get(name);
        if (!t) fprintf(stderr, "WARN: missing tensor: %s\n", name);
        return t;
    };

    spk.conv0_w = get("spk_enc.conv0.weight");
    spk.conv0_b = get("spk_enc.conv0.bias");

    for (int b = 0; b < 3; b++) {
        char buf[128];
        auto & blk = spk.blocks[b];
        int hf_idx = b + 1;
        snprintf(buf, sizeof(buf), "spk_enc.blk.%d.tdnn1.weight", hf_idx);     blk.tdnn1_w = get(buf);
        snprintf(buf, sizeof(buf), "spk_enc.blk.%d.tdnn1.bias", hf_idx);       blk.tdnn1_b = get(buf);
        snprintf(buf, sizeof(buf), "spk_enc.blk.%d.tdnn2.weight", hf_idx);     blk.tdnn2_w = get(buf);
        snprintf(buf, sizeof(buf), "spk_enc.blk.%d.tdnn2.bias", hf_idx);       blk.tdnn2_b = get(buf);
        for (int r = 0; r < 7; r++) {
            snprintf(buf, sizeof(buf), "spk_enc.blk.%d.res2net.%d.weight", hf_idx, r); blk.res2net_w[r] = get(buf);
            snprintf(buf, sizeof(buf), "spk_enc.blk.%d.res2net.%d.bias", hf_idx, r);   blk.res2net_b[r] = get(buf);
        }
        snprintf(buf, sizeof(buf), "spk_enc.blk.%d.se.conv1.weight", hf_idx);  blk.se_conv1_w = get(buf);
        snprintf(buf, sizeof(buf), "spk_enc.blk.%d.se.conv1.bias", hf_idx);    blk.se_conv1_b = get(buf);
        snprintf(buf, sizeof(buf), "spk_enc.blk.%d.se.conv2.weight", hf_idx);  blk.se_conv2_w = get(buf);
        snprintf(buf, sizeof(buf), "spk_enc.blk.%d.se.conv2.bias", hf_idx);    blk.se_conv2_b = get(buf);
    }

    spk.mfa_w     = get("spk_enc.mfa.weight");
    spk.mfa_b     = get("spk_enc.mfa.bias");
    spk.asp_conv_w = get("spk_enc.asp.conv.weight");
    spk.asp_conv_b = get("spk_enc.asp.conv.bias");
    spk.asp_tdnn_w = get("spk_enc.asp.tdnn.weight");
    spk.asp_tdnn_b = get("spk_enc.asp.tdnn.bias");
    spk.fc_w      = get("spk_enc.fc.weight");
    spk.fc_b      = get("spk_enc.fc.bias");

    bool ok = spk.conv0_w && spk.fc_w;
    if (!ok) fprintf(stderr, "ERROR: missing critical speaker encoder tensors\n");
    return ok;
}

static ggml_cgraph * spk_build_graph(ggml_context * ctx0, const spk_encoder_model & m, int n_frames) {
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, SPK_MAX_NODES, false);
    const int64_t seq_len_init = n_frames;

    ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_frames, SPK_N_MELS);
    ggml_set_name(mel, "mel_input"); ggml_set_input(mel);

    ggml_tensor * cur = ggml_reshape_3d(ctx0, mel, n_frames, SPK_N_MELS, 1);

    // Reflect pad left by 2
    // For the initial conv (kernel=5), we need pad=2 on each side
    // Using ggml_pad_ext for left padding
    cur = ggml_pad_ext(ctx0, cur, 2, 0, 0, 0, 0, 0, 0, 0);
    {
        ggml_tensor * conv0_w = m.conv0_w;
        if (conv0_w->type != GGML_TYPE_F16) {
            conv0_w = ggml_cast(ctx0, conv0_w, GGML_TYPE_F16);
        }
        cur = ggml_conv_1d(ctx0, conv0_w, cur, 1, 0, 1);
    }
    if (m.conv0_b) {
        int64_t oc = cur->ne[1];
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, m.conv0_b, 1, oc, 1));
    }
    cur = ggml_relu(ctx0, cur);
    ggml_set_name(cur, "conv0_out");

    int64_t seq_len = cur->ne[0];

    ggml_tensor * block_outputs[4];
    block_outputs[0] = cur;

    int dilations[3] = {2, 3, 4};

    for (int blk = 0; blk < 3; blk++) {
        const auto & block = m.blocks[blk];
        int dilation = dilations[blk];

        ggml_tensor * residual = cur;

        cur = spk_conv1d(ctx0, block.tdnn1_w, block.tdnn1_b, cur, 1, 0, 1);
        cur = ggml_relu(ctx0, cur);

        // Res2Net: split into 8 branches of 64 channels
        ggml_tensor * branches[SPK_SCALE];
        for (int b = 0; b < SPK_SCALE; b++) {
            branches[b] = ggml_view_3d(ctx0, cur, seq_len, SPK_BRANCH, 1,
                                        cur->nb[1], cur->nb[2], b * SPK_BRANCH * cur->nb[1]);
            branches[b] = ggml_cont(ctx0, branches[b]);
        }

        ggml_tensor * outputs[SPK_SCALE];
        outputs[0] = branches[0];
        for (int b = 1; b < SPK_SCALE; b++) {
            ggml_tensor * input = (b == 1) ? branches[b] : ggml_add(ctx0, branches[b], outputs[b - 1]);
            if (block.res2net_w[b - 1]) {
                int pad_val = (3 - 1) * dilation;
                input = ggml_pad_ext(ctx0, input, pad_val, 0, 0, 0, 0, 0, 0, 0);
                outputs[b] = spk_conv1d(ctx0, block.res2net_w[b - 1], block.res2net_b[b - 1],
                                        input, 1, 0, dilation);
                outputs[b] = ggml_relu(ctx0, outputs[b]);
            } else {
                outputs[b] = input;
            }
        }

        cur = outputs[0];
        for (int b = 1; b < SPK_SCALE; b++)
            cur = ggml_concat(ctx0, cur, outputs[b], 1);

        cur = spk_conv1d(ctx0, block.tdnn2_w, block.tdnn2_b, cur, 1, 0, 1);
        cur = ggml_relu(ctx0, cur);

        // SE (Squeeze-Excitation)
        ggml_tensor * se = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
        se = ggml_reshape_3d(ctx0, se, 1, SPK_HIDDEN, 1);
        se = spk_conv1d(ctx0, block.se_conv1_w, block.se_conv1_b, se, 1, 0, 1);
        se = ggml_relu(ctx0, se);
        se = spk_conv1d(ctx0, block.se_conv2_w, block.se_conv2_b, se, 1, 0, 1);
        se = ggml_sigmoid(ctx0, se);
        cur = ggml_mul(ctx0, cur, se);

        cur = ggml_add(ctx0, cur, residual);
        block_outputs[blk + 1] = cur;
    }

    // MFA: concatenate block outputs [1:]
    ggml_tensor * mfa_in = ggml_concat(ctx0, block_outputs[1], block_outputs[2], 1);
    mfa_in = ggml_concat(ctx0, mfa_in, block_outputs[3], 1);
    cur = spk_conv1d(ctx0, m.mfa_w, m.mfa_b, mfa_in, 1, 0, 1);
    cur = ggml_relu(ctx0, cur);
    ggml_set_name(cur, "mfa_out");

    // ASP (Attentive Statistics Pooling)
    const int mfa_ch = 1536;
    ggml_tensor * global_mean = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    global_mean = ggml_reshape_3d(ctx0, global_mean, 1, mfa_ch, 1);

    ggml_tensor * sq = ggml_sqr(ctx0, cur);
    ggml_tensor * mean_sq = ggml_pool_1d(ctx0, sq, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    mean_sq = ggml_reshape_3d(ctx0, mean_sq, 1, mfa_ch, 1);
    ggml_tensor * var = ggml_sub(ctx0, mean_sq, ggml_sqr(ctx0, global_mean));
    var = ggml_clamp(ctx0, var, 1e-12f, 1e10f);
    ggml_tensor * global_std = ggml_sqrt(ctx0, var);

    ggml_tensor * ref_3d = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, seq_len, mfa_ch, 1);
    ggml_tensor * mean_exp = ggml_repeat(ctx0, global_mean, ref_3d);
    ggml_tensor * std_exp  = ggml_repeat(ctx0, global_std, ref_3d);

    ggml_tensor * attn = ggml_concat(ctx0, cur, mean_exp, 1);
    attn = ggml_concat(ctx0, attn, std_exp, 1);

    attn = spk_conv1d(ctx0, m.asp_tdnn_w, m.asp_tdnn_b, attn, 1, 0, 1);
    attn = ggml_relu(ctx0, attn);
    attn = ggml_tanh(ctx0, attn);
    attn = spk_conv1d(ctx0, m.asp_conv_w, m.asp_conv_b, attn, 1, 0, 1);
    attn = ggml_soft_max(ctx0, attn);

    ggml_tensor * weighted = ggml_mul(ctx0, attn, cur);
    ggml_tensor * w_mean = ggml_pool_1d(ctx0, weighted, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    w_mean = ggml_scale(ctx0, w_mean, (float)seq_len);
    w_mean = ggml_reshape_3d(ctx0, w_mean, 1, mfa_ch, 1);

    ggml_tensor * mean_for_std = ggml_repeat(ctx0, w_mean, ref_3d);
    ggml_tensor * diff = ggml_sub(ctx0, cur, mean_for_std);
    ggml_tensor * diff_sq = ggml_sqr(ctx0, diff);
    ggml_tensor * w_var = ggml_mul(ctx0, attn, diff_sq);
    ggml_tensor * var_sum = ggml_pool_1d(ctx0, w_var, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    var_sum = ggml_scale(ctx0, var_sum, (float)seq_len);
    var_sum = ggml_reshape_3d(ctx0, var_sum, 1, mfa_ch, 1);
    var_sum = ggml_clamp(ctx0, var_sum, 1e-12f, 1e10f);
    ggml_tensor * w_std = ggml_sqrt(ctx0, var_sum);

    ggml_tensor * pooled = ggml_concat(ctx0, w_mean, w_std, 1);

    cur = spk_conv1d(ctx0, m.fc_w, m.fc_b, pooled, 1, 0, 1);
    // Embedding dim = fc output channels: 1024 for 0.6B, 2048 for 1.7B. Derive it from the
    // weight rather than hardcoding so both model sizes work.
    cur = ggml_reshape_1d(ctx0, cur, ggml_nelements(cur));
    ggml_set_name(cur, "embedding");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);
    (void)seq_len_init;
    return gf;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Speech Tokenizer Encoder (MimiModel-based)
//  Encodes raw audio → discrete codec tokens for ICL voice cloning.
//  Architecture: SEANet encoder → transformer (8 layers, RoPE) → SplitRVQ
//  Ported from llama.cpp mimi codec implementation.
// ═══════════════════════════════════════════════════════════════════════════════

static constexpr int ENC_DIM        = 512;
static constexpr int ENC_VQ_DIM     = 256;
static constexpr int ENC_VQ_BINS    = 2048;
static constexpr int ENC_N_Q        = 32;
static constexpr int ENC_N_HEADS    = 8;
static constexpr int ENC_HEAD_DIM   = ENC_DIM / ENC_N_HEADS;  // 64
static constexpr int ENC_FFN_DIM    = 2048;
static constexpr int ENC_N_TR       = 8;
static constexpr int ENC_N_SBLOCKS  = 4;
static constexpr float ENC_NORM_EPS = 1e-5f;
static constexpr float ENC_ROPE_THETA = 10000.0f;

static const int enc_ratios[4]  = {4, 5, 6, 8};
static const int enc_ch_in[4]   = { 64, 128, 256,  512};
static const int enc_ch_out[4]  = {128, 256, 512, 1024};
static const int enc_kernels[4] = {8, 10, 12, 16};

struct enc_seanet_block {
    ggml_tensor * res_conv1   = nullptr;  // residual conv1 [ch/2, ch, 3]
    ggml_tensor * res_conv1_b = nullptr;
    ggml_tensor * res_conv2   = nullptr;  // residual conv2 [ch, ch/2] (k=1, stored as 2D)
    ggml_tensor * res_conv2_b = nullptr;
    ggml_tensor * conv_stride   = nullptr;  // strided conv [ch_out, ch_in, 2*stride]
    ggml_tensor * conv_stride_b = nullptr;
};

struct enc_tr_layer {
    ggml_tensor * attn_norm   = nullptr;
    ggml_tensor * attn_norm_b = nullptr;
    ggml_tensor * attn_q      = nullptr;
    ggml_tensor * attn_k      = nullptr;
    ggml_tensor * attn_v      = nullptr;
    ggml_tensor * attn_output = nullptr;
    ggml_tensor * attn_scale  = nullptr;
    ggml_tensor * ffn_norm    = nullptr;
    ggml_tensor * ffn_norm_b  = nullptr;
    ggml_tensor * ffn_up      = nullptr;
    ggml_tensor * ffn_down    = nullptr;
    ggml_tensor * ffn_scale   = nullptr;
};

struct enc_model {
    ggml_tensor * conv_in     = nullptr;  // [7, 1, 64]
    ggml_tensor * conv_in_b   = nullptr;
    enc_seanet_block blocks[ENC_N_SBLOCKS];
    ggml_tensor * conv_out    = nullptr;  // [3, 1024, 512]
    ggml_tensor * conv_out_b  = nullptr;
    ggml_tensor * downsample  = nullptr;  // [4, 512, 512]
    ggml_tensor * downsample_b = nullptr;
    enc_tr_layer tr[ENC_N_TR];
    ggml_tensor * vq_semantic_input_proj = nullptr;   // [512, 256]
    ggml_tensor * vq_acoustic_input_proj = nullptr;   // [512, 256]
    ggml_tensor * vq_semantic_codebook[1]  = {};      // codebook 0
    ggml_tensor * vq_acoustic_codebook[31] = {};      // codebooks 1..31
};

static bool enc_load_weights(gguf_tensor_loader & loader, enc_model & enc) {
    auto get = [&](const char * name) -> ggml_tensor * {
        ggml_tensor * t = loader.get(name);
        return t;
    };

    // SEANet conv_in: encoder.layers[0] = initial conv (k=7, 1→64)
    enc.conv_in   = get("tok_enc.conv.0.weight");
    enc.conv_in_b = get("tok_enc.conv.0.bias");

    // SEANet blocks: encoder.layers[1..4] each have residual + strided conv
    // HF encoder.layers layout: [conv_in, block0, block1, block2, block3, conv_out, ...]
    // Each block_i = encoder.layers[1+i*3 .. 3+i*3] containing residual sub-block + ELU + strided conv
    // In the GGUF: tok_enc.conv.{idx} and tok_enc.res.{idx}
    // Layout from MimiEncoder:
    //   layers[0] = conv_in
    //   layers[1] = ResidualUnit (block 0), layers[2] = ELU, layers[3] = strided conv (block 0)
    //   layers[4] = ResidualUnit (block 1), layers[5] = ELU, layers[6] = strided conv (block 1)
    //   layers[7] = ResidualUnit (block 2), layers[8] = ELU, layers[9] = strided conv (block 2)
    //   layers[10] = ResidualUnit (block 3), layers[11] = ELU, layers[12] = strided conv (block 3)
    //   layers[13] = ELU, layers[14] = conv_out
    // SEANet encoder layer layout:
    //   0=conv_in, 1=res0, 2=elu, 3=stride0, 4=res1, 5=elu, 6=stride1,
    //   7=res2, 8=elu, 9=stride2, 10=res3, 11=elu, 12=stride3, 13=elu, 14=conv_out
    // Residual sub-blocks: blk.1 = conv1 (k=3, ch→ch/2), blk.3 = conv2 (k=1, ch/2→ch)
    static const int res_indices[4]    = {1, 4, 7, 10};
    static const int stride_indices[4] = {3, 6, 9, 12};
    for (int i = 0; i < ENC_N_SBLOCKS; i++) {
        auto & blk = enc.blocks[i];
        char buf[128];
        int ri = res_indices[i];
        int si = stride_indices[i];
        snprintf(buf, sizeof(buf), "tok_enc.res.%d.blk.1.weight", ri);  blk.res_conv1   = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.res.%d.blk.1.bias", ri);    blk.res_conv1_b = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.res.%d.blk.3.weight", ri);  blk.res_conv2   = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.res.%d.blk.3.bias", ri);    blk.res_conv2_b = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.conv.%d.weight", si);       blk.conv_stride   = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.conv.%d.bias", si);         blk.conv_stride_b = get(buf);
    }

    enc.conv_out   = get("tok_enc.conv.14.weight");
    enc.conv_out_b = get("tok_enc.conv.14.bias");

    enc.downsample   = get("tok_enc.downsample.weight");
    enc.downsample_b = nullptr;

    for (int i = 0; i < ENC_N_TR; i++) {
        char buf[128];
        auto & tr = enc.tr[i];
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.attn_norm.weight", i);  tr.attn_norm   = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.attn_norm.bias", i);    tr.attn_norm_b = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.attn_q.weight", i);     tr.attn_q      = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.attn_k.weight", i);     tr.attn_k      = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.attn_v.weight", i);     tr.attn_v      = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.attn_output.weight", i); tr.attn_output = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.attn_scale", i);        tr.attn_scale  = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.ffn_norm.weight", i);   tr.ffn_norm    = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.ffn_norm.bias", i);     tr.ffn_norm_b  = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.ffn_up.weight", i);     tr.ffn_up      = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.ffn_down.weight", i);   tr.ffn_down    = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.ffn_scale", i);         tr.ffn_scale   = get(buf);
    }

    enc.vq_semantic_input_proj = get("tok_enc.vq_semantic.input_proj.weight");
    enc.vq_acoustic_input_proj = get("tok_enc.vq_acoustic.input_proj.weight");

    // Codebooks: semantic has 1 codebook, acoustic has 31
    enc.vq_semantic_codebook[0] = get("tok_enc.vq_semantic.0.codebook");
    for (int i = 0; i < 31; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "tok_enc.vq_acoustic.%d.codebook", i);
        enc.vq_acoustic_codebook[i] = get(buf);
    }

    return enc.conv_in && enc.conv_out && enc.downsample &&
           enc.vq_semantic_input_proj && enc.vq_semantic_codebook[0];
}

// ELU activation
static inline float enc_elu(float x) { return x >= 0.f ? x : expf(x) - 1.f; }

// Causal Conv1d in C++. Weight layout: [OC, IC, K], data is channel-major [C*T].
// Causal padding = kernel - stride (matches HF MimiConv1d with use_causal_conv=True).
static void enc_causal_conv1d(
        const float * x,  int C_in,  int T_in,
        const float * w,  int C_out, int K,
        const float * b,  float * y, int stride) {
    const int pad   = K - stride;
    const int T_out = (T_in + pad - K) / stride + 1;
    memset(y, 0, (size_t)C_out * T_out * sizeof(float));
    for (int t_out = 0; t_out < T_out; t_out++) {
        for (int k = 0; k < K; k++) {
            int t_in = t_out * stride + k - pad;
            if (t_in < 0 || t_in >= T_in) continue;
            for (int co = 0; co < C_out; co++) {
                for (int ci = 0; ci < C_in; ci++) {
                    y[co * T_out + t_out] += x[ci * T_in + t_in] * w[co * C_in * K + ci * K + k];
                }
            }
        }
    }
    if (b) for (int co = 0; co < C_out; co++)
        for (int t = 0; t < T_out; t++) y[co * T_out + t] += b[co];
}

// Mat-mul for k=1 conv (weight is 2D [ch_out, ch_in])
static void enc_mul_mat(const float * w, int ci, int co,
                        const float * b, const float * x, int T, float * y) {
    memset(y, 0, (size_t)co * T * sizeof(float));
    for (int t = 0; t < T; t++) {
        for (int o = 0; o < co; o++) {
            float s = 0.f;
            for (int i = 0; i < ci; i++) s += x[i * T + t] * w[i + o * ci];
            y[o * T + t] = s;
        }
    }
    if (b) for (int o = 0; o < co; o++)
        for (int t = 0; t < T; t++) y[o * T + t] += b[o];
}

// SEANet residual block (identity skip)
static void enc_seanet_residual(
        const float * x, int ch, int T,
        const float * w1, const float * b1,
        const float * w2, const float * b2,
        float * y) {
    const int ch2 = ch / 2;
    std::vector<float> eu(ch * T), tmp(ch2 * T), branch(ch * T);
    for (int i = 0; i < ch * T; i++) eu[i] = enc_elu(x[i]);
    enc_causal_conv1d(eu.data(), ch, T, w1, ch2, 3, b1, tmp.data(), 1);
    for (int i = 0; i < ch2 * T; i++) tmp[i] = enc_elu(tmp[i]);
    enc_mul_mat(w2, ch2, ch, b2, tmp.data(), T, branch.data());
    for (int i = 0; i < ch * T; i++) y[i] = x[i] + branch[i];
}

// Get float data from a tensor (handles BF16 by casting to F32)
static const float * enc_get_f32(ggml_tensor * t, std::vector<float> & buf) {
    if (!t) return nullptr;
    if (t->type == GGML_TYPE_F32) return (const float *)t->data;
    int64_t n = ggml_nelements(t);
    buf.resize(n);
    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *)t->data;
        for (int64_t i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else if (t->type == GGML_TYPE_BF16) {
        const uint16_t * src = (const uint16_t *)t->data;
        for (int64_t i = 0; i < n; i++) {
            uint32_t tmp = (uint32_t)src[i] << 16;
            float f; memcpy(&f, &tmp, sizeof(f));
            buf[i] = f;
        }
    }
    return buf.data();
}

// SEANet encoder: raw audio → 512-dim latent (channel-major)
static std::vector<float> enc_seanet_encode(enc_model & m, const float * audio, int T_audio, int * T_out) {
    std::vector<float> b_conv_in, b_conv_in_b, b_res1, b_res1b, b_res2, b_res2b;
    std::vector<float> b_stride, b_strideb, b_co, b_cob, b_ds, b_dsb;

    const float * w_in  = enc_get_f32(m.conv_in,   b_conv_in);
    const float * b_in  = enc_get_f32(m.conv_in_b, b_conv_in_b);

    int T_cur = T_audio;
    std::vector<float> buf(64 * T_cur);
    enc_causal_conv1d(audio, 1, T_cur, w_in, 64, 7, b_in, buf.data(), 1);
    {
        float rms = 0.f;
        for (int i = 0; i < 64 * T_cur; i++) rms += buf[i] * buf[i];
        rms = sqrtf(rms / (64 * T_cur));
        printf("  [dbg] conv_in: T=%d, rms=%.6f, ch0[:5]=[%.6f, %.6f, %.6f, %.6f, %.6f]\n",
               T_cur, rms, buf[0], buf[T_cur], buf[2*T_cur], buf[3*T_cur], buf[4*T_cur]);
        printf("  [dbg] conv_in ch0 first5 samples: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
               buf[0], buf[1], buf[2], buf[3], buf[4]);
    }

    for (int i = 0; i < ENC_N_SBLOCKS; i++) {
        auto & blk = m.blocks[i];
        const int Ci  = enc_ch_in[i];
        const int Co  = enc_ch_out[i];
        const int K   = enc_kernels[i];
        const int str = enc_ratios[i];

        const float * w1  = enc_get_f32(blk.res_conv1,   b_res1);
        const float * b1  = enc_get_f32(blk.res_conv1_b, b_res1b);
        const float * w2  = enc_get_f32(blk.res_conv2,   b_res2);
        const float * b2  = enc_get_f32(blk.res_conv2_b, b_res2b);

        std::vector<float> res(Ci * T_cur);
        enc_seanet_residual(buf.data(), Ci, T_cur, w1, b1, w2, b2, res.data());
        buf = std::move(res);
        {
            float rms = 0.f;
            for (int j = 0; j < Ci * T_cur; j++) rms += buf[j] * buf[j];
            rms = sqrtf(rms / (Ci * T_cur));
            printf("  [dbg] blk%d res: T=%d, rms=%.6f, ch0[0]=%.6f\n", i, T_cur, rms, buf[0]);
        }

        for (auto & v : buf) v = enc_elu(v);

        const float * ws = enc_get_f32(blk.conv_stride,   b_stride);
        const float * bs = enc_get_f32(blk.conv_stride_b, b_strideb);
        const int T_ds = (T_cur - 1) / str + 1;
        std::vector<float> ds(Co * T_ds);
        enc_causal_conv1d(buf.data(), Ci, T_cur, ws, Co, K, bs, ds.data(), str);
        buf = std::move(ds);
        T_cur = T_ds;
        {
            float rms = 0.f;
            for (int j = 0; j < Co * T_cur; j++) rms += buf[j] * buf[j];
            rms = sqrtf(rms / (Co * T_cur));
            printf("  [dbg] blk%d stride: T=%d, rms=%.6f, ch0[0]=%.6f\n", i, T_cur, rms, buf[0]);
        }
    }

    for (auto & v : buf) v = enc_elu(v);
    {
        const float * wc = enc_get_f32(m.conv_out,   b_co);
        const float * bc = enc_get_f32(m.conv_out_b, b_cob);
        std::vector<float> co(ENC_DIM * T_cur);
        enc_causal_conv1d(buf.data(), 1024, T_cur, wc, ENC_DIM, 3, bc, co.data(), 1);
        buf = std::move(co);
    }
    // NOTE: downsample happens AFTER the transformer in HF's _encode_frame,
    // but SEANet returns here. The caller handles transformer → downsample order.

    *T_out = T_cur;
    return buf;
}

// Encoder transformer (GGML graph, 8 layers)
static std::vector<float> enc_run_transformer(enc_model & m, const std::vector<float> & ch_in, int T) {
    const size_t GMEM = 512ull * 1024 * 1024;
    struct ggml_init_params gp = { GMEM, nullptr, false };
    struct ggml_context * comp = ggml_init(gp);
    if (!comp) { fprintf(stderr, "enc_transformer: ggml_init OOM\n"); return {}; }

    ggml_tensor * inp = ggml_new_tensor_2d(comp, GGML_TYPE_F32, ENC_DIM, T);
    {
        float * dst = (float *)inp->data;
        for (int t = 0; t < T; t++)
            for (int c = 0; c < ENC_DIM; c++)
                dst[c + t * ENC_DIM] = ch_in[c * T + t];
    }
    ggml_tensor * pos = ggml_new_tensor_1d(comp, GGML_TYPE_I32, T);
    for (int i = 0; i < T; i++) ((int32_t *)pos->data)[i] = i;

    ggml_tensor * cur = inp;
    const float scale = 1.f / sqrtf((float)ENC_HEAD_DIM);

    for (int il = 0; il < ENC_N_TR; il++) {
        const auto & layer = m.tr[il];

        ggml_tensor * res1 = cur;
        ggml_tensor * x = ggml_norm(comp, cur, ENC_NORM_EPS);
        x = ggml_mul(comp, x, layer.attn_norm);
        x = ggml_add(comp, x, layer.attn_norm_b);

        ggml_tensor * Q = ggml_mul_mat(comp, layer.attn_q, x);
        ggml_tensor * K = ggml_mul_mat(comp, layer.attn_k, x);
        ggml_tensor * V = ggml_mul_mat(comp, layer.attn_v, x);

        Q = ggml_reshape_3d(comp, Q, ENC_HEAD_DIM, ENC_N_HEADS, T);
        K = ggml_reshape_3d(comp, K, ENC_HEAD_DIM, ENC_N_HEADS, T);
        V = ggml_reshape_3d(comp, V, ENC_HEAD_DIM, ENC_N_HEADS, T);

        Q = ggml_rope_ext(comp, Q, pos, nullptr, ENC_HEAD_DIM, 0, 250,
                          ENC_ROPE_THETA, 1.f, 0.f, 1.f, 0.f, 0.f);
        K = ggml_rope_ext(comp, K, pos, nullptr, ENC_HEAD_DIM, 0, 250,
                          ENC_ROPE_THETA, 1.f, 0.f, 1.f, 0.f, 0.f);

        Q = ggml_cont(comp, ggml_permute(comp, Q, 0, 2, 1, 3));
        K = ggml_cont(comp, ggml_permute(comp, K, 0, 2, 1, 3));
        V = ggml_cont(comp, ggml_permute(comp, V, 1, 2, 0, 3));

        ggml_tensor * attn = ggml_mul_mat(comp, K, Q);
        attn = ggml_scale(comp, attn, scale);
        attn = ggml_diag_mask_inf(comp, attn, 0);
        attn = ggml_soft_max(comp, attn);
        ggml_tensor * ao = ggml_mul_mat(comp, V, attn);

        ao = ggml_reshape_2d(comp, ggml_cont(comp, ggml_permute(comp, ao, 0, 2, 1, 3)), ENC_DIM, T);
        ao = ggml_mul_mat(comp, layer.attn_output, ao);
        ao = ggml_mul(comp, ao, layer.attn_scale);
        cur = ggml_add(comp, res1, ao);

        ggml_tensor * res2 = cur;
        cur = ggml_norm(comp, cur, ENC_NORM_EPS);
        cur = ggml_mul(comp, cur, layer.ffn_norm);
        cur = ggml_add(comp, cur, layer.ffn_norm_b);

        cur = ggml_mul_mat(comp, layer.ffn_up, cur);
        cur = ggml_gelu_erf(comp, cur);
        cur = ggml_mul_mat(comp, layer.ffn_down, cur);
        cur = ggml_mul(comp, cur, layer.ffn_scale);
        cur = ggml_add(comp, res2, cur);
    }

    ggml_cgraph * gf = ggml_new_graph(comp);
    ggml_build_forward_expand(gf, cur);
    struct ggml_cplan plan = ggml_graph_plan(gf, 4, nullptr);
    std::vector<uint8_t> work(plan.work_size > 0 ? plan.work_size : 1);
    plan.work_data = work.data();
    if (ggml_graph_compute(gf, &plan) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "enc_transformer: compute failed\n");
        ggml_free(comp); return {};
    }

    std::vector<float> out(ENC_DIM * T);
    const float * src = (const float *)cur->data;
    for (int t = 0; t < T; t++)
        for (int c = 0; c < ENC_DIM; c++)
            out[c * T + t] = src[c + t * ENC_DIM];

    {
        float rms = 0.f;
        for (int i = 0; i < ENC_DIM * T; i++) rms += out[i] * out[i];
        rms = sqrtf(rms / (ENC_DIM * T));
        printf("  [dbg] transformer out: rms=%.6f, frame0[:5]=[%.6f, %.6f, %.6f, %.6f, %.6f]\n",
               rms, src[0], src[1], src[2], src[3], src[4]);
        if (T > 1) {
            printf("  [dbg] transformer out: frame1[:5]=[%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                   src[ENC_DIM], src[ENC_DIM+1], src[ENC_DIM+2], src[ENC_DIM+3], src[ENC_DIM+4]);
        }
    }

    ggml_free(comp);
    return out;
}

// RVQ encode: 512-dim transformer output → discrete codes
static void enc_vq_encode(enc_model & m, const float * enc_out_tm, int T, int n_q, int32_t * codes) {
    std::vector<float> b_ip0, b_ipr;
    const float * ip0 = enc_get_f32(m.vq_semantic_input_proj, b_ip0);
    const float * ipr = m.vq_acoustic_input_proj ? enc_get_f32(m.vq_acoustic_input_proj, b_ipr) : ip0;

    std::vector<std::vector<float>> cb_norms(n_q, std::vector<float>(ENC_VQ_BINS));
    std::vector<std::vector<float>> cb_data(n_q);
    for (int q = 0; q < n_q; q++) {
        ggml_tensor * cb_t = (q == 0) ? m.vq_semantic_codebook[0] : m.vq_acoustic_codebook[q - 1];
        if (!cb_t) continue;
        std::vector<float> tmp;
        const float * cb = enc_get_f32(cb_t, tmp);
        cb_data[q].assign(cb, cb + ENC_VQ_BINS * ENC_VQ_DIM);
        for (int k = 0; k < ENC_VQ_BINS; k++) {
            const float * e = cb_data[q].data() + k * ENC_VQ_DIM;
            float n2 = 0.f;
            for (int d = 0; d < ENC_VQ_DIM; d++) n2 += e[d] * e[d];
            cb_norms[q][k] = n2;
        }
    }

    auto vq_nearest = [&](const float * z, int q) -> int32_t {
        const float * cb = cb_data[q].data();
        const float * norms = cb_norms[q].data();
        int best_k = 0;
        float best = FLT_MAX;
        for (int k = 0; k < ENC_VQ_BINS; k++) {
            const float * e = cb + k * ENC_VQ_DIM;
            float dot = 0.f;
            for (int d = 0; d < ENC_VQ_DIM; d++) dot += z[d] * e[d];
            float score = norms[k] - 2.f * dot;
            if (score < best) { best = score; best_k = k; }
        }
        return (int32_t)best_k;
    };

    for (int t = 0; t < T; t++) {
        const float * x_t = enc_out_tm + t * ENC_DIM;

        float z_sem[ENC_VQ_DIM] = {};
        for (int co = 0; co < ENC_VQ_DIM; co++) {
            float s = 0.f;
            for (int ci = 0; ci < ENC_DIM; ci++) s += x_t[ci] * ip0[ci + co * ENC_DIM];
            z_sem[co] = s;
        }
        codes[t * n_q + 0] = vq_nearest(z_sem, 0);
        if (n_q <= 1) continue;

        float z_ac[ENC_VQ_DIM] = {};
        for (int co = 0; co < ENC_VQ_DIM; co++) {
            float s = 0.f;
            for (int ci = 0; ci < ENC_DIM; ci++) s += x_t[ci] * ipr[ci + co * ENC_DIM];
            z_ac[co] = s;
        }
        float residual[ENC_VQ_DIM];
        memcpy(residual, z_ac, ENC_VQ_DIM * sizeof(float));
        for (int q = 1; q < n_q; q++) {
            int32_t bk = vq_nearest(residual, q);
            codes[t * n_q + q] = bk;
            const float * e = cb_data[q].data() + bk * ENC_VQ_DIM;
            for (int d = 0; d < ENC_VQ_DIM; d++) residual[d] -= e[d];
        }
    }
}

// Full encode pipeline: SEANet → Transformer → Downsample → VQ
// (HF order: encoder → encoder_transformer → downsample → quantizer)
static int enc_encode_audio(enc_model & m, const float * audio, int n_samples, int n_q, std::vector<std::vector<int32_t>> & out_frames) {
    int T_seanet = 0;
    printf("Speech tokenizer: encoding %d samples (%.2fs)...\n", n_samples, (float)n_samples / 24000.f);

    // Step 1: SEANet encoder (raw audio → 512-dim, channel-major)
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> seanet_out = enc_seanet_encode(m, audio, n_samples, &T_seanet);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (seanet_out.empty()) { fprintf(stderr, "enc_encode: SEANet failed\n"); return -1; }
    printf("  SEANet encoder: %d frames (%.1f ms)\n", T_seanet,
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    // Step 2: Encoder transformer (channel-major in/out)
    auto t2 = std::chrono::high_resolution_clock::now();
    std::vector<float> tr_out = enc_run_transformer(m, seanet_out, T_seanet);
    auto t3 = std::chrono::high_resolution_clock::now();
    if (tr_out.empty()) { fprintf(stderr, "enc_encode: transformer failed\n"); return -1; }
    printf("  Encoder transformer: %.1f ms\n",
           std::chrono::duration<double, std::milli>(t3 - t2).count());

    // Step 3: Downsample (stride=2, k=4) AFTER transformer
    std::vector<float> b_ds;
    const float * wd = enc_get_f32(m.downsample, b_ds);
    const int T_frames = (T_seanet - 1) / 2 + 1;
    std::vector<float> ds_out(ENC_DIM * T_frames);
    enc_causal_conv1d(tr_out.data(), ENC_DIM, T_seanet, wd, ENC_DIM, 4, nullptr, ds_out.data(), 2);
    printf("  Downsample: %d → %d frames\n", T_seanet, T_frames);

    // Convert channel-major → time-major for VQ
    std::vector<float> vq_in(ENC_DIM * T_frames);
    for (int t = 0; t < T_frames; t++)
        for (int c = 0; c < ENC_DIM; c++)
            vq_in[c + t * ENC_DIM] = ds_out[c * T_frames + t];

    {
        FILE * dbg = fopen("tools/tts/_enc_dump/vq_input_cpp.bin", "wb");
        if (dbg) {
            int32_t hdr[2] = { T_frames, ENC_DIM };
            fwrite(hdr, sizeof(int32_t), 2, dbg);
            fwrite(vq_in.data(), sizeof(float), ENC_DIM * T_frames, dbg);
            fclose(dbg);
            printf("  Dumped VQ input (%dx%d) to tools/tts/_enc_dump/vq_input_cpp.bin\n", T_frames, ENC_DIM);
        }
    }

    // Step 4: RVQ quantize
    auto t4 = std::chrono::high_resolution_clock::now();
    std::vector<int32_t> codes(T_frames * n_q);
    enc_vq_encode(m, vq_in.data(), T_frames, n_q, codes.data());
    auto t5 = std::chrono::high_resolution_clock::now();
    printf("  RVQ quantization (%d codebooks): %.1f ms\n", n_q,
           std::chrono::duration<double, std::milli>(t5 - t4).count());

    out_frames.resize(T_frames);
    for (int t = 0; t < T_frames; t++) {
        out_frames[t].resize(n_q);
        for (int q = 0; q < n_q; q++)
            out_frames[t][q] = codes[t * n_q + q];
    }
    printf("  Encoded: %d frames x %d codebooks\n", T_frames, n_q);
    return T_frames;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Vocoder Decoder (WavTokenizer-style)
//  VQ lookup → pre-conv → pre-transformer (8 layers, RoPE) →
//  upsample (ConvNeXt) → decoder blocks (Snake + ConvTranspose + residual) →
//  output conv → tanh → audio
// ═══════════════════════════════════════════════════════════════════════════════

static constexpr int VOC_N_CODEBOOKS  = 16;
static constexpr int VOC_CB_SIZE      = 2048;
static constexpr int VOC_CB_DIM       = 256;
static constexpr int VOC_HIDDEN       = 512;
static constexpr int VOC_LATENT       = 1024;
static constexpr int VOC_N_PRE_TFM    = 8;
static constexpr int VOC_N_HEADS      = 16;
static constexpr int VOC_HEAD_DIM     = 64;
static constexpr int VOC_DEC_DIM      = 1536;
static constexpr float VOC_RMS_EPS    = 1e-5f;
static constexpr float VOC_ROPE_THETA = 10000.0f;
#define VOC_MAX_NODES 32768

struct voc_pre_tfm_layer {
    ggml_tensor * attn_norm_w = nullptr;
    ggml_tensor * attn_q_w   = nullptr;
    ggml_tensor * attn_k_w   = nullptr;
    ggml_tensor * attn_v_w   = nullptr;
    ggml_tensor * attn_out_w = nullptr;
    ggml_tensor * attn_scale = nullptr;
    ggml_tensor * ffn_norm_w = nullptr;
    ggml_tensor * ffn_gate_w = nullptr;
    ggml_tensor * ffn_up_w   = nullptr;
    ggml_tensor * ffn_down_w = nullptr;
    ggml_tensor * ffn_scale  = nullptr;
};

struct voc_residual_block {
    int dilation = 1;
    ggml_tensor * act1_alpha = nullptr;
    ggml_tensor * act1_beta  = nullptr;
    ggml_tensor * conv1_w    = nullptr;
    ggml_tensor * conv1_b    = nullptr;
    ggml_tensor * act2_alpha = nullptr;
    ggml_tensor * act2_beta  = nullptr;
    ggml_tensor * conv2_w    = nullptr;
    ggml_tensor * conv2_b    = nullptr;
};

struct voc_decoder_block {
    ggml_tensor * snake_alpha = nullptr;
    ggml_tensor * snake_beta  = nullptr;
    ggml_tensor * conv_t_w    = nullptr;
    ggml_tensor * conv_t_b    = nullptr;
    voc_residual_block res[3];
};

struct voc_upsample_block {
    ggml_tensor * conv_w    = nullptr;
    ggml_tensor * conv_b    = nullptr;
    ggml_tensor * dwconv_w  = nullptr;
    ggml_tensor * dwconv_b  = nullptr;
    ggml_tensor * norm_w    = nullptr;
    ggml_tensor * norm_b    = nullptr;
    ggml_tensor * pwconv1_w = nullptr;
    ggml_tensor * pwconv1_b = nullptr;
    ggml_tensor * pwconv2_w = nullptr;
    ggml_tensor * pwconv2_b = nullptr;
    ggml_tensor * gamma     = nullptr;
};

struct voc_model {
    ggml_tensor * vq_first_output_proj = nullptr;
    ggml_tensor * vq_first_codebook    = nullptr;
    ggml_tensor * vq_rest_output_proj  = nullptr;
    ggml_tensor * vq_rest_codebook[15] = {};

    voc_upsample_block upsample[2];

    ggml_tensor * pre_tfm_input_proj_w  = nullptr;
    ggml_tensor * pre_tfm_input_proj_b  = nullptr;
    voc_pre_tfm_layer pre_tfm_layers[VOC_N_PRE_TFM];
    ggml_tensor * pre_tfm_norm_w        = nullptr;
    ggml_tensor * pre_tfm_output_proj_w = nullptr;
    ggml_tensor * pre_tfm_output_proj_b = nullptr;

    ggml_tensor * pre_conv_w = nullptr;
    ggml_tensor * pre_conv_b = nullptr;

    ggml_tensor * dec0_conv_w = nullptr;
    ggml_tensor * dec0_conv_b = nullptr;
    voc_decoder_block dec_blocks[4];
    ggml_tensor * dec5_snake_alpha = nullptr;
    ggml_tensor * dec5_snake_beta  = nullptr;
    ggml_tensor * dec6_conv_w      = nullptr;
    ggml_tensor * dec6_conv_b      = nullptr;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
};

static ggml_tensor * voc_ensure_f16(ggml_context * ctx, ggml_tensor * w) {
    return (w->type != GGML_TYPE_F16) ? ggml_cast(ctx, w, GGML_TYPE_F16) : w;
}
// CUDA conv_transpose_1d requires F32 weights+input; cast for the GPU vocoder path.
static ggml_tensor * voc_ensure_f32(ggml_context * ctx, ggml_tensor * w) {
    return (w->type != GGML_TYPE_F32) ? ggml_cast(ctx, w, GGML_TYPE_F32) : w;
}
// On the GPU path, conv_transpose weights must be F32 (CUDA); elsewhere F16 (CPU im2col path).
static ggml_tensor * voc_conv_t_w(ggml_context * ctx, ggml_tensor * w, bool gpu_path) {
    return gpu_path ? voc_ensure_f32(ctx, w) : voc_ensure_f16(ctx, w);
}

static ggml_tensor * voc_snake(ggml_context * ctx, ggml_tensor * x,
                                ggml_tensor * alpha, ggml_tensor * beta) {
    int64_t T = x->ne[0], C = x->ne[1], B = x->ne[2];

    ggml_tensor * a_exp = ggml_exp(ctx, alpha);
    ggml_tensor * a_3d  = ggml_reshape_3d(ctx, a_exp, 1, C, 1);
    ggml_tensor * ref   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, C, B);
    ggml_tensor * a_br  = ggml_repeat(ctx, a_3d, ref);

    ggml_tensor * ax     = ggml_mul(ctx, x, a_br);
    ggml_tensor * sin_ax = ggml_sin(ctx, ax);
    ggml_tensor * sin_sq = ggml_sqr(ctx, sin_ax);

    ggml_tensor * neg_b   = ggml_scale(ctx, beta, -1.0f);
    ggml_tensor * inv_b_e = ggml_exp(ctx, neg_b);
    ggml_tensor * inv_b_3 = ggml_reshape_3d(ctx, inv_b_e, 1, C, 1);
    ggml_tensor * inv_br  = ggml_repeat(ctx, inv_b_3, ref);

    return ggml_add(ctx, x, ggml_mul(ctx, sin_sq, inv_br));
}

static ggml_tensor * voc_rms_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, VOC_RMS_EPS), w);
}

static ggml_tensor * voc_pre_tfm_layer_fwd(ggml_context * ctx, ggml_tensor * x,
                                             const voc_pre_tfm_layer & l,
                                             int n_frames, ggml_tensor * pos) {
    if (!l.attn_norm_w || !l.attn_q_w) return x;

    ggml_tensor * res = x;
    ggml_tensor * n = voc_rms_norm(ctx, x, l.attn_norm_w);

    ggml_tensor * Q = ggml_mul_mat(ctx, l.attn_q_w, n);
    ggml_tensor * K = ggml_mul_mat(ctx, l.attn_k_w, n);
    ggml_tensor * V = ggml_mul_mat(ctx, l.attn_v_w, n);

    Q = ggml_reshape_3d(ctx, Q, VOC_HEAD_DIM, VOC_N_HEADS, n_frames);
    K = ggml_reshape_3d(ctx, K, VOC_HEAD_DIM, VOC_N_HEADS, n_frames);
    V = ggml_reshape_3d(ctx, V, VOC_HEAD_DIM, VOC_N_HEADS, n_frames);

    Q = ggml_rope_ext(ctx, Q, pos, nullptr, VOC_HEAD_DIM, GGML_ROPE_TYPE_NEOX, 0,
                      VOC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(ctx, K, pos, nullptr, VOC_HEAD_DIM, GGML_ROPE_TYPE_NEOX, 0,
                      VOC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
    K = ggml_permute(ctx, K, 0, 2, 1, 3);
    V = ggml_permute(ctx, V, 0, 2, 1, 3);

    ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);
    KQ = ggml_scale(ctx, KQ, 1.0f / sqrtf((float)VOC_HEAD_DIM));
    KQ = ggml_diag_mask_inf(ctx, KQ, 0);
    KQ = ggml_soft_max(ctx, KQ);

    V = ggml_cont(ctx, ggml_transpose(ctx, V));
    ggml_tensor * out = ggml_mul_mat(ctx, V, KQ);
    out = ggml_permute(ctx, out, 0, 2, 1, 3);
    out = ggml_cont_2d(ctx, out, VOC_N_HEADS * VOC_HEAD_DIM, n_frames);
    out = ggml_mul_mat(ctx, l.attn_out_w, out);
    if (l.attn_scale) out = ggml_mul(ctx, out, l.attn_scale);

    x = ggml_add(ctx, res, out);
    res = x;

    n = voc_rms_norm(ctx, x, l.ffn_norm_w);
    ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, l.ffn_gate_w, n));
    ggml_tensor * up   = ggml_mul_mat(ctx, l.ffn_up_w, n);
    ggml_tensor * ffn  = ggml_mul_mat(ctx, l.ffn_down_w, ggml_mul(ctx, gate, up));
    if (l.ffn_scale) ffn = ggml_mul(ctx, ffn, l.ffn_scale);

    return ggml_add(ctx, res, ffn);
}

static ggml_tensor * voc_residual_fwd(ggml_context * ctx, ggml_tensor * x,
                                       const voc_residual_block & blk) {
    ggml_tensor * res = x;
    if (blk.act1_alpha) x = voc_snake(ctx, x, blk.act1_alpha, blk.act1_beta);

    int64_t oc = blk.conv1_w->ne[2];
    int padding = 6 * blk.dilation;
    x = ggml_pad_ext(ctx, x, padding, 0, 0, 0, 0, 0, 0, 0);
    x = ggml_conv_1d(ctx, voc_ensure_f16(ctx, blk.conv1_w), x, 1, 0, blk.dilation);
    if (blk.conv1_b) x = ggml_add(ctx, x, ggml_reshape_3d(ctx, blk.conv1_b, 1, oc, 1));

    if (blk.act2_alpha) x = voc_snake(ctx, x, blk.act2_alpha, blk.act2_beta);

    oc = blk.conv2_w->ne[2];
    x = ggml_conv_1d(ctx, voc_ensure_f16(ctx, blk.conv2_w), x, 1, 0, 1);
    if (blk.conv2_b) x = ggml_add(ctx, x, ggml_reshape_3d(ctx, blk.conv2_b, 1, oc, 1));

    return ggml_add(ctx, res, x);
}

static ggml_tensor * voc_decoder_block_fwd(ggml_context * ctx, ggml_tensor * x,
                                             const voc_decoder_block & blk, int upsample_rate, bool gpu_path) {
    if (blk.snake_alpha) x = voc_snake(ctx, x, blk.snake_alpha, blk.snake_beta);

    int64_t T_in = x->ne[0], ch_in = x->ne[1];
    int64_t ch_out = blk.conv_t_w->ne[1];
    int kernel = (int)blk.conv_t_w->ne[0];

    ggml_tensor * x2d = ggml_reshape_2d(ctx, x, T_in, ch_in);
    x2d = ggml_conv_transpose_1d(ctx, voc_conv_t_w(ctx, blk.conv_t_w, gpu_path), x2d, upsample_rate, 0, 1);

    int64_t T_new = x2d->ne[0];
    x = ggml_reshape_3d(ctx, x2d, T_new, ch_out, 1);

    // CausalTransConvNet: remove right_pad = kernel - stride from the RIGHT only
    int right_pad = kernel - upsample_rate;
    int64_t T_out = T_new - right_pad;
    x = ggml_view_3d(ctx, x, T_out, ch_out, 1, x->nb[1], x->nb[2], 0);
    x = ggml_cont(ctx, x);

    if (blk.conv_t_b)
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, blk.conv_t_b, 1, ch_out, 1));

    for (int i = 0; i < 3; i++)
        x = voc_residual_fwd(ctx, x, blk.res[i]);

    return x;
}

static ggml_tensor * voc_upsample_fwd(ggml_context * ctx, ggml_tensor * x,
                                        const voc_upsample_block & blk, bool gpu_path) {
    int64_t T = x->ne[0], C = x->ne[1];

    ggml_tensor * x2d = ggml_reshape_2d(ctx, x, T, C);
    x2d = ggml_conv_transpose_1d(ctx, voc_conv_t_w(ctx, blk.conv_w, gpu_path), x2d, 2, 0, 1);
    int64_t T_new = x2d->ne[0];
    x = ggml_reshape_3d(ctx, x2d, T_new, C, 1);
    if (blk.conv_b) x = ggml_add(ctx, x, ggml_reshape_3d(ctx, blk.conv_b, 1, C, 1));

    ggml_tensor * res = x;

    if (blk.dwconv_w) {
        x = ggml_pad_ext(ctx, x, 6, 0, 0, 0, 0, 0, 0, 0);
        x = ggml_conv_1d_dw(ctx, voc_ensure_f16(ctx, blk.dwconv_w), x, 1, 0, 1);
        if (blk.dwconv_b) x = ggml_add(ctx, x, ggml_reshape_3d(ctx, blk.dwconv_b, 1, C, 1));
    }

    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);
    if (blk.norm_w && blk.norm_b) {
        x = ggml_norm(ctx, x, 1e-6f);
        x = ggml_mul(ctx, x, blk.norm_w);
        x = ggml_add(ctx, x, blk.norm_b);
    }
    x = ggml_mul_mat(ctx, blk.pwconv1_w, x);
    if (blk.pwconv1_b) x = ggml_add(ctx, x, blk.pwconv1_b);
    x = ggml_gelu(ctx, x);
    x = ggml_mul_mat(ctx, blk.pwconv2_w, x);
    if (blk.pwconv2_b) x = ggml_add(ctx, x, blk.pwconv2_b);
    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);

    if (blk.gamma) {
        ggml_tensor * g3 = ggml_reshape_3d(ctx, blk.gamma, 1, C, 1);
        ggml_tensor * ref = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T_new, C, 1);
        x = ggml_mul(ctx, x, ggml_repeat(ctx, g3, ref));
    }

    return ggml_add(ctx, res, x);
}

template<class Loader>
static bool voc_load_weights(Loader & loader, voc_model & voc) {
    auto get = [&](const char * name) -> ggml_tensor * {
        ggml_tensor * t = loader.get(name);
        if (!t) fprintf(stderr, "WARN: missing vocoder tensor: %s\n", name);
        return t;
    };

    voc.vq_first_codebook    = get("tok_dec.vq_first.0.codebook");
    voc.vq_first_output_proj = get("tok_dec.vq_first.output_proj.weight");
    voc.vq_rest_output_proj  = get("tok_dec.vq_rest.output_proj.weight");
    for (int i = 0; i < 15; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "tok_dec.vq_rest.%d.codebook", i);
        voc.vq_rest_codebook[i] = get(buf);
    }

    voc.pre_conv_w = get("tok_dec.pre_conv.weight");
    voc.pre_conv_b = get("tok_dec.pre_conv.bias");

    voc.pre_tfm_input_proj_w  = get("tok_dec.pre_tfm.input_proj.weight");
    voc.pre_tfm_input_proj_b  = get("tok_dec.pre_tfm.input_proj.bias");
    for (int i = 0; i < VOC_N_PRE_TFM; i++) {
        char buf[128];
        voc_pre_tfm_layer & layer = voc.pre_tfm_layers[i];
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.attn_norm.weight", i);   layer.attn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.attn_q.weight", i);      layer.attn_q_w   = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.attn_k.weight", i);      layer.attn_k_w   = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.attn_v.weight", i);      layer.attn_v_w   = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.attn_output.weight", i); layer.attn_out_w = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.attn_scale", i);         layer.attn_scale = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.ffn_norm.weight", i);   layer.ffn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.ffn_gate.weight", i);   layer.ffn_gate_w = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.ffn_up.weight", i);     layer.ffn_up_w   = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.ffn_down.weight", i);   layer.ffn_down_w = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.ffn_scale", i);         layer.ffn_scale  = get(buf);
    }
    voc.pre_tfm_norm_w         = get("tok_dec.pre_tfm.norm.weight");
    voc.pre_tfm_output_proj_w  = get("tok_dec.pre_tfm.output_proj.weight");
    voc.pre_tfm_output_proj_b  = get("tok_dec.pre_tfm.output_proj.bias");

    for (int i = 0; i < 2; i++) {
        char buf[128];
        voc_upsample_block & blk = voc.upsample[i];
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.conv.weight", i);       blk.conv_w    = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.conv.bias", i);         blk.conv_b    = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.dwconv.weight", i);    blk.dwconv_w  = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.dwconv.bias", i);       blk.dwconv_b  = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.norm.weight", i);      blk.norm_w    = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.norm.bias", i);        blk.norm_b    = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.pwconv1.weight", i);   blk.pwconv1_w = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.pwconv1.bias", i);     blk.pwconv1_b = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.pwconv2.weight", i);   blk.pwconv2_w = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.pwconv2.bias", i);     blk.pwconv2_b = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.gamma", i);            blk.gamma     = get(buf);
    }

    voc.dec0_conv_w = get("tok_dec.dec.0.conv.weight");
    voc.dec0_conv_b = get("tok_dec.dec.0.conv.bias");

    const int dec_dilations[3] = {1, 3, 9};
    for (int db = 0; db < 4; db++) {
        int dec_idx = db + 1;
        voc_decoder_block & blk = voc.dec_blocks[db];
        char buf[128];
        snprintf(buf, sizeof(buf), "tok_dec.dec.%d.snake.alpha", dec_idx); blk.snake_alpha = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.dec.%d.snake.beta", dec_idx);  blk.snake_beta  = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.dec.%d.conv_t.weight", dec_idx); blk.conv_t_w = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.dec.%d.conv_t.bias", dec_idx);   blk.conv_t_b = get(buf);
        for (int r = 0; r < 3; r++) {
            int res_idx = r + 2;
            voc_residual_block & res = blk.res[r];
            res.dilation = dec_dilations[r];
            snprintf(buf, sizeof(buf), "tok_dec.dec.%d.res.%d.act1.alpha", dec_idx, res_idx); res.act1_alpha = get(buf);
            snprintf(buf, sizeof(buf), "tok_dec.dec.%d.res.%d.act1.beta", dec_idx, res_idx);  res.act1_beta  = get(buf);
            snprintf(buf, sizeof(buf), "tok_dec.dec.%d.res.%d.conv1.weight", dec_idx, res_idx); res.conv1_w = get(buf);
            snprintf(buf, sizeof(buf), "tok_dec.dec.%d.res.%d.conv1.bias", dec_idx, res_idx);   res.conv1_b = get(buf);
            snprintf(buf, sizeof(buf), "tok_dec.dec.%d.res.%d.act2.alpha", dec_idx, res_idx); res.act2_alpha = get(buf);
            snprintf(buf, sizeof(buf), "tok_dec.dec.%d.res.%d.act2.beta", dec_idx, res_idx);  res.act2_beta  = get(buf);
            snprintf(buf, sizeof(buf), "tok_dec.dec.%d.res.%d.conv2.weight", dec_idx, res_idx); res.conv2_w = get(buf);
            snprintf(buf, sizeof(buf), "tok_dec.dec.%d.res.%d.conv2.bias", dec_idx, res_idx);   res.conv2_b = get(buf);
        }
    }

    voc.dec5_snake_alpha = get("tok_dec.dec.5.snake.alpha");
    voc.dec5_snake_beta  = get("tok_dec.dec.5.snake.beta");
    voc.dec6_conv_w      = get("tok_dec.dec.6.conv.weight");
    voc.dec6_conv_b      = get("tok_dec.dec.6.conv.bias");

    bool ok = voc.vq_first_codebook && voc.pre_conv_w && voc.dec0_conv_w && voc.dec6_conv_w;
    if (!ok) fprintf(stderr, "ERROR: missing critical vocoder tensors\n");
    return ok;
}

static ggml_cgraph * voc_build_graph(ggml_context * ctx0, const voc_model & voc,
                                      const std::vector<std::vector<int32_t>> & all_codes,
                                      bool gpu_path = false) {
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, VOC_MAX_NODES, false);
    const int n_frames = (int)all_codes.size();
    if (n_frames == 0) return gf;

    // 1. VQ codebook lookup
    std::vector<int32_t> cb0_ids(n_frames), cb_rest_ids[15];
    for (int c = 0; c < 15; c++) cb_rest_ids[c].resize(n_frames);
    for (int f = 0; f < n_frames; f++) {
        cb0_ids[f] = all_codes[f][0];
        for (int c = 1; c < 16; c++) cb_rest_ids[c - 1][f] = all_codes[f][c];
    }

    // ggml_get_rows(codebook[cb_dim, vocab], ids[n]) → [cb_dim, n_frames]
    // We need [n_frames, cb_dim, 1] for conv_1d (sequence along dim0)
    const int cb_dim = VOC_CB_DIM; // 256

    ggml_tensor * cb0_ids_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
    ggml_set_name(cb0_ids_t, "cb0_ids"); ggml_set_input(cb0_ids_t);
    ggml_tensor * vq0 = ggml_get_rows(ctx0, voc.vq_first_codebook, cb0_ids_t);
    // vq0: [cb_dim, n_frames] → permute to [n_frames, cb_dim] → reshape to [n_frames, cb_dim, 1]
    vq0 = ggml_cont(ctx0, ggml_transpose(ctx0, vq0));
    vq0 = ggml_reshape_3d(ctx0, vq0, n_frames, cb_dim, 1);
    vq0 = ggml_conv_1d(ctx0, voc_ensure_f16(ctx0, voc.vq_first_output_proj), vq0, 1, 0, 1);
    ggml_tensor * vq_sum = vq0;

    char buf_name[64];
    for (int c = 0; c < 15; c++) {
        ggml_tensor * ids_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
        snprintf(buf_name, sizeof(buf_name), "cb_rest_ids_%d", c);
        ggml_set_name(ids_t, buf_name);
        ggml_set_input(ids_t);
        ggml_tensor * vqc = ggml_get_rows(ctx0, voc.vq_rest_codebook[c], ids_t);
        vqc = ggml_cont(ctx0, ggml_transpose(ctx0, vqc));
        vqc = ggml_reshape_3d(ctx0, vqc, n_frames, cb_dim, 1);
        vqc = ggml_conv_1d(ctx0, voc_ensure_f16(ctx0, voc.vq_rest_output_proj), vqc, 1, 0, 1);
        vq_sum = ggml_add(ctx0, vq_sum, vqc);
    }

    // VQ output: [n_frames, 512, 1]
    ggml_tensor * cur = vq_sum;

    // 2. Pre-conv: CausalConvNet(512, 1024, kernel=3)
    //    causal padding = kernel_size - stride = 3 - 1 = 2 on the LEFT
    cur = ggml_pad_ext(ctx0, cur, 2, 0, 0, 0, 0, 0, 0, 0);
    cur = ggml_conv_1d(ctx0, voc_ensure_f16(ctx0, voc.pre_conv_w), cur, 1, 0, 1);
    if (voc.pre_conv_b) cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, voc.pre_conv_b, 1, (int)cur->ne[1], 1));

    // 3. Pre-transformer: project 1024→512, 8 transformer layers, project 512→1024
    int64_t T_seq = cur->ne[0];

    // Reshape to [T_seq, 1024] for linear projections (swap to channels-last)
    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_2d(ctx0, cur, (int)cur->ne[0], (int)cur->ne[1]);
    cur = ggml_mul_mat(ctx0, voc.pre_tfm_input_proj_w, cur);
    if (voc.pre_tfm_input_proj_b) cur = ggml_add(ctx0, cur, voc.pre_tfm_input_proj_b);

    ggml_tensor * pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_seq);
    ggml_set_name(pos, "voc_pos");
    ggml_set_input(pos);
    for (int i = 0; i < VOC_N_PRE_TFM; i++)
        cur = voc_pre_tfm_layer_fwd(ctx0, cur, voc.pre_tfm_layers[i], (int)T_seq, pos);

    cur = voc_rms_norm(ctx0, cur, voc.pre_tfm_norm_w);
    cur = ggml_mul_mat(ctx0, voc.pre_tfm_output_proj_w, cur);
    if (voc.pre_tfm_output_proj_b) cur = ggml_add(ctx0, cur, voc.pre_tfm_output_proj_b);

    // Reshape back to [T_seq, 1024, 1] (channels-first for conv)
    cur = ggml_reshape_3d(ctx0, cur, (int)cur->ne[0], (int)cur->ne[1], 1);
    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = ggml_cont(ctx0, cur);

    // No residual from pre-conv (HF does NOT add a skip connection here)
    // 4. Upsample (2 ConvNeXt blocks, each ×2)
    cur = voc_upsample_fwd(ctx0, cur, voc.upsample[0], gpu_path);
    cur = voc_upsample_fwd(ctx0, cur, voc.upsample[1], gpu_path);

    // 5. Decoder: dec0 is CausalConvNet(latent_dim, decoder_dim, kernel=7)
    //    causal padding = kernel_size - stride = 7 - 1 = 6 on the LEFT
    cur = ggml_pad_ext(ctx0, cur, 6, 0, 0, 0, 0, 0, 0, 0);
    cur = ggml_conv_1d(ctx0, voc_ensure_f16(ctx0, voc.dec0_conv_w), cur, 1, 0, 1);
    if (voc.dec0_conv_b) cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, voc.dec0_conv_b, 1, (int)cur->ne[1], 1));

    const int upsample_rates[4] = {8, 5, 4, 3};
    for (int i = 0; i < 4; i++)
        cur = voc_decoder_block_fwd(ctx0, cur, voc.dec_blocks[i], upsample_rates[i], gpu_path);

    // dec5 = SnakeBeta, dec6 = CausalConvNet(output_dim, 1, kernel=7)
    cur = voc_snake(ctx0, cur, voc.dec5_snake_alpha, voc.dec5_snake_beta);
    cur = ggml_pad_ext(ctx0, cur, 6, 0, 0, 0, 0, 0, 0, 0);
    cur = ggml_conv_1d(ctx0, voc_ensure_f16(ctx0, voc.dec6_conv_w), cur, 1, 0, 1);
    if (voc.dec6_conv_b) cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, voc.dec6_conv_b, 1, (int)cur->ne[1], 1));

    // HF uses clamp(-1, 1), not tanh
    cur = ggml_clamp(ctx0, cur, -1.0f, 1.0f);
    ggml_set_name(cur, "audio");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

// Decode frames [start, end) of all_codes into a waveform (CPU). Standalone: allocates and
// frees its own context/backend so it can be called repeatedly for chunked/streaming decode.
// The pre-transformer is causal and uses RoPE, so a window decoded with 0-based positions has
// identical relative geometry to a full decode for every retained frame; only attention mass
// from frames before `start` is dropped, which decays with enough left context. Sample i of the
// result corresponds to global frame `start` onward (offset start*SAMPLES_PER_FRAME).
static std::vector<float> voc_decode_audio_range(const voc_model & voc,
                                                 const std::vector<std::vector<int32_t>> & all_codes,
                                                 int start, int end) {
    std::vector<float> audio_data;
    int n_frames = end - start;
    if (n_frames <= 0) return audio_data;
    std::vector<std::vector<int32_t>> slice(all_codes.begin() + start, all_codes.begin() + end);

    // The CPU backend (with its threadpool) and the graph allocator are expensive to create, so
    // keep them alive across calls. Streaming decodes one window per chunk; re-initializing these
    // every call was the dominant cost (turning ~RT decode into ~0.2x). The allocator grows its
    // buffer to the largest window seen and reuses it. Only the small graph-metadata context is
    // built per call (no_alloc=true => no tensor data here, so a modest arena suffices).
    // thread_local so a pool of vocode workers can each decode concurrently on its own backend
    // (the scheduler thread keeps its own too, used for streaming chunks). Per-backend thread
    // count is capped via QWEN_VOC_THREADS so K workers x T threads doesn't oversubscribe cores.
    thread_local static ggml_backend_t backend = nullptr;
    thread_local static ggml_gallocr_t alloc   = nullptr;
    if (!backend) {
        backend = ggml_backend_cpu_init();
        const char * vt = getenv("QWEN_VOC_THREADS");
        if (vt) ggml_backend_cpu_set_n_threads(backend, std::max(1, atoi(vt)));
        alloc   = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }

    size_t ctx_size = ggml_tensor_overhead() * VOC_MAX_NODES + 16 * 1024 * 1024;
    struct ggml_init_params ctx_params = { ctx_size, nullptr, true };
    ggml_context * voc_ctx = ggml_init(ctx_params);
    ggml_cgraph * voc_gf = voc_build_graph(voc_ctx, voc, slice);

    ggml_tensor * cb0_in    = ggml_graph_get_tensor(voc_gf, "cb0_ids");
    ggml_tensor * audio_out = ggml_graph_get_tensor(voc_gf, "audio");
    ggml_tensor * pos_in    = ggml_graph_get_tensor(voc_gf, "voc_pos");

    ggml_gallocr_alloc_graph(alloc, voc_gf);

    std::vector<int32_t> cb0_ids(n_frames);
    for (int f = 0; f < n_frames; f++) cb0_ids[f] = slice[f][0];
    ggml_backend_tensor_set(cb0_in, cb0_ids.data(), 0, n_frames * sizeof(int32_t));

    char voc_buf_name[64];
    for (int c = 0; c < 15; c++) {
        snprintf(voc_buf_name, sizeof(voc_buf_name), "cb_rest_ids_%d", c);
        ggml_tensor * ids_in = ggml_graph_get_tensor(voc_gf, voc_buf_name);
        if (ids_in) {
            std::vector<int32_t> ids(n_frames);
            for (int f = 0; f < n_frames; f++) ids[f] = slice[f][c + 1];
            ggml_backend_tensor_set(ids_in, ids.data(), 0, n_frames * sizeof(int32_t));
        }
    }

    std::vector<int32_t> pos_data(n_frames);
    for (int i = 0; i < n_frames; i++) pos_data[i] = i;
    ggml_backend_tensor_set(pos_in, pos_data.data(), 0, n_frames * sizeof(int32_t));

    ggml_backend_graph_compute(backend, voc_gf);

    int n_samples = (int)(audio_out->ne[0] * audio_out->ne[1] * audio_out->ne[2]);
    audio_data.resize(n_samples);
    ggml_backend_tensor_get(audio_out, audio_data.data(), 0, n_samples * sizeof(float));

    ggml_free(voc_ctx); // keep the persistent backend + allocator alive for the next chunk
    return audio_data;
}

// Decode the first n_frames of all_codes into a waveform (one-shot, from frame 0).
static std::vector<float> voc_decode_audio(const voc_model & voc,
                                           const std::vector<std::vector<int32_t>> & all_codes,
                                           int n_frames) {
    return voc_decode_audio_range(voc, all_codes, 0, n_frames);
}

// ─── GPU vocoder ──────────────────────────────────────────────────────────────
// The conv vocoder is the dominant cost at finalize (~1.4s/clip on CPU). Mirror its
// weights into a GPU buffer once and run the decode graph through a [GPU, CPU] sched
// (any op CUDA can't take falls back to CPU automatically; conv_transpose_1d uses F32
// weights so it stays on the GPU). One mutex-guarded engine: GPU decode is ~10x faster
// than CPU, so a single instance keeps up with generation without per-thread weight copies.

// A GPU-resident clone of all vocoder weights (same names/shapes), usable as a
// voc_load_weights "loader" (provides .get and a .tensors map).
struct gpu_weight_set {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
    ~gpu_weight_set() { if (buf) ggml_backend_buffer_free(buf); if (ctx) ggml_free(ctx); }
    ggml_tensor * get(const char * name) const { auto it = tensors.find(name); return it!=tensors.end()?it->second:nullptr; }
};

static bool voc_mirror_to_gpu(gguf_tensor_loader & cpu_loader, ggml_backend_t be, gpu_weight_set & gw) {
    gw.ctx = ggml_init({ (cpu_loader.tensors.size()+1)*ggml_tensor_overhead(), nullptr, true });
    for (auto & kv : cpu_loader.tensors) {
        ggml_tensor * s = kv.second;
        ggml_tensor * d = ggml_new_tensor(gw.ctx, s->type, GGML_MAX_DIMS, s->ne);
        ggml_set_name(d, kv.first.c_str());
        gw.tensors[kv.first] = d;
    }
    gw.buf = ggml_backend_alloc_ctx_tensors(gw.ctx, be);
    if (!gw.buf) return false;
    for (auto & kv : cpu_loader.tensors) ggml_backend_tensor_set(gw.tensors[kv.first], kv.second->data, 0, ggml_nbytes(kv.second));
    return true;
}

struct gpu_vocoder {
    ggml_backend_t gpu = nullptr, cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    gpu_weight_set gw;
    voc_model vocg;
    std::mutex mu;
    bool ok = false;

    bool init(gguf_tensor_loader & cpu_loader) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) return false;
        gpu = ggml_backend_dev_init(dev, nullptr);
        if (!gpu) return false;
        cpu = ggml_backend_cpu_init();
        if (!voc_mirror_to_gpu(cpu_loader, gpu, gw)) return false;
        if (!voc_load_weights(gw, vocg)) return false;
        ggml_backend_t backs[2] = { gpu, cpu };
        sched = ggml_backend_sched_new(backs, nullptr, 2, VOC_MAX_NODES, false, true);
        if (!sched) return false;
        ok = true;
        return true;
    }
    ~gpu_vocoder() { if (sched) ggml_backend_sched_free(sched); if (cpu) ggml_backend_free(cpu); if (gpu) ggml_backend_free(gpu); }

    std::vector<float> decode(const std::vector<std::vector<int32_t>> & all, int start, int end) {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<float> audio;
        int n_frames = end - start;
        if (n_frames <= 0) return audio;
        std::vector<std::vector<int32_t>> slice(all.begin()+start, all.begin()+end);

        size_t cs = ggml_tensor_overhead()*VOC_MAX_NODES + 32*1024*1024;
        ggml_context * c = ggml_init({ cs, nullptr, true });
        // Default: conv_transpose runs on the CPU (sched routes just those ops there) because
        // ggml's CUDA conv_transpose_1d kernel is ~20x slower; everything else runs on the GPU.
        // ~330ms/clip vs ~33s if conv_transpose is forced onto CUDA. QWEN_VOC_CONVT_GPU=1 forces
        // the (slow) all-GPU path for experiments.
        bool gp = getenv("QWEN_VOC_CONVT_GPU") && atoi(getenv("QWEN_VOC_CONVT_GPU"));
        ggml_cgraph * gf = voc_build_graph(c, vocg, slice, gp);

        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) { ggml_free(c); return audio; }

        std::vector<int32_t> ids(n_frames);
        ggml_tensor * cb0_in = ggml_graph_get_tensor(gf, "cb0_ids");
        for (int f=0; f<n_frames; f++) ids[f] = slice[f][0];
        ggml_backend_tensor_set(cb0_in, ids.data(), 0, n_frames*sizeof(int32_t));
        char nm[64];
        for (int cci=0; cci<15; cci++) {
            snprintf(nm, sizeof(nm), "cb_rest_ids_%d", cci);
            ggml_tensor * t = ggml_graph_get_tensor(gf, nm);
            if (t) { for (int f=0; f<n_frames; f++) ids[f] = slice[f][cci+1]; ggml_backend_tensor_set(t, ids.data(), 0, n_frames*sizeof(int32_t)); }
        }
        ggml_tensor * pos_in = ggml_graph_get_tensor(gf, "voc_pos");
        for (int f=0; f<n_frames; f++) ids[f] = f;
        ggml_backend_tensor_set(pos_in, ids.data(), 0, n_frames*sizeof(int32_t));

        auto _t0 = std::chrono::steady_clock::now();
        ggml_backend_sched_graph_compute(sched, gf);

        ggml_tensor * ao = ggml_graph_get_tensor(gf, "audio");
        int ns = (int)ggml_nelements(ao);
        audio.resize(ns);
        ggml_backend_tensor_get(ao, audio.data(), 0, ns*sizeof(float));
        if (getenv("QWEN_PROFILE")) {
            double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-_t0).count();
            fprintf(stderr, "[gpuvoc] %d frames -> %d samples (%.2fs audio) in %.1f ms\n", n_frames, ns, ns/24000.0, ms);
        }
        ggml_free(c);
        return audio;
    }
};

// Dequantize a (bf16/f16/f32) tensor's full contents to an f32 buffer, preserving layout.
static void dequant_to_f32(const ggml_tensor * t, float * out) {
    int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) { memcpy(out, t->data, (size_t)n * sizeof(float)); }
    else if (t->type == GGML_TYPE_F16) { const ggml_fp16_t * s=(const ggml_fp16_t*)t->data; for (int64_t i=0;i<n;i++) out[i]=ggml_fp16_to_fp32(s[i]); }
    else if (t->type == GGML_TYPE_BF16) { ggml_bf16_to_fp32_row((const ggml_bf16_t*)t->data, out, n); }
    else { memset(out, 0, (size_t)n*sizeof(float)); }
}

// Persistent GPU GEMM engine for the per-frame "head" matmuls (codec_head, CP lm_heads,
// small_to_mtp projection). These otherwise run on the CPU (read_row + dot) and dominate
// batched throughput. Weights are uploaded once (f32); each call runs one mul_mat on the GPU
// across all active slots: out[out_dim, N] = mul_mat(weight[in_dim,out_dim], x[in_dim,N]).
struct gpu_gemm {
    ggml_backend_t be = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_context * wctx = nullptr;
    struct spec { int in, out; const ggml_tensor * src; };
    std::vector<spec> specs;
    std::vector<ggml_tensor*> w;

    // Embedding tables for fused on-GPU gather (lever b): codec_embd + the 15 CP
    // codec_embd tables. Resident f32 on the GPU; gather_proj / gather_sum_pad run
    // ggml_get_rows directly (and fuse the small_to_mtp projection) so the per-frame
    // codec-embedding lookups stop hitting the CPU.
    struct espec { int ne0, ne1; const ggml_tensor * src; };
    std::vector<espec> especs;
    std::vector<ggml_tensor*> etab;
    int s2m_w = -1;                 // index into w[] of small_to_mtp, or -1 => identity
    std::vector<float> pad_host;    // tts_pad embedding (added in gather_sum_pad)
    ggml_tensor * pad_t = nullptr;  // [n_embd] on GPU

    bool ok() const { return be != nullptr; }
    bool init() {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) return false;
        be = ggml_backend_dev_init(dev, nullptr);
        if (!be) return false;
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        return true;
    }
    int add(const ggml_tensor * src) { specs.push_back({(int)src->ne[0], (int)src->ne[1], src}); return (int)specs.size()-1; }
    int add_table(const ggml_tensor * src) { especs.push_back({(int)src->ne[0], (int)src->ne[1], src}); return (int)especs.size()-1; }
    void set_s2m_w(int idx) { s2m_w = idx; }
    void set_pad(const float * p, int n) { pad_host.assign(p, p+n); }
    bool finalize() {
        if (!be) return false;
        size_t ntens = specs.size() + especs.size() + (pad_host.empty()?0:1);
        wctx = ggml_init({ ntens*ggml_tensor_overhead() + 4096, nullptr, true });
        for (auto & s : specs)  w.push_back(ggml_new_tensor_2d(wctx, GGML_TYPE_F32, s.in, s.out));
        for (auto & e : especs) etab.push_back(ggml_new_tensor_2d(wctx, GGML_TYPE_F32, e.ne0, e.ne1));
        if (!pad_host.empty()) pad_t = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, (int)pad_host.size());
        if (!ggml_backend_alloc_ctx_tensors(wctx, be)) return false;
        for (size_t i=0;i<specs.size();i++) {
            std::vector<float> buf((size_t)specs[i].in * specs[i].out);
            dequant_to_f32(specs[i].src, buf.data());
            ggml_backend_tensor_set(w[i], buf.data(), 0, buf.size()*sizeof(float));
        }
        for (size_t i=0;i<especs.size();i++) {
            std::vector<float> buf((size_t)especs[i].ne0 * especs[i].ne1);
            dequant_to_f32(especs[i].src, buf.data());
            ggml_backend_tensor_set(etab[i], buf.data(), 0, buf.size()*sizeof(float));
        }
        if (pad_t) ggml_backend_tensor_set(pad_t, pad_host.data(), 0, pad_host.size()*sizeof(float));
        return true;
    }
    // x: [in_dim, N] row-major (slot-major: x[n*in + i]); out: [out_dim, N] (out[n*out + o]).
    void run(int idx, const float * x, int N, std::vector<float> & out) {
        int in=specs[idx].in, o=specs[idx].out;
        ggml_context * c = ggml_init({ ggml_tensor_overhead()*8 + ggml_graph_overhead() + 4096, nullptr, true });
        ggml_tensor * X = ggml_new_tensor_2d(c, GGML_TYPE_F32, in, N);
        ggml_tensor * Y = ggml_mul_mat(c, w[idx], X);
        ggml_cgraph * gf = ggml_new_graph(c); ggml_build_forward_expand(gf, Y);
        ggml_gallocr_alloc_graph(alloc, gf);
        ggml_backend_tensor_set(X, x, 0, (size_t)in*N*sizeof(float));
        ggml_backend_graph_compute(be, gf);
        out.resize((size_t)o*N);
        ggml_backend_tensor_get(Y, out.data(), 0, out.size()*sizeof(float));
        ggml_free(c);
    }

    // Gather one row per slot from embedding table tidx (ids[k] = token for slot k),
    // optionally project through small_to_mtp. out is slot-major: out[k*out_dim + o].
    void gather_proj(int tidx, const int32_t * ids, int N, std::vector<float> & out) {
        ggml_context * c = ggml_init({ ggml_tensor_overhead()*8 + ggml_graph_overhead() + 4096, nullptr, true });
        ggml_tensor * I = ggml_new_tensor_1d(c, GGML_TYPE_I32, N);
        ggml_tensor * R = ggml_get_rows(c, etab[tidx], I);              // [n_embd, N]
        ggml_tensor * Y = (s2m_w>=0) ? ggml_mul_mat(c, w[s2m_w], R) : R; // [n_embd_cp, N]
        ggml_cgraph * gf = ggml_new_graph(c); ggml_build_forward_expand(gf, Y);
        ggml_gallocr_alloc_graph(alloc, gf);
        ggml_backend_tensor_set(I, ids, 0, (size_t)N*sizeof(int32_t));
        ggml_backend_graph_compute(be, gf);
        int o = (int)Y->ne[0];
        out.resize((size_t)o*N);
        ggml_backend_tensor_get(Y, out.data(), 0, out.size()*sizeof(float));
        ggml_free(c);
    }

    // Summed-frame talker embedding: for slot k, out[:,k] = pad + sum_c get_rows(table[tidxs[c]], ids[c*N+k]).
    // ids is laid out [ncb][N]. out is slot-major: out[k*n_embd + o]. No projection (talker dim).
    void gather_sum_pad(const int * tidxs, int ncb, const int32_t * ids, int N, std::vector<float> & out) {
        ggml_context * c = ggml_init({ ggml_tensor_overhead()*(size_t)(ncb*2+8) + ggml_graph_overhead() + 4096, nullptr, true });
        ggml_tensor * acc = nullptr;
        std::vector<ggml_tensor*> Is(ncb);
        for (int j=0;j<ncb;j++) {
            Is[j] = ggml_new_tensor_1d(c, GGML_TYPE_I32, N);
            ggml_tensor * R = ggml_get_rows(c, etab[tidxs[j]], Is[j]);
            acc = acc ? ggml_add(c, acc, R) : R;
        }
        if (pad_t) acc = ggml_add(c, acc, ggml_reshape_2d(c, pad_t, pad_t->ne[0], 1));
        ggml_cgraph * gf = ggml_new_graph(c); ggml_build_forward_expand(gf, acc);
        ggml_gallocr_alloc_graph(alloc, gf);
        for (int j=0;j<ncb;j++) ggml_backend_tensor_set(Is[j], ids + (size_t)j*N, 0, (size_t)N*sizeof(int32_t));
        ggml_backend_graph_compute(be, gf);
        int o = (int)acc->ne[0];
        out.resize((size_t)o*N);
        ggml_backend_tensor_get(acc, out.data(), 0, out.size()*sizeof(float));
        ggml_free(c);
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Sampling (temperature, top-k, top-p, repetition penalty)
// ═══════════════════════════════════════════════════════════════════════════════

struct tts_sampler_params {
    float temp          = 0.9f;
    int   top_k         = 50;
    float top_p         = 1.0f;
    float rep_penalty   = 1.05f;
    int   rep_last_n    = 64;
    bool  greedy        = false;
    std::vector<int32_t> suppress_tokens;
};

static int32_t tts_sample(
        const float * logits, int n_vocab,
        const tts_sampler_params & params,
        const std::vector<int32_t> & recent_tokens,
        std::mt19937 & rng) {

    std::vector<std::pair<float, int32_t>> candidates(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        candidates[i] = {logits[i], (int32_t)i};
    }

    for (int32_t tok : params.suppress_tokens) {
        if (tok >= 0 && tok < n_vocab) {
            candidates[tok].first = -std::numeric_limits<float>::infinity();
        }
    }

    if (params.greedy) {
        int32_t best = 0;
        float best_v = candidates[0].first;
        for (int i = 1; i < n_vocab; i++) {
            if (candidates[i].first > best_v) { best_v = candidates[i].first; best = i; }
        }
        return best;
    }

    // Repetition penalty
    if (params.rep_penalty != 1.0f && !recent_tokens.empty()) {
        int lookback = std::min((int)recent_tokens.size(), params.rep_last_n);
        for (int k = (int)recent_tokens.size() - lookback; k < (int)recent_tokens.size(); k++) {
            int32_t tok = recent_tokens[k];
            if (tok >= 0 && tok < n_vocab) {
                if (candidates[tok].first > 0.0f) {
                    candidates[tok].first /= params.rep_penalty;
                } else {
                    candidates[tok].first *= params.rep_penalty;
                }
            }
        }
    }

    // Temperature
    float temp = std::max(params.temp, 1e-8f);
    for (auto & c : candidates) c.first /= temp;

    // Top-k: keep only top_k highest
    if (params.top_k > 0 && params.top_k < n_vocab) {
        std::partial_sort(candidates.begin(), candidates.begin() + params.top_k, candidates.end(),
                          [](const auto & a, const auto & b) { return a.first > b.first; });
        candidates.resize(params.top_k);
    } else {
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto & a, const auto & b) { return a.first > b.first; });
    }

    // Softmax
    float max_logit = candidates[0].first;
    float sum_exp = 0.0f;
    for (auto & c : candidates) {
        c.first = expf(c.first - max_logit);
        sum_exp += c.first;
    }
    for (auto & c : candidates) c.first /= sum_exp;

    // Top-p (nucleus)
    if (params.top_p > 0.0f && params.top_p < 1.0f) {
        float cum = 0.0f;
        int cutoff = (int)candidates.size();
        for (int i = 0; i < (int)candidates.size(); i++) {
            cum += candidates[i].first;
            if (cum >= params.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        candidates.resize(cutoff);
        sum_exp = 0.0f;
        for (auto & c : candidates) sum_exp += c.first;
        for (auto & c : candidates) c.first /= sum_exp;
    }

    // Weighted random sample
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cum = 0.0f;
    for (auto & c : candidates) {
        cum += c.first;
        if (r <= cum) return c.second;
    }
    return candidates.back().second;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════════

static const std::map<std::string, uint32_t> LANGUAGE_IDS = {
    {"english",    2050}, {"chinese",    2055}, {"german",  2053},
    {"spanish",    2054}, {"french",     2061}, {"italian", 2070},
    {"japanese",   2058}, {"korean",     2064}, {"portuguese", 2071},
    {"russian",    2069}, {"auto",       0},
};

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Qwen3-TTS: text-to-speech via llama.cpp\n\n"
        "Usage:\n"
        "  %s --model-talker <talker.gguf> \\\n"
        "     --model-cp <code-predictor.gguf> \\\n"
        "     --model-vocoder <tokenizer.gguf> \\\n"
        "     --text \"Hello world\" \\\n"
        "     --output <output.wav>\n\n"
        "Required:\n"
        "  --model-talker        Talker GGUF (contains speaker encoder + LLM)\n"
        "  --model-cp            Code Predictor GGUF\n"
        "  --text                Input text to synthesize\n\n"
        "Voice cloning:\n"
        "  --ref-audio <wav>     Reference audio for speaker cloning\n"
        "  --ref-text <text>     Reference transcript for ICL cloning (requires --ref-audio)\n"
        "  --ref-codes <file>    Precomputed codec codes file (optional for ICL)\n"
        "                        If omitted, codes are auto-encoded from ref-audio using built-in encoder\n\n"
        "Language:\n"
        "  --language <lang>     Target language (default: english)\n"
        "                        Supported: english, chinese, german, spanish, french,\n"
        "                        italian, japanese, korean, portuguese, russian, auto\n\n"
        "Sampling:\n"
        "  --temp <float>        Talker temperature (default: 0.9, 0 = greedy)\n"
        "  --top-k <int>         Talker top-k (default: 50, 0 = disabled)\n"
        "  --top-p <float>       Talker top-p / nucleus (default: 1.0)\n"
        "  --rep-penalty <float> Repetition penalty (default: 1.05, 1.0 = off)\n"
        "  --cp-temp <float>     Code Predictor temperature (default: 0.9)\n"
        "  --cp-top-k <int>      Code Predictor top-k (default: 50)\n"
        "  --greedy              Force greedy decoding (overrides temp/top-k)\n"
        "  --seed <int>          Random seed (default: random)\n\n"
        "Other options:\n"
        "  --model-vocoder       Tokenizer GGUF (vocoder decoder)\n"
        "  --output              Output WAV path (default: output.wav)\n"
        "  --max-tokens N        Max decode frames (default: 2048)\n"
        "  --streaming-text      Enable streaming text mode (feed text progressively)\n"
        "  --n-gpu-layers N      Number of GPU layers (default: 0)\n"
        "  --dump-intermediates  Directory to dump codec codes for parity testing\n\n",
        prog);
}

int main(int argc, char ** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    // Parse arguments
    std::string talker_path, cp_path, vocoder_path;
    std::string ref_audio_path, ref_text, ref_codes_path;
    std::string text, output_path = "output.wav";
    std::string dump_dir;
    std::string language = "english";
    int max_tokens = 2048;
    int stream_chunk = 0; // >0: vocode + emit every N frames (streaming); 0: one-shot at end
    int stream_chunk_initial = 0; // >0: small first chunk for low TTFB, then ramp up to stream_chunk
    int stream_left_ctx = 32; // frames of left context re-decoded per streaming chunk (bounds cost)
    int serve_slots = 0; // >1: enable the warm continuous-batching engine with this many slots
    int serve_port  = 8080;
    bool serve_http = false; // --serve: run the HTTP/WS server (vs the one-shot CLI path)
    int n_gpu = 0;
    int cp_n_gpu = -1; // code predictor GPU layers; -1 => same as talker. NOTE: CP currently
                       // produces NaN on the CPU backend (ggml CPU-backend issue in the CP graph),
                       // so keep it on GPU until that is fixed.
    bool streaming_text = false;

    tts_sampler_params talker_sparams;
    tts_sampler_params cp_sparams;
    int seed = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model-talker") == 0 && i + 1 < argc) talker_path = argv[++i];
        else if (strcmp(argv[i], "--model-cp") == 0 && i + 1 < argc) cp_path = argv[++i];
        else if (strcmp(argv[i], "--model-vocoder") == 0 && i + 1 < argc) vocoder_path = argv[++i];
        else if (strcmp(argv[i], "--ref-audio") == 0 && i + 1 < argc) ref_audio_path = argv[++i];
        else if (strcmp(argv[i], "--ref-text") == 0 && i + 1 < argc) ref_text = argv[++i];
        else if (strcmp(argv[i], "--ref-codes") == 0 && i + 1 < argc) ref_codes_path = argv[++i];
        else if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) text = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--stream-left-ctx") == 0 && i + 1 < argc) stream_left_ctx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--serve-slots") == 0 && i + 1 < argc) serve_slots = atoi(argv[++i]);
        else if (strcmp(argv[i], "--serve-port") == 0 && i + 1 < argc) serve_port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--serve") == 0) serve_http = true;
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) talker_sparams.temp = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) talker_sparams.top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) talker_sparams.top_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--rep-penalty") == 0 && i + 1 < argc) talker_sparams.rep_penalty = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--cp-temp") == 0 && i + 1 < argc) cp_sparams.temp = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--cp-top-k") == 0 && i + 1 < argc) cp_sparams.top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--greedy") == 0) { talker_sparams.greedy = true; cp_sparams.greedy = true; }
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n-gpu-layers") == 0 && i + 1 < argc) n_gpu = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cp-n-gpu-layers") == 0 && i + 1 < argc) cp_n_gpu = atoi(argv[++i]);
        else if (strcmp(argv[i], "--stream-chunk") == 0 && i + 1 < argc) stream_chunk = atoi(argv[++i]);
        else if (strcmp(argv[i], "--stream-chunk-initial") == 0 && i + 1 < argc) stream_chunk_initial = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dump-intermediates") == 0 && i + 1 < argc) dump_dir = argv[++i];
        else if (strcmp(argv[i], "--language") == 0 && i + 1 < argc) language = argv[++i];
        else if (strcmp(argv[i], "--streaming-text") == 0) streaming_text = true;
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]); return 0;
        }
    }

    // --temp 0 implies greedy
    if (talker_sparams.temp <= 0.0f) talker_sparams.greedy = true;
    if (cp_sparams.temp <= 0.0f) cp_sparams.greedy = true;

    if (talker_path.empty() || cp_path.empty()) {
        fprintf(stderr, "ERROR: --model-talker and --model-cp are required\n");
        return 1;
    }
    if (serve_http && serve_slots <= 0) serve_slots = 4; // default parallelism for the server
    const int n_slots = std::max(1, serve_slots);

    if (text.empty() && !serve_http) {
        fprintf(stderr, "ERROR: --text is required\n");
        return 1;
    }
    if (serve_http && text.empty()) text = "warm up"; // throwaway prefill; server re-prefills per request

    // ── Load Talker model ──────────────────────────────────────────────────
    printf("Loading Talker model: %s\n", talker_path.c_str());
    llama_model_params talker_mparams = llama_model_default_params();
    talker_mparams.n_gpu_layers = n_gpu;
    llama_model * talker_model = llama_model_load_from_file(talker_path.c_str(), talker_mparams);
    if (!talker_model) { fprintf(stderr, "ERROR: failed to load talker model\n"); return 1; }

    // ── Load Code Predictor model ──────────────────────────────────────────
    printf("Loading Code Predictor model: %s\n", cp_path.c_str());
    llama_model_params cp_mparams = llama_model_default_params();
    // The code predictor does 15 sequential single-token decodes per frame; on the GPU each
    // pays ~5ms of dispatch latency. CPU would avoid that, but the CP graph currently emits
    // NaN on the CPU backend, so default to the talker's device (GPU) unless overridden.
    cp_mparams.n_gpu_layers = (cp_n_gpu >= 0) ? cp_n_gpu : n_gpu;
    llama_model * cp_model = llama_model_load_from_file(cp_path.c_str(), cp_mparams);
    if (!cp_model) { fprintf(stderr, "ERROR: failed to load code predictor model\n"); return 1; }

    // ── Create Talker context ──────────────────────────────────────────────
    llama_context_params talker_cparams = llama_context_default_params();
    talker_cparams.n_ctx      = 4096 * n_slots; // per-seq 4096; total KV across slots
    talker_cparams.n_batch    = 512;
    talker_cparams.n_seq_max  = n_slots;
    talker_cparams.no_perf    = true;
    talker_cparams.embeddings = true;
    // Models are fully GPU-offloaded, so the CPU threadpool mostly spin-waits on GPU completion
    // and steals cores from the vocode pool. Cap it (QWEN_LLAMA_THREADS) to free cores for vocode.
    int llama_threads = getenv("QWEN_LLAMA_THREADS") ? std::max(1, atoi(getenv("QWEN_LLAMA_THREADS"))) : 0;
    if (llama_threads) { talker_cparams.n_threads = llama_threads; talker_cparams.n_threads_batch = llama_threads; }
    llama_context * talker_ctx = llama_init_from_model(talker_model, talker_cparams);
    if (!talker_ctx) { fprintf(stderr, "ERROR: failed to create talker context\n"); return 1; }

    // ── Create Code Predictor context ──────────────────────────────────────
    llama_context_params cp_cparams = llama_context_default_params();
    cp_cparams.n_ctx      = 32 * n_slots; // 17 positions/frame per seq; total across slots
    cp_cparams.n_batch    = 32 * n_slots;
    cp_cparams.n_seq_max  = n_slots;
    cp_cparams.no_perf    = true;
    cp_cparams.embeddings = true;
    cp_cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp_cparams.type_k     = GGML_TYPE_F32;
    cp_cparams.type_v     = GGML_TYPE_F32;
    if (llama_threads) { cp_cparams.n_threads = llama_threads; cp_cparams.n_threads_batch = llama_threads; }
    llama_context * cp_ctx = llama_init_from_model(cp_model, cp_cparams);
    if (!cp_ctx) { fprintf(stderr, "ERROR: failed to create code predictor context\n"); return 1; }

    printf("Models loaded successfully.\n");

    // ── Initialize RNG ────────────────────────────────────────────────
    if (seed < 0) {
        seed = (int)std::random_device{}();
    }
    std::mt19937 rng((uint32_t)seed);
    printf("Sampling: temp=%.2f top_k=%d top_p=%.2f rep_penalty=%.2f %s (seed=%d)\n",
           talker_sparams.temp, talker_sparams.top_k, talker_sparams.top_p,
           talker_sparams.rep_penalty,
           talker_sparams.greedy ? "[greedy]" : "",
           seed);
    if (!cp_sparams.greedy) {
        printf("CP sampling: temp=%.2f top_k=%d\n", cp_sparams.temp, cp_sparams.top_k);
    }

    // ── Declare GGUF tensor loaders before any goto cleanup ────────────
    gguf_tensor_loader tts_loader;
    gguf_tensor_loader cp_loader;
    gguf_tensor_loader voc_loader;

    // Load the vocoder once up front so it can be used both for streaming (during the
    // decode loop) and for one-shot decode at the end.
    voc_model voc;
    bool voc_ready = false;
    if (!vocoder_path.empty()) {
        if (voc_loader.load(vocoder_path.c_str(), "tok_dec.") && voc_load_weights(voc_loader, voc)) {
            voc_ready = true;
            printf("Vocoder loaded: %s\n", vocoder_path.c_str());
        } else {
            fprintf(stderr, "WARN: failed to load vocoder; audio generation disabled\n");
        }
    }

    // ── Load reference audio (shared by speaker encoder + speech tokenizer) ──
    std::vector<float> ref_audio_samples;
    std::vector<float> speaker_embedding; // sized from the encoder output (1024 for 0.6B, 2048 for 1.7B)
    if (!ref_audio_path.empty()) {
        if (!read_wav(ref_audio_path.c_str(), ref_audio_samples)) {
            fprintf(stderr, "ERROR: cannot read reference audio\n");
            goto cleanup;
        }
    }

    // ── Speaker encoding ───────────────────────────────────────────────────
    if (!ref_audio_path.empty()) {
        printf("Extracting speaker embedding from: %s\n", ref_audio_path.c_str());

        gguf_tensor_loader spk_loader;
        if (spk_loader.load(talker_path.c_str(), "spk_enc.")) {
            spk_encoder_model spk_model;
            if (spk_load_weights(spk_loader, spk_model)) {
                int n_frames = 0;
                std::vector<float> mel_data;
                if (compute_mel_spectrogram(ref_audio_samples.data(), (int)ref_audio_samples.size(), mel_data, n_frames)) {
                    size_t ctx_size = ggml_tensor_overhead() * SPK_MAX_NODES + 256 * 1024 * 1024;
                    struct ggml_init_params ctx_params = { ctx_size, nullptr, true };
                    ggml_context * spk_ctx = ggml_init(ctx_params);
                    ggml_cgraph * spk_gf = spk_build_graph(spk_ctx, spk_model, n_frames);

                    ggml_tensor * mel_in = ggml_graph_get_tensor(spk_gf, "mel_input");
                    ggml_tensor * emb_out = ggml_graph_get_tensor(spk_gf, "embedding");
                    speaker_embedding.assign((size_t)ggml_nelements(emb_out), 0.0f);

                    ggml_backend_t backend = ggml_backend_cpu_init();
                    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                    ggml_gallocr_alloc_graph(alloc, spk_gf);

                    ggml_backend_tensor_set(mel_in, mel_data.data(), 0, mel_data.size() * sizeof(float));
                    ggml_backend_graph_compute(backend, spk_gf);
                    ggml_backend_tensor_get(emb_out, speaker_embedding.data(), 0, speaker_embedding.size() * sizeof(float));

                    printf("  Speaker embedding extracted (first 3: %.4f %.4f %.4f)\n",
                           speaker_embedding[0], speaker_embedding[1], speaker_embedding[2]);

                    ggml_gallocr_free(alloc);
                    ggml_backend_free(backend);
                    ggml_free(spk_ctx);
                }
            }
        } else {
            printf("  No speaker encoder weights found, using zero embedding.\n");
        }
    } else {
        printf("No reference audio provided, using default speaker embedding.\n");
    }

    // ── Load text projection + codec embedding weights from GGUF ─────────
    if (!tts_loader.load(talker_path.c_str(), "tts.")) {
        fprintf(stderr, "ERROR: cannot load TTS tensors from Talker GGUF\n");
        goto cleanup;
    }

    {
        ggml_tensor * w_text_embd    = tts_loader.get("tts.text_embd.weight");
        ggml_tensor * w_proj_up      = tts_loader.get("tts.text_proj_up.weight");
        ggml_tensor * w_proj_up_b    = tts_loader.get("tts.text_proj_up.bias");
        ggml_tensor * w_proj_down    = tts_loader.get("tts.text_proj_down.weight");
        ggml_tensor * w_proj_down_b  = tts_loader.get("tts.text_proj_down.bias");
        ggml_tensor * w_codec_embd   = tts_loader.get("tts.codec_embd.weight");
        ggml_tensor * w_codec_head   = tts_loader.get("tts.codec_head.weight");

        if (!w_text_embd || !w_proj_up || !w_proj_down || !w_codec_embd || !w_codec_head) {
            fprintf(stderr, "ERROR: missing required TTS tensors\n");
            goto cleanup;
        }

        const int n_embd      = (int)llama_model_n_embd(talker_model);
        const int n_text_embd = (int)w_text_embd->ne[0]; // 2048

        printf("Text embd dim: %d, model embd: %d\n", n_text_embd, n_embd);
        // Codec embd: [n_embd, vocab], Codec head: [n_embd, vocab]

        // ── Read special IDs from GGUF metadata ─────────────────────────
        // These were stored in the Talker GGUF during conversion
        auto get_meta_u32 = [&](const char * key, uint32_t def) -> uint32_t {
            int idx = gguf_find_key(tts_loader.guf, key);
            if (idx < 0) return def;
            return (uint32_t)gguf_get_val_u32(tts_loader.guf, idx);
        };

        const uint32_t tts_bos_id     = get_meta_u32("qwen3tts.tts_bos_token_id",  151672);
        const uint32_t tts_eos_id     = get_meta_u32("qwen3tts.tts_eos_token_id",  151673);
        const uint32_t tts_pad_id     = get_meta_u32("qwen3tts.tts_pad_token_id",  151671);
        const uint32_t codec_bos_id   = get_meta_u32("qwen3tts.codec.bos_id",      2149);
        const uint32_t codec_eos_id   = get_meta_u32("qwen3tts.codec.eos_id",      2150);
        const uint32_t codec_pad_id   = get_meta_u32("qwen3tts.codec.pad_id",      2148);
        const uint32_t codec_nothink  = get_meta_u32("qwen3tts.codec.nothink_id",  2155);
        const uint32_t codec_think_bos = get_meta_u32("qwen3tts.codec.think_bos_id", 2156);
        const uint32_t codec_think_eos = get_meta_u32("qwen3tts.codec.think_eos_id", 2157);
        const uint32_t codec_think_id = get_meta_u32("qwen3tts.codec.think_id",    2154);
        printf("Special IDs: tts_bos=%u tts_eos=%u tts_pad=%u codec_bos=%u codec_eos=%u codec_pad=%u\n",
               tts_bos_id, tts_eos_id, tts_pad_id, codec_bos_id, codec_eos_id, codec_pad_id);

        // Resolve language → codec language ID
        std::string lang_lower = language;
        std::transform(lang_lower.begin(), lang_lower.end(), lang_lower.begin(), ::tolower);
        auto lang_it = LANGUAGE_IDS.find(lang_lower);
        if (lang_it == LANGUAGE_IDS.end()) {
            fprintf(stderr, "ERROR: unsupported language '%s'\nSupported:", language.c_str());
            for (auto & kv : LANGUAGE_IDS) fprintf(stderr, " %s", kv.first.c_str());
            fprintf(stderr, "\n");
            goto cleanup;
        }
        uint32_t codec_lang_id = lang_it->second;
        printf("Language: %s (codec_id=%u)\n", language.c_str(), codec_lang_id);

        // ── Tokenize input text ─────────────────────────────────────────
        // Wrap in chat template: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
        const llama_vocab * vocab = llama_model_get_vocab(talker_model);
        printf("Talker vocab size: %d\n", llama_vocab_n_tokens(vocab));

        std::string tts_prompt = "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
        std::vector<llama_token> tokens(tts_prompt.size() + 32);
        int n_tokens = llama_tokenize(vocab, tts_prompt.c_str(), (int)tts_prompt.size(),
                                       tokens.data(), (int)tokens.size(),
                                       false, true);
        if (n_tokens < 0) {
            tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, tts_prompt.c_str(), (int)tts_prompt.size(),
                                       tokens.data(), (int)tokens.size(),
                                       false, true);
        }
        tokens.resize(n_tokens);
        printf("Input text: \"%s\" (%d tokens)\n", text.c_str(), n_tokens);
        printf("Token IDs:");
        for (int i = 0; i < n_tokens; i++) printf(" %d", tokens[i]);
        printf("\n");

        // ── CPU helpers for text projection and codec embedding ────────
        // The GGUF tensor loader reads weights with no_alloc=false, so
        // tensor->data points to CPU memory directly.

        auto read_row = [](const ggml_tensor * t, int64_t row, float * out, int64_t cols) {
            if (t->type == GGML_TYPE_F16) {
                const ggml_fp16_t * src = (const ggml_fp16_t *)t->data + row * cols;
                for (int64_t i = 0; i < cols; i++) out[i] = ggml_fp16_to_fp32(src[i]);
            } else if (t->type == GGML_TYPE_BF16) {
                const ggml_bf16_t * src = (const ggml_bf16_t *)t->data + row * cols;
                ggml_bf16_to_fp32_row(src, out, cols);
            } else {
                const float * src = (const float *)t->data + row * cols;
                memcpy(out, src, cols * sizeof(float));
            }
        };

        auto read_bias = [](const ggml_tensor * t, float * out, int64_t n) {
            if (!t) return;
            if (t->type == GGML_TYPE_F16) {
                const ggml_fp16_t * src = (const ggml_fp16_t *)t->data;
                for (int64_t i = 0; i < n; i++) out[i] += ggml_fp16_to_fp32(src[i]);
            } else if (t->type == GGML_TYPE_BF16) {
                const ggml_bf16_t * src = (const ggml_bf16_t *)t->data;
                for (int64_t i = 0; i < n; i++) { float v; ggml_bf16_to_fp32_row(src + i, &v, 1); out[i] += v; }
            } else {
                const float * src = (const float *)t->data;
                for (int64_t i = 0; i < n; i++) out[i] += src[i];
            }
        };

        // text_projection(token_ids): text_embd → fc1 → silu → fc2
        // text_embd: [n_text_embd, n_text_vocab]  (GGML stores row-major, ne[0]=n_text_embd)
        // proj_up:   [n_text_embd, n_text_embd]
        // proj_down: [n_text_embd, n_embd]
        auto text_project = [&](const std::vector<int32_t> & ids) -> std::vector<float> {
            int n = (int)ids.size();
            std::vector<float> result(n * n_embd, 0.0f);
            std::vector<float> emb_row(n_text_embd);
            std::vector<float> fc1_out(n_text_embd);
            std::vector<float> fc2_out(n_embd);

            for (int t = 0; t < n; t++) {
                int32_t id = ids[t];
                read_row(w_text_embd, id, emb_row.data(), n_text_embd);

                // fc1 = emb_row @ proj_up^T + bias
                // proj_up is [n_text_embd, n_text_embd] in GGML: ne[0]=n_text_embd, ne[1]=n_text_embd
                // Row i of proj_up = output neuron i's weights
                for (int i = 0; i < n_text_embd; i++) {
                    float sum = 0.0f;
                    std::vector<float> w_row(n_text_embd);
                    read_row(w_proj_up, i, w_row.data(), n_text_embd);
                    for (int j = 0; j < n_text_embd; j++) sum += emb_row[j] * w_row[j];
                    fc1_out[i] = sum;
                }
                read_bias(w_proj_up_b, fc1_out.data(), n_text_embd);

                // silu activation
                for (int i = 0; i < n_text_embd; i++) {
                    float x = fc1_out[i];
                    fc1_out[i] = x / (1.0f + expf(-x));
                }

                // fc2 = fc1_out @ proj_down^T + bias
                // proj_down: [n_text_embd, n_embd] → ne[0]=n_text_embd, ne[1]=n_embd
                for (int i = 0; i < n_embd; i++) {
                    float sum = 0.0f;
                    std::vector<float> w_row(n_text_embd);
                    read_row(w_proj_down, i, w_row.data(), n_text_embd);
                    for (int j = 0; j < n_text_embd; j++) sum += fc1_out[j] * w_row[j];
                    fc2_out[i] = sum;
                }
                read_bias(w_proj_down_b, fc2_out.data(), n_embd);

                memcpy(&result[t * n_embd], fc2_out.data(), n_embd * sizeof(float));
            }
            return result;
        };

        // Read a row from codec_embd: [n_embd, n_codec_vocab]
        auto codec_embed = [&](int32_t id) -> std::vector<float> {
            std::vector<float> row(n_embd);
            read_row(w_codec_embd, id, row.data(), n_embd);
            return row;
        };

        // ── Compute text projections ────────────────────────────────────
        // Project special TTS tokens
        std::vector<int32_t> special_ids = {(int32_t)tts_bos_id, (int32_t)tts_eos_id, (int32_t)tts_pad_id};
        std::vector<float> special_proj = text_project(special_ids);
        // special_proj layout: [tts_bos(n_embd), tts_eos(n_embd), tts_pad(n_embd)]
        const float * tts_bos_embed = special_proj.data();
        const float * tts_eos_embed = special_proj.data() + n_embd;
        const float * tts_pad_embed = special_proj.data() + 2 * n_embd;

        // tts_bos and tts_pad embeddings loaded

        // Project all text tokens (including role tokens at the front)
        std::vector<int32_t> text_ids(tokens.begin(), tokens.end());
        std::vector<float> text_proj = text_project(text_ids);

        // text projection computed

        // ── Determine cloning mode ────────────────────────────────────
        bool has_speaker   = !ref_audio_path.empty();
        bool icl_mode      = has_speaker && !ref_text.empty() && (!ref_codes_path.empty() || !vocoder_path.empty());

        // ── Load or compute ICL reference codes ──────────────────────
        std::vector<std::vector<int32_t>> ref_code_frames;
        if (!ref_codes_path.empty()) {
            std::ifstream rc_file(ref_codes_path);
            if (!rc_file.is_open()) {
                fprintf(stderr, "ERROR: cannot open ref-codes file: %s\n", ref_codes_path.c_str());
                goto cleanup;
            }
            std::string line;
            while (std::getline(rc_file, line)) {
                if (line.empty()) continue;
                std::istringstream iss(line);
                std::vector<int32_t> frame;
                int32_t v;
                while (iss >> v) frame.push_back(v);
                if (!frame.empty()) ref_code_frames.push_back(frame);
            }
            printf("Loaded %d ref_code frames from %s\n", (int)ref_code_frames.size(), ref_codes_path.c_str());
            if (ref_text.empty()) {
                fprintf(stderr, "WARNING: --ref-codes provided without --ref-text, ICL disabled\n");
                icl_mode = false;
            }
        } else if (icl_mode && !vocoder_path.empty() && !ref_audio_samples.empty()) {
            printf("Auto-encoding reference audio for ICL mode...\n");
            gguf_tensor_loader enc_loader;
            if (enc_loader.load(vocoder_path.c_str(), "tok_enc.")) {
                enc_model enc;
                if (enc_load_weights(enc_loader, enc)) {
                    int n_active_q = 16;
                    int result = enc_encode_audio(enc, ref_audio_samples.data(), (int)ref_audio_samples.size(), n_active_q, ref_code_frames);
                    if (result <= 0) {
                        fprintf(stderr, "WARNING: speech tokenizer encoding failed, falling back to x-vector mode\n");
                        icl_mode = false;
                    } else if (!dump_dir.empty()) {
#ifdef _WIN32
                        _mkdir(dump_dir.c_str());
#else
                        mkdir(dump_dir.c_str(), 0755);
#endif
                        std::string codes_path = dump_dir + "/ref_codes_cpp.txt";
                        FILE * fp = fopen(codes_path.c_str(), "w");
                        if (fp) {
                            for (const auto & frame : ref_code_frames) {
                                for (size_t q = 0; q < frame.size(); q++) {
                                    if (q > 0) fprintf(fp, " ");
                                    fprintf(fp, "%d", frame[q]);
                                }
                                fprintf(fp, "\n");
                            }
                            fclose(fp);
                            printf("  Dumped ref codes to %s\n", codes_path.c_str());
                        }
                    }
                } else {
                    fprintf(stderr, "WARNING: encoder weights not found in vocoder GGUF, falling back to x-vector mode\n");
                    icl_mode = false;
                }
            } else {
                fprintf(stderr, "WARNING: cannot load encoder from vocoder GGUF, falling back to x-vector mode\n");
                icl_mode = false;
            }
        }

        if (icl_mode) {
            printf("Voice cloning mode: ICL (in-context learning)\n");
        } else if (has_speaker) {
            printf("Voice cloning mode: x-vector only\n");
        } else {
            printf("Voice cloning mode: none (default voice)\n");
        }

        // ── Get codec embeddings for prefill ────────────────────────────
        std::vector<uint32_t> codec_prefill_ids;
        if (codec_lang_id != 0) {
            codec_prefill_ids = {codec_think_id, codec_think_bos, codec_lang_id, codec_think_eos};
        } else {
            codec_prefill_ids = {codec_nothink, codec_think_bos, codec_think_eos};
        }

        std::vector<std::vector<float>> codec_prefill_embeds;
        for (uint32_t id : codec_prefill_ids)
            codec_prefill_embeds.push_back(codec_embed((int32_t)id));

        if (has_speaker) {
            codec_prefill_embeds.push_back(speaker_embedding);
        }

        codec_prefill_embeds.push_back(codec_embed((int32_t)codec_pad_id));
        codec_prefill_embeds.push_back(codec_embed((int32_t)codec_bos_id));

        int n_codec_pre = (int)codec_prefill_embeds.size();

        // ── Build prefill embedding ──────────────────────────────────
        // tokens layout: [<|im_start|>, assistant, \n, text..., <|im_end|>, \n, <|im_start|>, assistant, \n]
        // tokens[:3] = role prefix, tokens[3:-5] = text content, tokens[-5:] = suffix
        std::vector<float> codec_pad_emb = codec_embed((int32_t)codec_pad_id);

        const int n_role = 3;
        const int n_text_content = std::max(0, n_tokens - 3 - 5);
        const int n_codec_preamble = n_codec_pre - 1;

        int n_prefill;
        std::vector<float> prefill_embed;

        // In ICL mode the model re-synthesizes the reference text before the target text, so the
        // first ~n_ref_frames generated frames are the reference being spoken back. They are kept
        // as vocoder left-context but trimmed from the emitted audio (matches the reference impl).
        int icl_ref_frames = 0;

        if (icl_mode) {
            // ICL mode: generate_icl_prompt
            // Tokenize ref_text with the same chat template wrapper
            std::string ref_prompt = "<|im_start|>assistant\n" + ref_text + "<|im_end|>\n<|im_start|>assistant\n";
            std::vector<llama_token> ref_tokens(ref_prompt.size() + 32);
            int n_ref_tokens = llama_tokenize(vocab, ref_prompt.c_str(), (int)ref_prompt.size(),
                                               ref_tokens.data(), (int)ref_tokens.size(),
                                               false, true);
            if (n_ref_tokens < 0) {
                ref_tokens.resize(-n_ref_tokens);
                n_ref_tokens = llama_tokenize(vocab, ref_prompt.c_str(), (int)ref_prompt.size(),
                                               ref_tokens.data(), (int)ref_tokens.size(),
                                               false, true);
            }
            ref_tokens.resize(n_ref_tokens);
            int n_ref_content = std::max(0, n_ref_tokens - 3 - 5);

            // Project reference text tokens
            std::vector<int32_t> ref_ids(ref_tokens.begin(), ref_tokens.end());
            std::vector<float> ref_proj = text_project(ref_ids);

            // ICL text = ref_text[3:-5] + target_text[3:-5]
            int n_icl_text = n_ref_content + n_text_content;
            int n_icl_text_section = n_icl_text + 1; // +1 for eos

            // Build codec embed for each ref frame:
            // sum of codec_embed[codebook_i](code) over all 16 codebooks
            int n_ref_frames = (int)ref_code_frames.size();
            icl_ref_frames = n_ref_frames; // trim this many leading frames from the output
            std::vector<std::vector<float>> ref_codec_embeds(n_ref_frames, std::vector<float>(n_embd, 0.0f));

            // Load CP codec embeddings for codebooks 1-15
            if (cp_loader.tensors.empty()) {
                cp_loader.load(cp_path.c_str(), "tts.cp.");
            }

            for (int f = 0; f < n_ref_frames; f++) {
                auto & frame = ref_code_frames[f];
                // cb0: from talker codec_embed
                std::vector<float> cb0_emb = codec_embed(frame[0]);
                for (int j = 0; j < n_embd; j++) ref_codec_embeds[f][j] += cb0_emb[j];
                // cb1..15: from code predictor's per-codebook embeddings
                for (int cb = 1; cb < 16 && cb < (int)frame.size(); cb++) {
                    char tname[64];
                    snprintf(tname, sizeof(tname), "tts.cp.codec_embd.%d.weight", cb - 1);
                    ggml_tensor * cp_embd = cp_loader.get(tname);
                    if (cp_embd) {
                        std::vector<float> cb_emb(n_embd);
                        read_row(cp_embd, frame[cb], cb_emb.data(), n_embd);
                        for (int j = 0; j < n_embd; j++) ref_codec_embeds[f][j] += cb_emb[j];
                    }
                }
            }

            // ICL prefill layout (non-streaming):
            //   [0..2]: role = text_proj(tokens[:3])
            //   [3..3+n_codec_preamble-1]: tts_pad*(n-2) + tts_bos overlaid on codec_embed_preamble[:-1]
            //   ICL prompt (from generate_icl_prompt):
            //     ref_text + target_text + eos overlaid on codec_pad
            //     codec_bos + ref_codec_embeds overlaid on tts_pad
            //   final: tts_pad + codec_bos

            // generate_icl_prompt returns:
            //   text_embed = text_proj(ref_tokens[3:-5] + target_tokens[3:-5]) + eos_embed
            //   codec_embed_icl = codec_bos + sum_of_all_codebook_embeds for each ref frame
            //   In non-streaming mode:
            //     icl_input = text_embed + codec_pad overlay
            //     codec part = codec_embed_icl + tts_pad overlay
            //     concat: [text_with_pad, codec_with_pad]
            // Then: [role, preamble, icl_input, final_tts_pad+codec_bos]

            int n_icl_input = n_icl_text_section + n_ref_frames + 1; // text+eos + bos + ref frames
            n_prefill = n_role + n_codec_preamble + n_icl_input + 1; // +1 for final

            prefill_embed.resize(n_prefill * n_embd, 0.0f);
            int pos = 0;

            // (a) Role tokens
            for (int i = 0; i < n_role && i < n_tokens; i++) {
                memcpy(&prefill_embed[pos * n_embd], &text_proj[i * n_embd], n_embd * sizeof(float));
                pos++;
            }

            // (b) Codec preamble (same as x-vector mode)
            for (int i = 0; i < n_codec_pre - 2; i++) {
                for (int j = 0; j < n_embd; j++) {
                    prefill_embed[pos * n_embd + j] = tts_pad_embed[j] + codec_prefill_embeds[i][j];
                }
                pos++;
            }
            for (int j = 0; j < n_embd; j++) {
                prefill_embed[pos * n_embd + j] = tts_bos_embed[j] + codec_prefill_embeds[n_codec_pre - 2][j];
            }
            pos++;

            // (c) ICL text section: (ref_text + target_text + eos) overlaid on codec_pad
            for (int i = 0; i < n_ref_content; i++) {
                int tok_idx = 3 + i;
                for (int j = 0; j < n_embd; j++) {
                    prefill_embed[pos * n_embd + j] = ref_proj[tok_idx * n_embd + j] + codec_pad_emb[j];
                }
                pos++;
            }
            for (int i = 0; i < n_text_content; i++) {
                int tok_idx = 3 + i;
                for (int j = 0; j < n_embd; j++) {
                    prefill_embed[pos * n_embd + j] = text_proj[tok_idx * n_embd + j] + codec_pad_emb[j];
                }
                pos++;
            }
            // tts_eos + codec_pad
            for (int j = 0; j < n_embd; j++) {
                prefill_embed[pos * n_embd + j] = tts_eos_embed[j] + codec_pad_emb[j];
            }
            pos++;

            // (d) ICL codec section: codec_bos + ref_codec_embeds, overlaid with tts_pad
            std::vector<float> codec_bos_emb = codec_embed((int32_t)codec_bos_id);
            for (int j = 0; j < n_embd; j++) {
                prefill_embed[pos * n_embd + j] = tts_pad_embed[j] + codec_bos_emb[j];
            }
            pos++;
            for (int f = 0; f < n_ref_frames; f++) {
                for (int j = 0; j < n_embd; j++) {
                    prefill_embed[pos * n_embd + j] = tts_pad_embed[j] + ref_codec_embeds[f][j];
                }
                pos++;
            }

            // (e) Final: tts_pad + codec_bos
            for (int j = 0; j < n_embd; j++) {
                prefill_embed[pos * n_embd + j] = tts_pad_embed[j] + codec_prefill_embeds[n_codec_pre - 1][j];
            }
            pos++;

            GGML_ASSERT(pos == n_prefill);

        } else if (streaming_text && !icl_mode) {
            // Streaming text mode: prefill = role + preamble + first_text_token+codec_bos
            // Remaining text fed one-by-one during decode via trailing_text
            n_prefill = n_role + n_codec_preamble + 1; // +1 for first_text+codec_bos

            prefill_embed.resize(n_prefill * n_embd, 0.0f);
            int pos = 0;

            // (a) Role tokens
            for (int i = 0; i < n_role && i < n_tokens; i++) {
                memcpy(&prefill_embed[pos * n_embd], &text_proj[i * n_embd], n_embd * sizeof(float));
                pos++;
            }

            // (b) Codec preamble
            for (int i = 0; i < n_codec_pre - 2; i++) {
                for (int j = 0; j < n_embd; j++) {
                    prefill_embed[pos * n_embd + j] = tts_pad_embed[j] + codec_prefill_embeds[i][j];
                }
                pos++;
            }
            for (int j = 0; j < n_embd; j++) {
                prefill_embed[pos * n_embd + j] = tts_bos_embed[j] + codec_prefill_embeds[n_codec_pre - 2][j];
            }
            pos++;

            // (c) First text token + codec_bos
            if (n_text_content > 0) {
                for (int j = 0; j < n_embd; j++) {
                    prefill_embed[pos * n_embd + j] = text_proj[3 * n_embd + j]
                                                      + codec_prefill_embeds[n_codec_pre - 1][j];
                }
            } else {
                for (int j = 0; j < n_embd; j++) {
                    prefill_embed[pos * n_embd + j] = tts_pad_embed[j]
                                                      + codec_prefill_embeds[n_codec_pre - 1][j];
                }
            }
            pos++;

            GGML_ASSERT(pos == n_prefill);

        } else {
            // Non-streaming mode (default)
            const int n_text_section = n_text_content + 1;
            n_prefill = n_role + n_codec_preamble + n_text_section + 1;

            prefill_embed.resize(n_prefill * n_embd, 0.0f);
            int pos = 0;

            // (a) Role tokens
            for (int i = 0; i < n_role && i < n_tokens; i++) {
                memcpy(&prefill_embed[pos * n_embd], &text_proj[i * n_embd], n_embd * sizeof(float));
                pos++;
            }

            // (b) Codec preamble
            for (int i = 0; i < n_codec_pre - 2; i++) {
                for (int j = 0; j < n_embd; j++) {
                    prefill_embed[pos * n_embd + j] = tts_pad_embed[j] + codec_prefill_embeds[i][j];
                }
                pos++;
            }
            for (int j = 0; j < n_embd; j++) {
                prefill_embed[pos * n_embd + j] = tts_bos_embed[j] + codec_prefill_embeds[n_codec_pre - 2][j];
            }
            pos++;

            // (c) All text content + eos, overlaid on codec_pad
            for (int i = 0; i < n_text_content; i++) {
                int tok_idx = 3 + i;
                for (int j = 0; j < n_embd; j++) {
                    prefill_embed[pos * n_embd + j] = text_proj[tok_idx * n_embd + j] + codec_pad_emb[j];
                }
                pos++;
            }
            for (int j = 0; j < n_embd; j++) {
                prefill_embed[pos * n_embd + j] = tts_eos_embed[j] + codec_pad_emb[j];
            }
            pos++;

            // (d) Final: tts_pad + codec_bos
            for (int j = 0; j < n_embd; j++) {
                prefill_embed[pos * n_embd + j] = tts_pad_embed[j] + codec_prefill_embeds[n_codec_pre - 1][j];
            }
            pos++;

            GGML_ASSERT(pos == n_prefill);
        }

        // ── Trailing text for streaming mode ─────────────────────────
        std::vector<float> trailing_text;
        int n_trailing_text = 0;
        std::vector<float> pad_embed_vec(tts_pad_embed, tts_pad_embed + n_embd);

        if (streaming_text && !icl_mode && n_text_content > 1) {
            // tokens[4:-5] = remaining text content (skip first text token already in prefill)
            // + tts_eos at the end
            int n_remaining = n_text_content - 1;
            n_trailing_text = n_remaining + 1; // +1 for eos
            trailing_text.resize(n_trailing_text * n_embd);
            for (int i = 0; i < n_remaining; i++) {
                int tok_idx = 4 + i; // tokens[4..n_tokens-6]
                memcpy(&trailing_text[i * n_embd], &text_proj[tok_idx * n_embd], n_embd * sizeof(float));
            }
            memcpy(&trailing_text[n_remaining * n_embd], tts_eos_embed, n_embd * sizeof(float));
        } else {
            trailing_text.resize(n_embd);
            memcpy(trailing_text.data(), tts_pad_embed, n_embd * sizeof(float));
            n_trailing_text = 0;
        }

        printf("Prefill: %d positions (%d trailing text)%s\n",
               n_prefill, n_trailing_text,
               streaming_text ? " [streaming]" : "");

        // ── Send prefill embeddings to Talker ───────────────────────────
        // The Talker uses MRoPE (n_pos_per_embd=4). When sending embeddings
        // (not tokens), llama_batch expects explicit multi-dim positions.
        // pos layout: [dim0*n, dim1*n, dim2*n, dim3*n] where each dim has
        // n_tokens entries. For TTS, all 3 active dims use the same position
        // and dim3 is 0.

        const int n_pos_per_embd = 4; // iMRoPE
        llama_batch batch = llama_batch_init(512, n_embd, 1);

        // Reallocate pos for multi-dim positions
        free(batch.pos);
        batch.pos = (llama_pos *)malloc(sizeof(llama_pos) * 512 * n_pos_per_embd);
        memset(batch.pos, 0, sizeof(llama_pos) * 512 * n_pos_per_embd);

        batch.n_tokens = n_prefill;

        for (int i = 0; i < n_prefill; i++) {
            memcpy(batch.embd + i * n_embd, &prefill_embed[i * n_embd], n_embd * sizeof(float));
            // MRoPE: dims 0,1,2 get sequential position; dim 3 = 0
            for (int d = 0; d < 3; d++) {
                batch.pos[d * n_prefill + i] = i;
            }
            batch.pos[3 * n_prefill + i] = 0;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]   = (i == n_prefill - 1) ? 1 : 0;
        }

        printf("\nRunning Talker prefill (%d embeddings)...\n", n_prefill);
        auto t_prefill_start = std::chrono::high_resolution_clock::now();
        int ret = llama_decode(talker_ctx, batch);
        auto t_prefill_end = std::chrono::high_resolution_clock::now();
        if (ret != 0) {
            fprintf(stderr, "ERROR: Talker prefill decode failed: %d\n", ret);
            llama_batch_free(batch);
            goto cleanup;
        }

        double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();

        // Get hidden state from last position (the one with logits=1)
        float * hidden = llama_get_embeddings_ith(talker_ctx, -1);
        if (!hidden) {
            fprintf(stderr, "ERROR: no embeddings from Talker prefill\n");
            llama_batch_free(batch);
            goto cleanup;
        }

        printf("Talker prefill OK.  (%d tokens in %.1f ms = %.1f tok/s)\n",
               n_prefill, prefill_ms, n_prefill / (prefill_ms / 1000.0));

        // ── Apply codec_head externally to get cb0 logits ───────────────
        // logits = hidden @ codec_head^T
        // codec_head: [n_embd, n_codec_vocab] → row i = output token i's weights
        int n_codec_vocab = (int)w_codec_head->ne[1]; // 3072

        // HF suppresses tokens [vocab_size-1024, vocab_size) except EOS during Talker generation.
        // These are reserved special tokens that should never appear in codec output.
        {
            int suppress_start = n_codec_vocab - 1024;
            for (int i = suppress_start; i < n_codec_vocab; i++) {
                if (i != (int)codec_eos_id) {
                    talker_sparams.suppress_tokens.push_back((int32_t)i);
                }
            }
            printf("Talker: suppressing %d tokens [%d..%d) except EOS=%d\n",
                   (int)talker_sparams.suppress_tokens.size(), suppress_start, n_codec_vocab, (int)codec_eos_id);
        }

        auto compute_logits = [&](const float * h, std::vector<float> & out_logits) {
            out_logits.resize(n_codec_vocab);
            std::vector<float> w_row(n_embd);
            for (int i = 0; i < n_codec_vocab; i++) {
                read_row(w_codec_head, i, w_row.data(), n_embd);
                float sum = 0.0f;
                for (int j = 0; j < n_embd; j++) sum += h[j] * w_row[j];
                out_logits[i] = sum;
            }
        };

        std::vector<float> cb0_logits;
        compute_logits(hidden, cb0_logits);

        std::vector<int32_t> talker_recent_tokens;
        int cb0_token = tts_sample(cb0_logits.data(), n_codec_vocab,
                                    talker_sparams, talker_recent_tokens, rng);
        talker_recent_tokens.push_back(cb0_token);

        printf("Prefill cb0 token: %d (logit=%.4f)\n", cb0_token, cb0_logits[cb0_token]);

        // ── Load CP per-codebook weights ────────────────────────────────
        if (!cp_loader.load(cp_path.c_str(), "tts.cp.")) {
            fprintf(stderr, "ERROR: cannot load CP tensors from Code Predictor GGUF\n");
            llama_batch_free(batch);
            goto cleanup;
        }

        const int n_codebooks = 15; // cb1..cb15
        const int cp_vocab = 2048;
        ggml_tensor * cp_lm_heads[15]   = {};
        ggml_tensor * cp_codec_embds[15] = {};

        for (int i = 0; i < n_codebooks; i++) {
            char buf[128];
            snprintf(buf, sizeof(buf), "tts.cp.lm_head.%d.weight", i);
            cp_lm_heads[i] = cp_loader.get(buf);
            snprintf(buf, sizeof(buf), "tts.cp.codec_embd.%d.weight", i);
            cp_codec_embds[i] = cp_loader.get(buf);
            if (!cp_lm_heads[i] || !cp_codec_embds[i]) {
                fprintf(stderr, "ERROR: missing CP tensor: %s\n", buf);
                llama_batch_free(batch);
                goto cleanup;
            }
        }
        printf("Code Predictor per-codebook tensors loaded (%d lm_heads, %d codec_embeds)\n",
               n_codebooks, n_codebooks);

        // CP hidden size (1024). For larger talkers it is smaller than the talker hidden
        // (n_embd), so CP inputs are projected from n_embd -> n_embd_cp by small_to_mtp.
        const int n_embd_cp = (int)llama_model_n_embd(cp_model);
        ggml_tensor * w_cp_s2m = cp_loader.get("tts.cp.small_to_mtp.weight"); // [n_embd, n_embd_cp] or null
        ggml_tensor * b_cp_s2m = cp_loader.get("tts.cp.small_to_mtp.bias");   // [n_embd_cp] or null
        if (w_cp_s2m) {
            printf("CP talker->predictor projection: %d -> %d\n", n_embd, n_embd_cp);
        }

        // Project a talker-space embedding (n_embd) into CP hidden space (n_embd_cp).
        // When the dims match (e.g. 0.6B, no small_to_mtp) this is the identity.
        auto project_cp = [&](const std::vector<float> & in) -> std::vector<float> {
            if (!w_cp_s2m) return in;
            std::vector<float> out(n_embd_cp, 0.0f);
            std::vector<float> w_row(n_embd);
            for (int o = 0; o < n_embd_cp; o++) {
                read_row(w_cp_s2m, o, w_row.data(), n_embd);
                float acc = 0.0f;
                for (int i = 0; i < n_embd; i++) acc += w_row[i] * in[i];
                out[o] = acc;
            }
            read_bias(b_cp_s2m, out.data(), n_embd_cp);
            return out;
        };

        // CP helpers: read row from per-codebook embedding, apply lm_head. CP codec embeddings
        // live in talker space (n_embd) and are projected to CP space at feed time; lm_heads
        // map CP hidden (n_embd_cp) -> vocab.
        auto cp_codec_embed = [&](int cb_idx, int32_t token_id) -> std::vector<float> {
            std::vector<float> row(n_embd);
            read_row(cp_codec_embds[cb_idx], token_id, row.data(), n_embd);
            return row;
        };

        auto cp_compute_logits = [&](int cb_idx, const float * h, std::vector<float> & out) {
            out.resize(cp_vocab);
            std::vector<float> w_row(n_embd_cp);
            for (int i = 0; i < cp_vocab; i++) {
                read_row(cp_lm_heads[cb_idx], i, w_row.data(), n_embd_cp);
                float sum = 0.0f;
                for (int j = 0; j < n_embd_cp; j++) sum += h[j] * w_row[j];
                out[i] = sum;
            }
        };

        // ════════════════════════════════════════════════════════════════════
        //  Phase 2: warm HTTP server with continuous batching (sglang-omni API).
        //  Models stay resident; requests arrive via HTTP, are admitted into free
        //  slots (per-request prefill), and decode together through the batched
        //  engine. One scheduler thread owns the llama contexts; HTTP handlers only
        //  touch the request queue + per-request promise.
        // ════════════════════════════════════════════════════════════════════
        if (serve_http) {
            const int N = n_slots;
            llama_memory_t tmem = llama_get_memory(talker_ctx);
            llama_memory_t cmem = llama_get_memory(cp_ctx);

            // Voice cloning: load the speaker encoder (talker GGUF) and the ICL audio encoder
            // (vocoder GGUF) once; both run on CPU and are reused across requests.
            gguf_tensor_loader spk_loader, enc_loader; spk_encoder_model spk_model; enc_model enc_m;
            bool spk_ok = spk_loader.load(talker_path.c_str(), "spk_enc.") && spk_load_weights(spk_loader, spk_model);
            bool enc_ok = !vocoder_path.empty() && enc_loader.load(vocoder_path.c_str(), "tok_enc.") && enc_load_weights(enc_loader, enc_m);
            printf("[serve] speaker_encoder=%s  icl_audio_encoder=%s\n", spk_ok?"on":"off", enc_ok?"on":"off");

            auto extract_spk = [&](const std::vector<float> & samples) -> std::vector<float> {
                std::vector<float> emb;
                if (!spk_ok) return emb;
                int nf=0; std::vector<float> mel;
                if (!compute_mel_spectrogram(samples.data(), (int)samples.size(), mel, nf)) return emb;
                size_t cs = ggml_tensor_overhead()*SPK_MAX_NODES + 256*1024*1024;
                ggml_context * c = ggml_init({cs, nullptr, true});
                ggml_cgraph * gf = spk_build_graph(c, spk_model, nf);
                ggml_tensor * mi = ggml_graph_get_tensor(gf, "mel_input");
                ggml_tensor * eo = ggml_graph_get_tensor(gf, "embedding");
                emb.assign((size_t)ggml_nelements(eo), 0.f);
                ggml_backend_t b = ggml_backend_cpu_init();
                ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(b));
                ggml_gallocr_alloc_graph(al, gf);
                ggml_backend_tensor_set(mi, mel.data(), 0, mel.size()*sizeof(float));
                ggml_backend_graph_compute(b, gf);
                ggml_backend_tensor_get(eo, emb.data(), 0, emb.size()*sizeof(float));
                ggml_gallocr_free(al); ggml_backend_free(b); ggml_free(c);
                return emb;
            };

            // GPU head matmuls: offload the per-frame logits (codec_head, CP lm_heads) from the
            // CPU dot loops to batched GPU GEMMs. Falls back to CPU if no GPU backend.
            gpu_gemm gg; bool gg_ok = gg.init();
            int GG_HEAD=-1, GG_S2M=-1; std::vector<int> GG_CPLM(n_codebooks,-1);
            // Embedding-table indices for the fused on-GPU gather (lever b).
            int GE_TALK=-1; std::vector<int> GE_CP(n_codebooks,-1);
            if (gg_ok) {
                GG_HEAD = gg.add(w_codec_head);
                for (int i=0;i<n_codebooks;i++) GG_CPLM[i] = gg.add(cp_lm_heads[i]);
                if (w_cp_s2m) GG_S2M = gg.add(w_cp_s2m);
                // codec_embd + 15 CP codec_embd tables, resident on GPU for get_rows.
                GE_TALK = gg.add_table(w_codec_embd);
                for (int i=0;i<n_codebooks;i++) GE_CP[i] = gg.add_table(cp_codec_embds[i]);
                gg.set_s2m_w(GG_S2M);
                gg.set_pad(tts_pad_embed, n_embd);
                gg_ok = gg.finalize();
            }
            std::vector<float> s2m_bias;
            if (b_cp_s2m) { s2m_bias.resize((size_t)ggml_nelements(b_cp_s2m)); dequant_to_f32(b_cp_s2m, s2m_bias.data()); }
            // GPU gather is used only when it fuses with a real s2m projection (1.7B-style); on
            // 0.6B (identity s2m) the single-row CPU gather is cheaper than a GPU round-trip. The
            // summed talker frame (16 gathers) always goes to GPU when available -- see Op C.
            bool ge_proj = gg_ok && GG_S2M>=0;
            printf("[serve] gpu head matmuls: %s  (projection %s, embed-gather %s)\n",
                   gg_ok ? "on" : "off (CPU fallback)",
                   (gg_ok && GG_S2M>=0) ? "gpu" : (w_cp_s2m ? "cpu" : "identity"),
                   gg_ok ? "gpu" : "cpu");

            // Project a batch of n_embd inputs [n_embd, na] -> [n_embd_cp, na] (small_to_mtp + bias).
            auto proj_batch = [&](const std::vector<float> & in, int na_, std::vector<float> & out) {
                if (gg_ok && GG_S2M>=0) {
                    gg.run(GG_S2M, in.data(), na_, out);
                    if (!s2m_bias.empty()) for (int k=0;k<na_;k++) for (int o=0;o<n_embd_cp;o++) out[(size_t)k*n_embd_cp+o]+=s2m_bias[o];
                } else if (!w_cp_s2m) {
                    out = in; // dims equal (e.g. 0.6B): identity
                } else {
                    out.resize((size_t)n_embd_cp*na_);
                    for (int k=0;k<na_;k++){ std::vector<float> v(in.begin()+(size_t)k*n_embd, in.begin()+(size_t)(k+1)*n_embd);
                        std::vector<float> p=project_cp(v); memcpy(&out[(size_t)k*n_embd_cp], p.data(), n_embd_cp*sizeof(float)); }
                }
            };

            // Per-request prefill. Base voice, x-vector clone (ref_audio only), or ICL clone
            // (ref_audio + ref_text). icl_ref returns the number of leading frames to trim.
            auto tok_wrap = [&](const std::string & s) {
                std::string p = "<|im_start|>assistant\n" + s + "<|im_end|>\n<|im_start|>assistant\n";
                std::vector<llama_token> t(p.size() + 32);
                int n = llama_tokenize(vocab, p.c_str(), (int)p.size(), t.data(), (int)t.size(), false, true);
                if (n < 0) { t.resize(-n); n = llama_tokenize(vocab, p.c_str(), (int)p.size(), t.data(), (int)t.size(), false, true); }
                t.resize(n); return t;
            };
            auto build_prefill = [&](const std::string & rtext, const std::string & rlang,
                                     const std::string & raudio, const std::string & rreftext,
                                     const std::vector<float> & spk_in,
                                     const std::vector<std::vector<int32_t>> & refc_in,
                                     std::vector<float> & pe, int & npf, int & icl_ref) {
                icl_ref = 0;
                std::vector<llama_token> toks = tok_wrap(rtext); int nt=(int)toks.size();
                std::vector<int32_t> ids(toks.begin(), toks.end());
                std::vector<float> tproj = text_project(ids);
                std::string ll = rlang; std::transform(ll.begin(), ll.end(), ll.begin(), ::tolower);
                auto it = LANGUAGE_IDS.find(ll);
                uint32_t lang_id = (it != LANGUAGE_IDS.end()) ? it->second : LANGUAGE_IDS.at("english");

                // Use precomputed embedding/codes if given (cacheable per voice); else extract from ref audio.
                std::vector<float> spk_emb = spk_in;
                std::vector<std::vector<int32_t>> ref_frames = refc_in;
                bool need_read = !raudio.empty() && (spk_emb.empty() || (!rreftext.empty() && ref_frames.empty()));
                if (need_read) {
                    std::vector<float> samp;
                    if (read_wav(raudio.c_str(), samp)) {
                        if (spk_emb.empty()) spk_emb = extract_spk(samp);
                        if (!rreftext.empty() && ref_frames.empty() && enc_ok)
                            enc_encode_audio(enc_m, samp.data(), (int)samp.size(), 16, ref_frames);
                    }
                }
                bool has_spk = !spk_emb.empty();
                bool icl = has_spk && !rreftext.empty() && !ref_frames.empty();

                std::vector<uint32_t> pre = (lang_id!=0)
                    ? std::vector<uint32_t>{codec_think_id,codec_think_bos,lang_id,codec_think_eos}
                    : std::vector<uint32_t>{codec_nothink,codec_think_bos,codec_think_eos};
                std::vector<std::vector<float>> pe_emb; for (uint32_t id : pre) pe_emb.push_back(codec_embed((int32_t)id));
                if (has_spk) pe_emb.push_back(spk_emb);
                pe_emb.push_back(codec_embed((int32_t)codec_pad_id));
                pe_emb.push_back(codec_embed((int32_t)codec_bos_id));
                int ncp=(int)pe_emb.size(), ncp_pre=ncp-1, n_text_content=std::max(0,nt-8);
                std::vector<float> cpad = codec_embed((int32_t)codec_pad_id);

                if (icl) {
                    std::vector<llama_token> rtk = tok_wrap(rreftext); int nrt=(int)rtk.size();
                    int n_ref_content = std::max(0, nrt-8);
                    std::vector<int32_t> rids(rtk.begin(), rtk.end());
                    std::vector<float> rproj = text_project(rids);
                    int nrf=(int)ref_frames.size(); icl_ref=nrf;
                    std::vector<std::vector<float>> rce(nrf, std::vector<float>(n_embd,0.f));
                    for (int f=0;f<nrf;f++){ auto& fr=ref_frames[f]; std::vector<float> e0=codec_embed(fr[0]);
                        for(int j=0;j<n_embd;j++) rce[f][j]+=e0[j];
                        for(int cb=1;cb<16&&cb<(int)fr.size();cb++){ std::vector<float> ce(n_embd); read_row(cp_codec_embds[cb-1],fr[cb],ce.data(),n_embd); for(int j=0;j<n_embd;j++) rce[f][j]+=ce[j]; } }
                    int nts = n_ref_content + n_text_content + 1;
                    npf = 3 + ncp_pre + (nts + nrf + 1) + 1;
                    pe.assign((size_t)npf*n_embd,0.f); int pos=0;
                    for(int i=0;i<3;i++){memcpy(&pe[pos*n_embd],&tproj[i*n_embd],n_embd*sizeof(float));pos++;}
                    for(int i=0;i<ncp-2;i++){for(int j=0;j<n_embd;j++)pe[pos*n_embd+j]=tts_pad_embed[j]+pe_emb[i][j];pos++;}
                    for(int j=0;j<n_embd;j++)pe[pos*n_embd+j]=tts_bos_embed[j]+pe_emb[ncp-2][j];pos++;
                    for(int i=0;i<n_ref_content;i++){for(int j=0;j<n_embd;j++)pe[pos*n_embd+j]=rproj[(3+i)*n_embd+j]+cpad[j];pos++;}
                    for(int i=0;i<n_text_content;i++){for(int j=0;j<n_embd;j++)pe[pos*n_embd+j]=tproj[(3+i)*n_embd+j]+cpad[j];pos++;}
                    for(int j=0;j<n_embd;j++)pe[pos*n_embd+j]=tts_eos_embed[j]+cpad[j];pos++;
                    { std::vector<float> cb=codec_embed((int32_t)codec_bos_id); for(int j=0;j<n_embd;j++)pe[pos*n_embd+j]=tts_pad_embed[j]+cb[j]; pos++; }
                    for(int f=0;f<nrf;f++){for(int j=0;j<n_embd;j++)pe[pos*n_embd+j]=tts_pad_embed[j]+rce[f][j];pos++;}
                    for(int j=0;j<n_embd;j++)pe[pos*n_embd+j]=tts_pad_embed[j]+pe_emb[ncp-1][j];pos++;
                } else {
                    int nts=n_text_content+1; npf=3+ncp_pre+nts+1;
                    pe.assign((size_t)npf*n_embd,0.f); int pos=0;
                    for(int i=0;i<3;i++){memcpy(&pe[pos*n_embd],&tproj[i*n_embd],n_embd*sizeof(float));pos++;}
                    for(int i=0;i<ncp-2;i++){for(int j=0;j<n_embd;j++)pe[pos*n_embd+j]=tts_pad_embed[j]+pe_emb[i][j];pos++;}
                    for(int j=0;j<n_embd;j++)pe[pos*n_embd+j]=tts_bos_embed[j]+pe_emb[ncp-2][j];pos++;
                    for(int i=0;i<n_text_content;i++){for(int j=0;j<n_embd;j++)pe[pos*n_embd+j]=tproj[(3+i)*n_embd+j]+cpad[j];pos++;}
                    for(int j=0;j<n_embd;j++)pe[pos*n_embd+j]=tts_eos_embed[j]+cpad[j];pos++;
                    for(int j=0;j<n_embd;j++)pe[pos*n_embd+j]=tts_pad_embed[j]+pe_emb[ncp-1][j];pos++;
                }
            };

            struct req_t {
                std::string text, language, ref_audio, ref_text;
                std::vector<float> spk_in;                       // precomputed x-vector (optional)
                std::vector<std::vector<int32_t>> refc_in;       // precomputed ref codes (optional)
                tts_sampler_params tsp, csp;
                int max_tok = 2048; uint32_t seed = 0;
                std::promise<std::vector<float>> result;   // non-streaming: full WAV
                // streaming (stream=true): incremental PCM pushed as it's vocoded
                bool stream = false;
                std::mutex smu; std::condition_variable scv;
                std::deque<std::vector<float>> chunks; bool sdone = false;
                void push_chunk(std::vector<float> c) {
                    { std::lock_guard<std::mutex> lk(smu); chunks.push_back(std::move(c)); }
                    scv.notify_one();
                }
                void finish_stream() {
                    { std::lock_guard<std::mutex> lk(smu); sdone = true; }
                    scv.notify_one();
                }
            };
            std::deque<std::shared_ptr<req_t>> queue;
            std::mutex qmu; std::condition_variable qcv;
            std::atomic<bool> running{true};

            // Off-thread vocode pool. Non-streaming finalize hands the finished codes to a worker
            // and frees the slot immediately, so the scheduler keeps generating for other slots
            // instead of blocking ~1.4s/clip on the CPU vocoder (measured ~83% of scheduler time).
            // Each worker decodes on its own thread_local vocoder backend -> K clips vocode in
            // parallel across cores. Tune with QWEN_VOC_WORKERS / QWEN_VOC_THREADS.
            struct voc_job { std::shared_ptr<req_t> req; std::vector<std::vector<int32_t>> codes; int icl_ref; };
            std::deque<std::shared_ptr<voc_job>> vq; std::mutex vqm; std::condition_variable vqcv;
            const int SPF_VOC = 1920;
            // Vocoder backend. GPU vocode (transformer + im2col convs on GPU, conv_transpose on
            // CPU via the sched) is ~330ms/clip vs ~1.4s on CPU, and it runs INLINE on the
            // scheduler thread so only one host thread ever touches CUDA (no device wedge -- ggml's
            // CUDA backend is not multi-thread-safe). Default ON when a GPU is present; set
            // QWEN_VOC_GPU=0 to force the CPU vocode worker pool instead.
            std::unique_ptr<gpu_vocoder> gpu_voc;
            bool want_gpu_voc = voc_ready && !(getenv("QWEN_VOC_GPU") && atoi(getenv("QWEN_VOC_GPU"))==0);
            if (want_gpu_voc) {
                gpu_voc = std::make_unique<gpu_vocoder>();
                if (gpu_voc->init(voc_loader)) printf("[serve] vocoder backend: GPU inline (conv_transpose on CPU)\n");
                else { gpu_voc.reset(); printf("[serve] vocoder backend: no GPU -> CPU pool\n"); }
            }
            const bool voc_gpu_inline = (bool)gpu_voc;        // GPU vocode always runs inline (safe)
            int n_voc_workers = voc_gpu_inline ? 0 : 6;        // CPU pool only used when no GPU vocode
            if (!voc_gpu_inline) {
                if (const char* e=getenv("QWEN_VOC_WORKERS")) n_voc_workers = std::max(1, atoi(e));
                if (!getenv("QWEN_VOC_THREADS")) {
                    unsigned hw = std::thread::hardware_concurrency(); if (!hw) hw = 8;
                    char b[16]; snprintf(b, sizeof(b), "%d", std::max(1u, hw / (unsigned)n_voc_workers));
                    setenv("QWEN_VOC_THREADS", b, 1);
                }
                printf("[serve] vocode pool: %d workers x %s threads\n", n_voc_workers, getenv("QWEN_VOC_THREADS"));
            }
            std::vector<std::thread> vworkers;
            for (int w=0; w<n_voc_workers; w++) vworkers.emplace_back([&](){
                while (running) {
                    std::shared_ptr<voc_job> j;
                    { std::unique_lock<std::mutex> lk(vqm);
                      vqcv.wait(lk, [&]{ return !vq.empty() || !running; });
                      if (!vq.empty()) { j = vq.front(); vq.pop_front(); } }
                    if (!j) continue;
                    std::vector<float> pcm;
                    if (voc_ready && !j->codes.empty()) {
                        if (gpu_voc) pcm = gpu_voc->decode(j->codes, 0, (int)j->codes.size());
                        else         pcm = voc_decode_audio(voc, j->codes, (int)j->codes.size());
                    }
                    int trim = j->icl_ref * SPF_VOC; // trim the regenerated ICL reference prefix
                    if (trim > 0 && trim < (int)pcm.size()) {
                        int cut = find_zero_crossing(pcm, trim);
                        pcm.erase(pcm.begin(), pcm.begin() + cut);
                        fade_in(pcm, 240);
                    }
                    j->req->result.set_value(std::move(pcm));
                }
            });

            std::thread sched([&]() {
                struct sslot {
                    bool active=false; std::shared_ptr<req_t> req;
                    std::vector<float> hidden; int cb0=0,tpos=0,gstep=0;
                    std::vector<int32_t> recent;
                    std::vector<std::vector<int32_t>> codes;
                    std::mt19937 rng;
                    int emitted_samples=0, last_voc_n=0; // streaming vocode state
                    int icl_ref_frames=0;                // leading frames to trim (ICL clone)
                };
                std::vector<sslot> S(N);
                // --- per-stage profiling (env QWEN_PROFILE), single scheduler thread so plain doubles are safe ---
                const bool PROF = getenv("QWEN_PROFILE")!=nullptr;
                struct { double dec=0, aux=0, read=0, samp=0, voc=0, frame=0, finvoc=0, prefill=0; long frames=0, iters=0; } P;
                auto CLK=[](){ return std::chrono::steady_clock::now(); };
                auto EL =[&](std::chrono::steady_clock::time_point t){ return std::chrono::duration<double,std::milli>(CLK()-t).count(); };
                const int SPF=1920, SV_CHUNK=12, SV_LEFT=16, SV_MARGIN=6;
                // Emit any newly-safe streaming audio for slot s (windowed vocode, emit-once).
                auto stream_emit = [&](int s, bool flush){
                    if (!voc_ready) return;
                    int n_done = (int)S[s].codes.size();
                    if (!flush && n_done - S[s].last_voc_n < SV_CHUNK) return;
                    int safe = flush ? n_done : (n_done - SV_MARGIN);
                    if (safe * SPF <= S[s].emitted_samples) { S[s].last_voc_n = n_done; return; }
                    int ef = S[s].emitted_samples / SPF;
                    int win0 = std::max(0, ef - SV_LEFT);
                    int win_off = win0 * SPF;
                    std::vector<float> a = voc_decode_audio_range(voc, S[s].codes, win0, n_done);
                    int target = std::min(win_off + (int)a.size(), safe * SPF);
                    if (target > S[s].emitted_samples) {
                        std::vector<float> chunk(a.begin() + (S[s].emitted_samples - win_off), a.begin() + (target - win_off));
                        S[s].req->push_chunk(std::move(chunk));
                        S[s].emitted_samples = target;
                    }
                    S[s].last_voc_n = n_done;
                };
                llama_batch pf = llama_batch_init(2048, n_embd, N);
                free(pf.pos); pf.pos = (llama_pos*)malloc(sizeof(llama_pos)*2048*n_pos_per_embd);
                llama_batch tb = llama_batch_init(N, n_embd, N);
                free(tb.pos); tb.pos = (llama_pos*)malloc(sizeof(llama_pos)*N*n_pos_per_embd);
                llama_batch cpb = llama_batch_init(2*N, n_embd_cp, N);
                std::vector<std::vector<int32_t>> cprec(N);

                auto release = [&](int s){ llama_memory_seq_rm(tmem,s,-1,-1); S[s].active=false; S[s].req.reset(); };
                // Normal completion: stream -> flush tail + done; non-stream -> full WAV via promise.
                auto finalize = [&](int s){
                    if (S[s].req->stream) { stream_emit(s, true); S[s].req->finish_stream(); }
                    else if (voc_gpu_inline) {
                        // Inline GPU vocode on the scheduler thread (only thread touching CUDA).
                        std::vector<float> pcm;
                        auto _fv=CLK();
                        if (voc_ready && !S[s].codes.empty()) pcm = gpu_voc->decode(S[s].codes, 0, (int)S[s].codes.size());
                        P.finvoc+=EL(_fv);
                        int trim = S[s].icl_ref_frames * SPF;
                        if (trim > 0 && trim < (int)pcm.size()) { int cut=find_zero_crossing(pcm,trim); pcm.erase(pcm.begin(),pcm.begin()+cut); fade_in(pcm,240); }
                        S[s].req->result.set_value(std::move(pcm));
                    }
                    else {
                        // Hand the finished codes to the vocode pool and free the slot now; a worker
                        // vocodes + trims off-thread and fulfills the request's promise.
                        auto j = std::make_shared<voc_job>();
                        j->req = S[s].req; j->codes = std::move(S[s].codes); j->icl_ref = S[s].icl_ref_frames;
                        { std::lock_guard<std::mutex> lk(vqm); vq.push_back(std::move(j)); } vqcv.notify_one();
                    }
                    release(s);
                };
                // Error path: deliver empty result and free the slot.
                auto fail = [&](int s){ if (S[s].req->stream) S[s].req->finish_stream(); else S[s].req->result.set_value({}); release(s); };

                while (running) {
                    // Admit waiting requests into free slots (per-request prefill).
                    for (int s = 0; s < N; s++) {
                        if (S[s].active) continue;
                        std::shared_ptr<req_t> r;
                        { std::lock_guard<std::mutex> lk(qmu); if (!queue.empty()) { r = queue.front(); queue.pop_front(); } }
                        if (!r) break;
                        auto _pf=CLK();
                        std::vector<float> pe; int npf=0, icl_ref=0;
                        build_prefill(r->text, r->language, r->ref_audio, r->ref_text, r->spk_in, r->refc_in, pe, npf, icl_ref);
                        llama_memory_seq_rm(tmem, s, -1, -1);
                        pf.n_tokens = npf;
                        memset(pf.pos, 0, sizeof(llama_pos)*2048*n_pos_per_embd);
                        for (int i=0;i<npf;i++){
                            memcpy(pf.embd+i*n_embd, &pe[i*n_embd], n_embd*sizeof(float));
                            for(int d=0;d<3;d++) pf.pos[d*npf+i]=i;
                            pf.pos[3*npf+i]=0;
                            pf.n_seq_id[i]=1; pf.seq_id[i][0]=s; pf.logits[i]=(i==npf-1)?1:0;
                        }
                        if (llama_decode(talker_ctx, pf)!=0){ r->result.set_value({}); continue; }
                        const float* h = llama_get_embeddings_ith(talker_ctx, npf-1);
                        S[s].hidden.assign(h, h+n_embd);
                        std::vector<float> cl; compute_logits(h, cl);
                        S[s].rng = std::mt19937(r->seed); S[s].recent.clear();
                        S[s].cb0 = tts_sample(cl.data(), n_codec_vocab, r->tsp, S[s].recent, S[s].rng);
                        S[s].recent.push_back(S[s].cb0);
                        S[s].tpos=npf; S[s].gstep=0; S[s].codes.clear(); S[s].req=r; S[s].active=true;
                        S[s].icl_ref_frames=icl_ref;
                        // Skip the regenerated reference prefix from streamed output (kept as vocoder context).
                        S[s].emitted_samples = icl_ref * SPF; S[s].last_voc_n=0;
                        P.prefill+=EL(_pf);
                    }

                    std::vector<int> act; for (int s=0;s<N;s++) if (S[s].active) act.push_back(s);
                    if (act.empty()) {
                        std::unique_lock<std::mutex> lk(qmu);
                        qcv.wait_for(lk, std::chrono::milliseconds(50), [&]{ return !queue.empty() || !running; });
                        continue;
                    }

                    // Finalize finished slots; collect the rest to step.
                    std::vector<int> st;
                    for (int s : act) {
                        if (S[s].cb0 == (int)codec_eos_id || S[s].gstep >= S[s].req->max_tok) finalize(s);
                        else st.push_back(s);
                    }
                    if (st.empty()) continue;
                    int na = (int)st.size();
                    auto _fr = CLK();

                    // Batched code predictor (frame-synchronized across slots).
                    std::vector<std::vector<int32_t>> fc(N, std::vector<int32_t>(16,0));
                    for (int s : st){ fc[s][0]=S[s].cb0; cprec[s].clear(); llama_memory_seq_rm(cmem,s,-1,-1); }
                    cpb.n_tokens = 2*na;
                    { std::vector<float> Hh((size_t)n_embd*na), thP, e0P;
                      for (int k=0;k<na;k++) memcpy(&Hh[(size_t)k*n_embd], S[st[k]].hidden.data(), n_embd*sizeof(float));
                      auto _a=CLK();
                      proj_batch(Hh, na, thP);
                      if (ge_proj) { std::vector<int32_t> ids(na); for(int k=0;k<na;k++) ids[k]=S[st[k]].cb0;
                          gg.gather_proj(GE_TALK, ids.data(), na, e0P);
                          if (!s2m_bias.empty()) for(int k=0;k<na;k++) for(int o=0;o<n_embd_cp;o++) e0P[(size_t)k*n_embd_cp+o]+=s2m_bias[o]; }
                      else { std::vector<float> E0((size_t)n_embd*na);
                          for (int k=0;k<na;k++){ std::vector<float> e=codec_embed(S[st[k]].cb0); memcpy(&E0[(size_t)k*n_embd], e.data(), n_embd*sizeof(float)); }
                          proj_batch(E0, na, e0P); }
                      P.aux+=EL(_a);
                      for (int k=0;k<na;k++){ int s=st[k];
                          memcpy(cpb.embd+(2*k)*n_embd_cp,   &thP[(size_t)k*n_embd_cp], n_embd_cp*sizeof(float));
                          memcpy(cpb.embd+(2*k+1)*n_embd_cp, &e0P[(size_t)k*n_embd_cp], n_embd_cp*sizeof(float));
                          cpb.pos[2*k]=0; cpb.pos[2*k+1]=1; cpb.n_seq_id[2*k]=1; cpb.n_seq_id[2*k+1]=1;
                          cpb.seq_id[2*k][0]=s; cpb.seq_id[2*k+1][0]=s; cpb.logits[2*k]=0; cpb.logits[2*k+1]=1; } }
                    { auto _d=CLK(); int rc=llama_decode(cp_ctx,cpb); P.dec+=EL(_d); if (rc!=0){ for(int s:st) fail(s); continue; } }
                    // CP step 0 logits: batched GPU GEMM (cp lm_head[0]) across slots, else CPU.
                    { std::vector<float> H((size_t)n_embd_cp*na), L;
                      auto _r=CLK(); for (int k=0;k<na;k++) memcpy(&H[(size_t)k*n_embd_cp], llama_get_embeddings_ith(cp_ctx,2*k+1), n_embd_cp*sizeof(float)); P.read+=EL(_r);
                      auto _a=CLK(); if (gg_ok) gg.run(GG_CPLM[0], H.data(), na, L); P.aux+=EL(_a);
                      auto _s=CLK();
                      for (int k=0;k<na;k++){ int s=st[k]; std::vector<float> buf; float* lg;
                          if (gg_ok) lg=&L[(size_t)k*cp_vocab]; else { cp_compute_logits(0,&H[(size_t)k*n_embd_cp],buf); lg=buf.data(); }
                          fc[s][1]=tts_sample(lg,cp_vocab,S[s].req->csp,cprec[s],S[s].rng); cprec[s].push_back(fc[s][1]); }
                      P.samp+=EL(_s); }
                    for (int cb_step=1; cb_step<n_codebooks; cb_step++){
                        cpb.n_tokens=na;
                        { std::vector<float> seP; auto _a=CLK();
                          if (ge_proj) { std::vector<int32_t> ids(na); for(int k=0;k<na;k++) ids[k]=fc[st[k]][cb_step];
                              gg.gather_proj(GE_CP[cb_step-1], ids.data(), na, seP);
                              if (!s2m_bias.empty()) for(int k=0;k<na;k++) for(int o=0;o<n_embd_cp;o++) seP[(size_t)k*n_embd_cp+o]+=s2m_bias[o]; }
                          else { std::vector<float> Se((size_t)n_embd*na);
                              for(int k=0;k<na;k++){ std::vector<float> e=cp_codec_embed(cb_step-1,fc[st[k]][cb_step]); memcpy(&Se[(size_t)k*n_embd], e.data(), n_embd*sizeof(float)); }
                              proj_batch(Se, na, seP); }
                          P.aux+=EL(_a);
                          for(int k=0;k<na;k++){int s=st[k]; memcpy(cpb.embd+k*n_embd_cp,&seP[(size_t)k*n_embd_cp],n_embd_cp*sizeof(float)); cpb.pos[k]=2+(cb_step-1); cpb.n_seq_id[k]=1; cpb.seq_id[k][0]=s; cpb.logits[k]=1;} }
                        { auto _d=CLK(); int rc=llama_decode(cp_ctx,cpb); P.dec+=EL(_d); if (rc!=0) break; }
                        std::vector<float> H((size_t)n_embd_cp*na), L;
                        auto _r=CLK(); for (int k=0;k<na;k++) memcpy(&H[(size_t)k*n_embd_cp], llama_get_embeddings_ith(cp_ctx,k), n_embd_cp*sizeof(float)); P.read+=EL(_r);
                        auto _a2=CLK(); if (gg_ok) gg.run(GG_CPLM[cb_step], H.data(), na, L); P.aux+=EL(_a2);
                        auto _s=CLK();
                        for(int k=0;k<na;k++){int s=st[k]; std::vector<float> buf; float* lg;
                            if (gg_ok) lg=&L[(size_t)k*cp_vocab]; else { cp_compute_logits(cb_step,&H[(size_t)k*n_embd_cp],buf); lg=buf.data(); }
                            fc[s][cb_step+1]=tts_sample(lg,cp_vocab,S[s].req->csp,cprec[s],S[s].rng); cprec[s].push_back(fc[s][cb_step+1]); }
                        P.samp+=EL(_s);
                    }
                    for (int s:st) S[s].codes.push_back(fc[s]);
                    { auto _v=CLK(); for (int s:st) if (S[s].req->stream) stream_emit(s, false); P.voc+=EL(_v); } // push incremental PCM

                    // Batched talker step (one summed-frame embedding per slot). The 16 codec
                    // embedding gathers + sum + tts_pad run as one fused GPU op when available.
                    tb.n_tokens=na; memset(tb.pos,0,sizeof(llama_pos)*N*n_pos_per_embd);
                    std::vector<float> NE;
                    if (gg_ok) {
                        std::vector<int32_t> ids((size_t)16*na);
                        for (int k=0;k<na;k++){ int s=st[k]; ids[k]=fc[s][0];
                            for(int cbi=0;cbi<n_codebooks;cbi++) ids[(size_t)(cbi+1)*na+k]=fc[s][cbi+1]; }
                        int tidxs[16]; tidxs[0]=GE_TALK; for(int i=0;i<n_codebooks;i++) tidxs[i+1]=GE_CP[i];
                        auto _a=CLK(); gg.gather_sum_pad(tidxs, 16, ids.data(), na, NE); P.aux+=EL(_a); // includes + tts_pad
                    }
                    for (int k=0;k<na;k++){ int s=st[k];
                        if (gg_ok) memcpy(tb.embd+k*n_embd, &NE[(size_t)k*n_embd], n_embd*sizeof(float));
                        else { std::vector<float> ne(n_embd,0.f); std::vector<float> e0=codec_embed(fc[s][0]);
                            for(int j=0;j<n_embd;j++)ne[j]+=e0[j];
                            for(int cbi=0;cbi<n_codebooks;cbi++){std::vector<float> ec=cp_codec_embed(cbi,fc[s][cbi+1]); for(int j=0;j<n_embd;j++)ne[j]+=ec[j];}
                            for(int j=0;j<n_embd;j++)ne[j]+=tts_pad_embed[j];
                            memcpy(tb.embd+k*n_embd,ne.data(),n_embd*sizeof(float)); }
                        for(int d=0;d<3;d++)tb.pos[d*na+k]=S[s].tpos; tb.pos[3*na+k]=0;
                        tb.n_seq_id[k]=1; tb.seq_id[k][0]=s; tb.logits[k]=1; }
                    { auto _d=CLK(); int rc=llama_decode(talker_ctx,tb); P.dec+=EL(_d); if (rc!=0){ for(int s:st) fail(s); continue; } }
                    // Talker cb0 logits: batched GPU GEMM (codec_head) across slots, else CPU.
                    { std::vector<float> Hh((size_t)n_embd*na), L;
                      auto _r=CLK(); for (int k=0;k<na;k++){ const float* h=llama_get_embeddings_ith(talker_ctx,k); S[st[k]].hidden.assign(h,h+n_embd); memcpy(&Hh[(size_t)k*n_embd],h,n_embd*sizeof(float)); } P.read+=EL(_r);
                      auto _a=CLK(); if (gg_ok) gg.run(GG_HEAD, Hh.data(), na, L); P.aux+=EL(_a);
                      auto _s=CLK();
                      for(int k=0;k<na;k++){ int s=st[k]; std::vector<float> cl; float* lg;
                          if (gg_ok) lg=&L[(size_t)k*n_codec_vocab]; else { compute_logits(&Hh[(size_t)k*n_embd],cl); lg=cl.data(); }
                          S[s].cb0=tts_sample(lg,n_codec_vocab,S[s].req->tsp,S[s].recent,S[s].rng);
                          S[s].recent.push_back(S[s].cb0); S[s].tpos++; S[s].gstep++; }
                      P.samp+=EL(_s); }
                    P.frame+=EL(_fr); P.iters++; P.frames+=na;
                    if (PROF && P.iters%200==0) {
                        double t=P.frame; printf("[prof] iters=%ld frames=%ld | LOOP %.1fms/iter: dec=%.0f%% aux=%.0f%% read=%.0f%% samp=%.0f%% voc=%.0f%% other=%.0f%% | OUTSIDE-LOOP totals: prefill=%.0fms finalize_vocode=%.0fms loop=%.0fms\n",
                            P.iters,P.frames, t/P.iters, 100*P.dec/t,100*P.aux/t,100*P.read/t,100*P.samp/t,100*P.voc/t,
                            100*(t-P.dec-P.aux-P.read-P.samp-P.voc)/t, P.prefill, P.finvoc, P.frame);
                        fflush(stdout);
                    }
                }
                llama_batch_free(pf); llama_batch_free(tb); llama_batch_free(cpb);
            });

            auto jstr=[](const nlohmann::json&j,const char*k,const std::string&d){ return j.contains(k)&&j[k].is_string()?j[k].get<std::string>():d; };
            httplib::Server svr;
            svr.Get("/health",[](const httplib::Request&,httplib::Response&res){ res.set_content("{\"status\":\"healthy\"}","application/json"); });
            svr.Get("/v1/models",[](const httplib::Request&,httplib::Response&res){ res.set_content("{\"object\":\"list\",\"data\":[{\"id\":\"qwen3-tts\",\"object\":\"model\",\"owned_by\":\"qwen3-tts-llama\"}]}","application/json"); });
            svr.Post("/v1/audio/speech",[&](const httplib::Request&req,httplib::Response&res){
                nlohmann::json j;
                try { j = nlohmann::json::parse(req.body); } catch(...) { res.status=400; res.set_content("{\"error\":\"bad json\"}","application/json"); return; }
                auto r = std::make_shared<req_t>();
                r->text = jstr(j,"input",""); r->language = jstr(j,"language","english");
                r->ref_audio = jstr(j,"ref_audio",""); r->ref_text = jstr(j,"ref_text","");
                if (j.contains("spk_embedding") && j["spk_embedding"].is_array())
                    for (auto & v : j["spk_embedding"]) r->spk_in.push_back(v.get<float>());
                if (j.contains("ref_codes") && j["ref_codes"].is_array())
                    for (auto & frame : j["ref_codes"]) { std::vector<int32_t> f; if (frame.is_array()) for (auto & c : frame) f.push_back(c.get<int>()); if (!f.empty()) r->refc_in.push_back(std::move(f)); }
                r->tsp = talker_sparams; r->csp = cp_sparams;
                if (j.contains("temperature")&&j["temperature"].is_number()) r->tsp.temp=j["temperature"].get<float>();
                if (j.contains("top_k")&&j["top_k"].is_number()) r->tsp.top_k=j["top_k"].get<int>();
                if (j.contains("top_p")&&j["top_p"].is_number()) r->tsp.top_p=j["top_p"].get<float>();
                r->max_tok = (j.contains("max_new_tokens")&&j["max_new_tokens"].is_number())?j["max_new_tokens"].get<int>():max_tokens;
                r->seed = (j.contains("seed")&&j["seed"].is_number())?(uint32_t)j["seed"].get<long long>():(uint32_t)std::random_device{}();
                r->stream = j.contains("stream") && j["stream"].is_boolean() && j["stream"].get<bool>();
                if (r->text.empty()){ res.status=400; res.set_content("{\"error\":\"input required\"}","application/json"); return; }
                std::future<std::vector<float>> fut;
                if (!r->stream) fut = r->result.get_future(); // before enqueue, to avoid racing set_value
                { std::lock_guard<std::mutex> lk(qmu); queue.push_back(r); } qcv.notify_one();
                if (r->stream) {
                    // Chunked raw 16-bit PCM as it is vocoded (sglang-style streaming).
                    res.set_header("X-Audio-Sample-Rate","24000");
                    res.set_header("X-Audio-Channels","1");
                    res.set_header("X-Audio-Codec","pcm_s16le");
                    res.set_chunked_content_provider("audio/pcm",
                        [r](size_t, httplib::DataSink & sink) -> bool {
                            std::unique_lock<std::mutex> lk(r->smu);
                            r->scv.wait(lk, [&]{ return !r->chunks.empty() || r->sdone; });
                            while (!r->chunks.empty()) {
                                std::vector<float> c = std::move(r->chunks.front()); r->chunks.pop_front();
                                lk.unlock();
                                std::string b; b.reserve(c.size()*2);
                                for (float v : c) { float x=fmaxf(-1.f,fminf(1.f,v)); int16_t q=(int16_t)(x*32767.f); b.push_back((char)(q&0xff)); b.push_back((char)((q>>8)&0xff)); }
                                if (!sink.write(b.data(), b.size())) return false;
                                lk.lock();
                            }
                            if (r->sdone) { sink.done(); return false; }
                            return true;
                        });
                } else {
                    std::vector<float> pcm = fut.get();
                    std::string wav = wav_bytes(pcm.data(), (int)pcm.size(), 24000);
                    res.set_header("X-Audio-Sample-Rate","24000");
                    res.set_content(wav, "audio/wav");
                }
            });
            // Precompute a voice's x-vector (+ ICL ref codes) for caching in a DB. Runs on the
            // request thread (CPU encoders, independent of the scheduler's llama contexts).
            svr.Post("/v1/audio/encode",[&](const httplib::Request&req,httplib::Response&res){
                nlohmann::json j;
                try { j = nlohmann::json::parse(req.body); } catch(...) { res.status=400; res.set_content("{\"error\":\"bad json\"}","application/json"); return; }
                std::string raudio = jstr(j,"ref_audio","");
                if (raudio.empty()){ res.status=400; res.set_content("{\"error\":\"ref_audio required\"}","application/json"); return; }
                bool with_codes = !(j.contains("with_codes") && j["with_codes"].is_boolean() && !j["with_codes"].get<bool>()); // default true
                std::vector<float> samp;
                if (!read_wav(raudio.c_str(), samp)){ res.status=400; res.set_content("{\"error\":\"cannot read ref_audio\"}","application/json"); return; }
                std::vector<float> spk = extract_spk(samp);
                std::vector<std::vector<int32_t>> frames;
                if (with_codes && enc_ok) enc_encode_audio(enc_m, samp.data(), (int)samp.size(), 16, frames);
                nlohmann::json out; out["spk_embedding"] = spk; out["embedding_dim"] = (int)spk.size();
                if (!frames.empty()) { out["ref_codes"] = frames; out["ref_frames"] = (int)frames.size(); }
                res.set_content(out.dump(), "application/json");
            });
            printf("[serve] HTTP server on 0.0.0.0:%d (slots=%d): POST /v1/audio/speech, /v1/audio/encode | GET /health /v1/models\n", serve_port, N);
            svr.listen("0.0.0.0", serve_port);
            running=false; qcv.notify_all(); vqcv.notify_all(); sched.join();
            for (auto & w : vworkers) w.join();
            llama_batch_free(batch);
            goto cleanup;
        }

        // ════════════════════════════════════════════════════════════════════
        //  Phase 1: warm continuous-batching engine (test path).
        //  Runs `serve_slots` copies of this request concurrently through ONE warm
        //  model, batching the talker decode (multi-seq llama_batch) and the
        //  code-predictor (frame-synchronized) across slots. Proves true batching
        //  before the HTTP/WS API is layered on top.
        // ════════════════════════════════════════════════════════════════════
        if (serve_slots >= 1 && !serve_http) { // >=1 so N=1 isolates batched-loop bugs from multi-seq bugs
            const int N = serve_slots;
            llama_memory_t talker_mem = llama_get_memory(talker_ctx);
            llama_memory_t cp_mem     = llama_get_memory(cp_ctx);

            // Replicate the prefilled seq 0 KV into the other slots (identical request).
            for (int s = 1; s < N; s++) llama_memory_seq_cp(talker_mem, 0, s, -1, -1);

            struct slot_state {
                std::vector<float> hidden;                 // talker hidden for CP input
                int cb0 = 0, tpos = 0, gstep = 0;
                bool active = true;
                std::vector<int32_t> recent;               // talker rep-penalty history
                std::vector<std::vector<int32_t>> codes;   // [frames][16]
                std::mt19937 rng;
            };
            std::vector<slot_state> S(N);
            for (int s = 0; s < N; s++) {
                S[s].hidden.assign(hidden, hidden + n_embd);
                S[s].cb0   = cb0_token;
                S[s].tpos  = n_prefill;
                S[s].recent = { cb0_token };
                S[s].rng   = std::mt19937((uint32_t)((seed >= 0 ? seed : 1234) + s));
            }

            llama_batch tb = llama_batch_init(N, n_embd, N);          // talker step batch
            free(tb.pos); tb.pos = (llama_pos *)malloc(sizeof(llama_pos) * N * n_pos_per_embd);
            llama_batch cpb = llama_batch_init(2 * N, n_embd_cp, N);  // CP batch (<=2N tokens)
            std::vector<std::vector<int32_t>> cprec(N);               // CP rep-penalty per slot/frame

            printf("\n[serve] warm batched engine: %d slots\n", N);
            auto t0 = std::chrono::high_resolution_clock::now();
            int n_active = N;

            while (n_active > 0) {
                for (auto & sl : S) {
                    if (sl.active && (sl.cb0 == (int)codec_eos_id || sl.gstep >= max_tokens)) sl.active = false;
                }
                std::vector<int> act;
                for (int s = 0; s < N; s++) if (S[s].active) act.push_back(s);
                n_active = (int)act.size();
                if (n_active == 0) break;

                std::vector<std::vector<int32_t>> fc(N, std::vector<int32_t>(16, 0));
                for (int s : act) { fc[s][0] = S[s].cb0; cprec[s].clear(); llama_memory_seq_rm(cp_mem, s, -1, -1); }

                // CP prefill: [proj(hidden), proj(codec_embed(cb0))] per slot
                cpb.n_tokens = 2 * n_active;
                for (int k = 0; k < n_active; k++) {
                    int s = act[k];
                    std::vector<float> th = project_cp(S[s].hidden);
                    std::vector<float> e0 = project_cp(codec_embed(S[s].cb0));
                    memcpy(cpb.embd + (2*k+0)*n_embd_cp, th.data(), n_embd_cp * sizeof(float));
                    memcpy(cpb.embd + (2*k+1)*n_embd_cp, e0.data(), n_embd_cp * sizeof(float));
                    cpb.pos[2*k+0] = 0; cpb.pos[2*k+1] = 1;
                    cpb.n_seq_id[2*k+0] = 1; cpb.n_seq_id[2*k+1] = 1;
                    cpb.seq_id[2*k+0][0] = s; cpb.seq_id[2*k+1][0] = s;
                    cpb.logits[2*k+0] = 0; cpb.logits[2*k+1] = 1;
                }
                if (llama_decode(cp_ctx, cpb) != 0) { fprintf(stderr, "[serve] CP prefill failed\n"); break; }
                for (int k = 0; k < n_active; k++) {
                    int s = act[k];
                    // CP prefill output is the 2nd token of slot k -> batch index 2k+1 (logits=1 there).
                    std::vector<float> buf; cp_compute_logits(0, llama_get_embeddings_ith(cp_ctx, 2*k + 1), buf);
                    fc[s][1] = tts_sample(buf.data(), cp_vocab, cp_sparams, cprec[s], S[s].rng);
                    cprec[s].push_back(fc[s][1]);
                }
                for (int cb_step = 1; cb_step < n_codebooks; cb_step++) {
                    cpb.n_tokens = n_active;
                    for (int k = 0; k < n_active; k++) {
                        int s = act[k];
                        std::vector<float> se = project_cp(cp_codec_embed(cb_step - 1, fc[s][cb_step]));
                        memcpy(cpb.embd + k*n_embd_cp, se.data(), n_embd_cp * sizeof(float));
                        cpb.pos[k] = 2 + (cb_step - 1);
                        cpb.n_seq_id[k] = 1; cpb.seq_id[k][0] = s; cpb.logits[k] = 1;
                    }
                    if (llama_decode(cp_ctx, cpb) != 0) { fprintf(stderr, "[serve] CP step failed\n"); break; }
                    for (int k = 0; k < n_active; k++) {
                        int s = act[k];
                        std::vector<float> buf; cp_compute_logits(cb_step, llama_get_embeddings_ith(cp_ctx, k), buf);
                        fc[s][cb_step + 1] = tts_sample(buf.data(), cp_vocab, cp_sparams, cprec[s], S[s].rng);
                        cprec[s].push_back(fc[s][cb_step + 1]);
                    }
                }
                for (int s : act) S[s].codes.push_back(fc[s]);

                // Batched talker: one summed-frame embedding per active slot
                tb.n_tokens = n_active;
                memset(tb.pos, 0, sizeof(llama_pos) * N * n_pos_per_embd);
                for (int k = 0; k < n_active; k++) {
                    int s = act[k];
                    std::vector<float> ne(n_embd, 0.0f);
                    std::vector<float> e0 = codec_embed(fc[s][0]);
                    for (int j = 0; j < n_embd; j++) ne[j] += e0[j];
                    for (int cbi = 0; cbi < n_codebooks; cbi++) {
                        std::vector<float> ec = cp_codec_embed(cbi, fc[s][cbi + 1]);
                        for (int j = 0; j < n_embd; j++) ne[j] += ec[j];
                    }
                    const float * tx = (S[s].gstep < n_trailing_text)
                        ? &trailing_text[S[s].gstep * n_embd] : pad_embed_vec.data();
                    for (int j = 0; j < n_embd; j++) ne[j] += tx[j];
                    memcpy(tb.embd + k*n_embd, ne.data(), n_embd * sizeof(float));
                    for (int d = 0; d < 3; d++) tb.pos[d*n_active + k] = S[s].tpos;
                    tb.pos[3*n_active + k] = 0;
                    tb.n_seq_id[k] = 1; tb.seq_id[k][0] = s; tb.logits[k] = 1;
                }
                if (llama_decode(talker_ctx, tb) != 0) { fprintf(stderr, "[serve] talker decode failed\n"); break; }
                for (int k = 0; k < n_active; k++) {
                    int s = act[k];
                    const float * h = llama_get_embeddings_ith(talker_ctx, k);
                    S[s].hidden.assign(h, h + n_embd);
                    std::vector<float> cl; compute_logits(h, cl);
                    S[s].cb0 = tts_sample(cl.data(), n_codec_vocab, talker_sparams, S[s].recent, S[s].rng);
                    S[s].recent.push_back(S[s].cb0);
                    S[s].tpos++; S[s].gstep++;
                }
            }

            double wall = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count() / 1000.0;
            double tot_audio = 0;
            std::string base = output_path.size() > 4 ? output_path.substr(0, output_path.size() - 4) : output_path;
            for (int s = 0; s < N; s++) {
                int nf = (int)S[s].codes.size();
                tot_audio += nf / 12.5;
                if (voc_ready && nf > 0) {
                    std::vector<float> a = voc_decode_audio(voc, S[s].codes, nf);
                    std::string outp = base + "_slot" + std::to_string(s) + ".wav";
                    write_wav(outp.c_str(), a.data(), (int)a.size(), 24000);
                }
            }
            printf("[serve] %d slots: decode wall %.2fs, total audio %.2fs, aggregate RTF %.2fx\n",
                   N, wall, tot_audio, wall > 0 ? tot_audio / wall : 0.0);
            llama_batch_free(tb); llama_batch_free(cpb); llama_batch_free(batch);
            goto cleanup;
        }

        // ── Autoregressive decode loop ──────────────────────────────────
        std::vector<std::vector<int32_t>> all_codes; // [n_frames, 16]

        int gen_step = 0;
        int talker_pos = n_prefill;

        // Pre-allocate reusable Talker step batch (avoids alloc/free per frame)
        llama_batch talker_step_batch = llama_batch_init(1, n_embd, 1);
        free(talker_step_batch.pos);
        talker_step_batch.pos = (llama_pos *)malloc(sizeof(llama_pos) * n_pos_per_embd);

        printf("\n[Starting autoregressive decode loop]\n");
        auto t_decode_start = std::chrono::high_resolution_clock::now();
        double talker_decode_ms = 0, cp_decode_ms = 0, head_ms = 0;

        // ── Streaming state (used when stream_chunk > 0) ──────────────────────
        const int SAMPLES_PER_FRAME = 1920; // 12.5 Hz codec -> 24 kHz
        const int stream_margin = 6;        // frames held back so committed samples have right-context
        std::vector<float> stream_audio;    // all emitted samples (for the final WAV)
        // Skip the ICL reference frames from the output; they remain in all_codes as left-context.
        int   emitted_samples = icl_ref_frames * SAMPLES_PER_FRAME;
        int   last_decode_n   = 0;
        // Adaptive chunk size: a small first chunk minimizes time-to-first-audio, then it ramps up to
        // stream_chunk so most of the stream is decoded in larger, more efficient chunks (less of the
        // left-context window is re-decoded per emitted frame). Inspired by vllm-omni's initial chunk.
        int   stream_cur = (stream_chunk_initial > 0 && stream_chunk_initial < stream_chunk)
                               ? stream_chunk_initial : stream_chunk;
        bool  ttfb_recorded   = false;
        double ttfb_ms        = 0.0;
        const bool streaming  = stream_chunk > 0 && voc_ready;

        for (int frame = 0; frame < max_tokens; frame++) {
            if (cb0_token == (int)codec_eos_id) {
                printf("  EOS detected at frame %d\n", frame);
                break;
            }

            // ── Step 1: Code Predictor → generate cb1..cb15 ─────────────
            auto t_cp_start = std::chrono::high_resolution_clock::now();
            // Each frame is an independent CP sequence that restarts at position 0.
            // Clear the KV cache to reset it - far cheaper than freeing and recreating
            // the whole context (which also re-runs graph reservation) every frame.
            llama_memory_clear(llama_get_memory(cp_ctx), false);

            std::vector<int32_t> frame_codes(16, 0);
            frame_codes[0] = cb0_token;

            // Prefill CP with 2 embeddings: [talker_hidden, cb0_embd], each projected into CP space.
            std::vector<float> talker_h = project_cp(std::vector<float>(hidden, hidden + n_embd));
            std::vector<float> cb0_emb  = project_cp(codec_embed(cb0_token));

            llama_batch cp_prefill_batch = llama_batch_init(2, n_embd_cp, 1);
            cp_prefill_batch.n_tokens = 2;
            memcpy(cp_prefill_batch.embd,              talker_h.data(), n_embd_cp * sizeof(float));
            memcpy(cp_prefill_batch.embd + n_embd_cp,  cb0_emb.data(),  n_embd_cp * sizeof(float));
            cp_prefill_batch.pos[0] = 0; cp_prefill_batch.pos[1] = 1;
            cp_prefill_batch.n_seq_id[0] = 1; cp_prefill_batch.n_seq_id[1] = 1;
            cp_prefill_batch.seq_id[0][0] = 0; cp_prefill_batch.seq_id[1][0] = 0;
            cp_prefill_batch.logits[0] = 0; cp_prefill_batch.logits[1] = 1;

            ret = llama_decode(cp_ctx, cp_prefill_batch);
            llama_batch_free(cp_prefill_batch);

            if (ret != 0) {
                fprintf(stderr, "ERROR: CP prefill failed at frame %d: %d\n", frame, ret);
                break;
            }

            float * cp_hidden = llama_get_embeddings_ith(cp_ctx, -1);
            if (!cp_hidden || !std::isfinite(cp_hidden[0])) {
                fprintf(stderr, "ERROR: CP prefill produced invalid output at frame %d\n", frame);
                break;
            }
            // Step 1: apply lm_head[0] to get cb1
            std::vector<float> cp_logits_buf;
            std::vector<int32_t> cp_recent;
            cp_compute_logits(0, cp_hidden, cp_logits_buf);
            frame_codes[1] = tts_sample(cp_logits_buf.data(), cp_vocab,
                                         cp_sparams, cp_recent, rng);
            cp_recent.push_back(frame_codes[1]);

            // Steps 2-15: autoregressive decode for cb2..cb15
            int32_t prev_token = frame_codes[1];
            for (int cb_step = 1; cb_step < n_codebooks; cb_step++) {
                // Embed previous token using codec_embeds[cb_step - 1], projected into CP space
                std::vector<float> step_emb = project_cp(cp_codec_embed(cb_step - 1, prev_token));

                // Single-token decode
                llama_batch step_batch = llama_batch_init(1, n_embd_cp, 1);
                step_batch.n_tokens = 1;
                memcpy(step_batch.embd, step_emb.data(), n_embd_cp * sizeof(float));
                step_batch.pos[0]      = 2 + (cb_step - 1); // positions 2,3,4,...,15
                step_batch.n_seq_id[0] = 1;
                step_batch.seq_id[0][0] = 0;
                step_batch.logits[0]   = 1;

                ret = llama_decode(cp_ctx, step_batch);
                llama_batch_free(step_batch);

                if (ret != 0) {
                    fprintf(stderr, "ERROR: CP decode step %d failed at frame %d: %d\n", cb_step, frame, ret);
                    break;
                }

                float * step_hidden = llama_get_embeddings_ith(cp_ctx, -1);
                if (!step_hidden || !std::isfinite(step_hidden[0])) {
                    fprintf(stderr, "ERROR: CP step %d produced invalid output at frame %d\n", cb_step, frame);
                    break;
                }

                // Apply lm_head[cb_step] to get the next codebook token
                cp_compute_logits(cb_step, step_hidden, cp_logits_buf);
                frame_codes[cb_step + 1] = tts_sample(cp_logits_buf.data(), cp_vocab,
                                                       cp_sparams, cp_recent, rng);
                cp_recent.push_back(frame_codes[cb_step + 1]);
                prev_token = frame_codes[cb_step + 1];
            }

            auto t_cp_end = std::chrono::high_resolution_clock::now();
            cp_decode_ms += std::chrono::duration<double, std::milli>(t_cp_end - t_cp_start).count();

            all_codes.push_back(frame_codes);

            // ── Streaming: every stream_chunk frames, vocode a bounded window and emit the new tail ──
            // Decode [win_start, n_done) with win_start held stream_left_ctx frames behind the last
            // emitted frame, instead of from frame 0. The pre-transformer is causal + RoPE, so retained
            // frames keep the same relative geometry as a full decode; stream_left_ctx absorbs the
            // dropped-history error and the conv decoder's left receptive field. Right context is held
            // back by stream_margin. Each sample is committed once -> no seam artifacts, and per-chunk
            // cost is bounded (no O(n^2) re-decode of the whole sequence).
            if (streaming) {
                int n_done = (int)all_codes.size();
                if (n_done - last_decode_n >= stream_cur) {
                    stream_cur = std::min(stream_chunk, stream_cur * 2); // ramp toward the steady-state chunk
                    int safe_frames = n_done - stream_margin;
                    if (safe_frames * SAMPLES_PER_FRAME > emitted_samples) {
                        int emitted_frames = emitted_samples / SAMPLES_PER_FRAME;
                        int win_start = std::max(0, emitted_frames - stream_left_ctx);
                        int win_off   = win_start * SAMPLES_PER_FRAME; // global sample index of a[0]
                        std::vector<float> a = voc_decode_audio_range(voc, all_codes, win_start, n_done);
                        int target = std::min(win_off + (int)a.size(), safe_frames * SAMPLES_PER_FRAME);
                        for (int i = emitted_samples; i < target; i++) stream_audio.push_back(a[i - win_off]);
                        if (!ttfb_recorded && target > emitted_samples) {
                            ttfb_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::high_resolution_clock::now() - t_decode_start).count();
                            ttfb_recorded = true;
                            printf("  [stream] first audio chunk ready at %.0f ms (%d frames)\n",
                                   ttfb_ms, n_done);
                        }
                        emitted_samples = target;
                    }
                    last_decode_n = n_done;
                }
            }

            // Build next Talker input
            std::vector<float> next_embd(n_embd, 0.0f);
            std::vector<float> emb0 = codec_embed(cb0_token);
            for (int j = 0; j < n_embd; j++) next_embd[j] += emb0[j];
            for (int cb = 0; cb < n_codebooks; cb++) {
                std::vector<float> emb_cb = cp_codec_embed(cb, frame_codes[cb + 1]);
                for (int j = 0; j < n_embd; j++) next_embd[j] += emb_cb[j];
            }
            if (gen_step < n_trailing_text) {
                for (int j = 0; j < n_embd; j++)
                    next_embd[j] += trailing_text[gen_step * n_embd + j];
            } else {
                for (int j = 0; j < n_embd; j++)
                    next_embd[j] += pad_embed_vec[j];
            }

            // Talker decode (reuse pre-allocated batch)
            auto t_talker_step_start = std::chrono::high_resolution_clock::now();
            talker_step_batch.n_tokens = 1;
            memcpy(talker_step_batch.embd, next_embd.data(), n_embd * sizeof(float));
            for (int d = 0; d < 3; d++) talker_step_batch.pos[d] = talker_pos;
            talker_step_batch.pos[3] = 0;
            talker_step_batch.n_seq_id[0] = 1;
            talker_step_batch.seq_id[0][0] = 0;
            talker_step_batch.logits[0]   = 1;

            ret = llama_decode(talker_ctx, talker_step_batch);
            if (ret != 0) {
                fprintf(stderr, "ERROR: Talker decode failed at frame %d: %d\n", frame, ret);
                break;
            }

            hidden = llama_get_embeddings_ith(talker_ctx, -1);
            if (!hidden) {
                fprintf(stderr, "ERROR: no embeddings at frame %d\n", frame);
                break;
            }

            auto t_head_start = std::chrono::high_resolution_clock::now();
            compute_logits(hidden, cb0_logits);
            cb0_token = tts_sample(cb0_logits.data(), n_codec_vocab,
                                    talker_sparams, talker_recent_tokens, rng);
            talker_recent_tokens.push_back(cb0_token);
            auto t_head_end = std::chrono::high_resolution_clock::now();
            head_ms += std::chrono::duration<double, std::milli>(t_head_end - t_head_start).count();

            auto t_talker_step_end = std::chrono::high_resolution_clock::now();
            talker_decode_ms += std::chrono::duration<double, std::milli>(t_talker_step_end - t_talker_step_start).count();

            talker_pos++;
            gen_step++;

            printf("  frame %d: cb0=%d", frame, frame_codes[0]);
            for (int cb = 1; cb < 16; cb++) printf(" %d", frame_codes[cb]);
            printf("\n");
            fflush(stdout);
        }

        llama_batch_free(talker_step_batch);

        auto t_decode_end = std::chrono::high_resolution_clock::now();
        double total_decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
        int n_gen = (int)all_codes.size();

        printf("\nGeneration complete: %d frames\n", n_gen);
        printf("\n=== Performance ===\n");
        printf("  Prefill:  %d tokens in %.1f ms  (%.1f tok/s)\n",
               n_prefill, prefill_ms, n_prefill / (prefill_ms / 1000.0));
        printf("  Decode:   %d frames in %.1f ms  (%.2f frames/s)\n",
               n_gen, total_decode_ms, n_gen / (total_decode_ms / 1000.0));
        if (n_gen > 0) {
            printf("    Talker:    %.1f ms total  (%.1f ms/frame)\n",
                   talker_decode_ms, talker_decode_ms / n_gen);
            printf("    CP:        %.1f ms total  (%.1f ms/frame)\n",
                   cp_decode_ms, cp_decode_ms / n_gen);
            printf("    Head:      %.1f ms total  (%.1f ms/frame)\n",
                   head_ms, head_ms / n_gen);
            double audio_s = n_gen / 12.0;
            double wall_s = (prefill_ms + total_decode_ms) / 1000.0;
            printf("  Real-time factor: %.2fx  (%.2fs audio in %.2fs)\n",
                   audio_s / wall_s, audio_s, wall_s);
        }

        // ── Dump intermediates (for parity testing) ───────────────────────────
        if (!dump_dir.empty() && !all_codes.empty()) {
#ifdef _WIN32
            _mkdir(dump_dir.c_str());
#else
            mkdir(dump_dir.c_str(), 0755);
#endif
            std::string codes_path = dump_dir + "/all_codes.txt";
            FILE * fp = fopen(codes_path.c_str(), "w");
            if (fp) {
                fprintf(fp, "# frame cb0 cb1 cb2 ... cb15 (16 codebooks per frame)\n");
                for (size_t f = 0; f < all_codes.size(); f++) {
                    for (int c = 0; c < 16; c++) {
                        if (c > 0) fprintf(fp, " ");
                        fprintf(fp, "%d", all_codes[f][c]);
                    }
                    fprintf(fp, "\n");
                }
                fclose(fp);
                printf("Dumped codec codes to %s\n", codes_path.c_str());
            }
        }

        // ── Vocoder: decode codebooks → audio ────────────────────────────────────
        if (voc_ready && !all_codes.empty()) {
            if (streaming) {
                // Flush the remaining (held-back) tail with the final, complete decode.
                std::vector<float> a = voc_decode_audio(voc, all_codes, (int)all_codes.size());
                for (int i = emitted_samples; i < (int)a.size(); i++) stream_audio.push_back(a[i]);
                emitted_samples = (int)stream_audio.size();
                write_wav(output_path.c_str(), stream_audio.data(), emitted_samples, 24000);
                printf("Streaming: TTFB %.0f ms, total %d samples (%.2fs), chunk=%d frames\n",
                       ttfb_ms, emitted_samples, emitted_samples / 24000.0, stream_chunk);
            } else {
                std::vector<float> audio_data = voc_decode_audio(voc, all_codes, (int)all_codes.size());
                // Trim the leading reference-text audio in ICL mode (decoded with full context, then
                // dropped). Snap the cut to the nearest zero-crossing and fade in to avoid a click.
                int icl_trim = icl_ref_frames * SAMPLES_PER_FRAME;
                if (icl_trim > 0 && icl_trim < (int)audio_data.size()) {
                    int cut = find_zero_crossing(audio_data, icl_trim);
                    audio_data.erase(audio_data.begin(), audio_data.begin() + cut);
                    fade_in(audio_data, 240);
                    printf("ICL: trimmed %d ref frames, cut at %d samples (%.2fs), zero-crossing + fade-in\n",
                           icl_ref_frames, cut, cut / 24000.0);
                }
                write_wav(output_path.c_str(), audio_data.data(), (int)audio_data.size(), 24000);
            }
        } else if (!voc_ready) {
            printf("No vocoder loaded, skipping audio generation.\n");
        }

        llama_batch_free(batch);
    }

cleanup:
    llama_free(cp_ctx);
    llama_free(talker_ctx);
    llama_model_free(cp_model);
    llama_model_free(talker_model);

    return 0;
}
