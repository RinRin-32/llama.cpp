#include "models.h"

void llama_model_qwen3tts_cp::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_TTS_NUM_CODE_GROUPS, hparams.tts_num_code_groups, false);
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_qwen3tts_cp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int n_code_groups  = hparams.tts_num_code_groups > 0 ? hparams.tts_num_code_groups : 16;
    const int n_cp_codebooks = n_code_groups - 1; // 15

    uint32_t n_cp_vocab_u32 = 0;
    ml.get_key(LLM_KV_VOCAB_SIZE, n_cp_vocab_u32, false);

    // Projection from talker hidden to predictor hidden. Present only when the talker and
    // code-predictor hidden sizes differ (e.g. 1.7B: 2048 -> 1024); for 0.6B the dims are
    // equal and the tensor is absent. It is applied host-side by the CLI before feeding
    // embeddings, so the graph stays purely in CP-dim -- just account for the tensors here
    // so done_getting_tensors() passes.
    if (ml.get_weight("tts.cp.small_to_mtp.weight")) {
        ml.n_created += 1;
        if (ml.get_weight("tts.cp.small_to_mtp.bias")) {
            ml.n_created += 1;
        }
    }

    // output norm
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    // per-codebook embeddings and lm_heads are indexed by codebook (0..14), not by layer,
    // and are loaded externally by the CLI tool. Account for them so done_getting_tensors() passes.
    tts_cp_codec_embd.resize(n_cp_codebooks);
    tts_cp_lm_head.resize(n_cp_codebooks);
    for (int cb = 0; cb < n_cp_codebooks; ++cb) {
        std::string embd_name = format("tts.cp.codec_embd.%d.weight", cb);
        std::string head_name = format("tts.cp.lm_head.%d.weight", cb);
        const auto * embd_w = ml.get_weight(embd_name.c_str());
        const auto * head_w = ml.get_weight(head_name.c_str());
        if (!embd_w || !head_w) {
            throw std::runtime_error(format("missing codebook tensor: %s or %s", embd_name.c_str(), head_name.c_str()));
        }
        ml.n_created += 2;
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen3tts_cp::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen3tts_cp::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // The code predictor always receives pre-computed embeddings via ubatch.embd:
    //   - first call per frame: [talker_hidden_projected, cb0_embed] (2 tokens)
    //   - subsequent calls: [prev_cb_embed_projected] (1 token)
    // There is no tok_embd for this model, pass nullptr.
    inpL = build_inp_embd(nullptr);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);

    // The code predictor exposes the hidden state as the embedding output. The CLI tool
    // applies the per-codebook lm_head externally (each of the sequential steps uses a
    // different lm_head).
    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}
