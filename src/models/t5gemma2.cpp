#include "models.h"

#include "../llama-ext.h"
#include "../llama-kv-cache.h"
#include "../llama-kv-cache-iswa.h"

#include <cmath>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

void llama_model_t5gemma2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    hparams.dec_n_layer = hparams.n_layer();
    ml.get_key(LLM_KV_DECODER_BLOCK_COUNT, hparams.dec_n_layer, false);

    uint32_t dec_start_token_id = 0;
    if (ml.get_key(LLM_KV_DECODER_START_TOKEN_ID, dec_start_token_id, false)) {
        hparams.dec_start_token_id = dec_start_token_id;
    }

    const bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
    if (found_swa && hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        load_swa_pattern(ml, 6);
        ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
    } else {
        hparams.swa_type = LLAMA_SWA_TYPE_NONE;
    }

    hparams.f_embedding_scale = std::sqrt(float(hparams.n_embd));
    ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale, false);

    hparams.f_attention_scale = hparams.n_embd_head_k() > 0
        ? 1.0f / std::sqrt(float(hparams.n_embd_head_k()))
        : 0.0f;
    ml.get_key(LLM_KV_ATTENTION_SCALE, hparams.f_attention_scale, false);

    hparams.f_attn_logit_softcapping = 0.0f;
    if (ml.get_key(LLM_KV_ATTN_LOGIT_SOFTCAPPING, hparams.f_attn_logit_softcapping, false) &&
            std::isfinite(hparams.f_attn_logit_softcapping) && hparams.f_attn_logit_softcapping > 0.0f) {
        hparams.attn_soft_cap = true;
    } else {
        hparams.attn_soft_cap = false;
    }

    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);
    if (!std::isfinite(hparams.f_final_logit_softcapping)) {
        throw std::runtime_error("invalid t5gemma2 final_logit_softcapping");
    }

    std::string hidden_act;
    if (ml.get_key(LLM_KV_HIDDEN_ACT, hidden_act, false) && hidden_act != "gelu_pytorch_tanh") {
        throw std::runtime_error("unsupported t5gemma2 hidden_activation: " + hidden_act);
    }

    if (hparams.dec_n_layer == 0 || hparams.dec_n_layer > hparams.n_layer()) {
        throw std::runtime_error(format("invalid t5gemma2 decoder block count: %u (encoder blocks: %u)",
                hparams.dec_n_layer, hparams.n_layer()));
    }
    if (hparams.n_head() == 0 || hparams.n_head_kv() == 0 || hparams.n_head() % hparams.n_head_kv() != 0) {
        throw std::runtime_error(format("invalid t5gemma2 head counts: n_head=%u, n_head_kv=%u",
                hparams.n_head(), hparams.n_head_kv()));
    }
    if (hparams.n_embd_head_k_full == 0 || hparams.n_embd_head_k_full != hparams.n_embd_head_v_full ||
            hparams.n_embd_head_k_swa != hparams.n_embd_head_k_full ||
            hparams.n_embd_head_v_swa != hparams.n_embd_head_v_full) {
        throw std::runtime_error(format("invalid t5gemma2 head dims: k=%u/%u, v=%u/%u",
                hparams.n_embd_head_k_full, hparams.n_embd_head_k_swa,
                hparams.n_embd_head_v_full, hparams.n_embd_head_v_swa));
    }
    if (!std::isfinite(hparams.f_norm_rms_eps) || hparams.f_norm_rms_eps <= 0.0f ||
            !std::isfinite(hparams.f_embedding_scale) || hparams.f_embedding_scale <= 0.0f ||
            !std::isfinite(hparams.f_attention_scale) || hparams.f_attention_scale <= 0.0f) {
        throw std::runtime_error("invalid t5gemma2 norm eps or scale hparams");
    }

    switch (hparams.n_layer()) {
        case 18: type = LLM_TYPE_270M; break;
        case 26: type = LLM_TYPE_1B;   break;
        case 34: type = LLM_TYPE_4B;   break;
        default: type = LLM_TYPE_UNKNOWN;
    }

    const auto tn = LLM_TN(arch);
    if (ml.get_weight(tn(LLM_TENSOR_DECISION_KEY, "weight").str().c_str()) != nullptr) {
        // bongard judgment head outputs one score per candidate
        hparams.n_embd_out_impl = 1;
    }
}

void llama_model_t5gemma2::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    if (hparams.dec_start_token_id != LLAMA_TOKEN_NULL &&
            (hparams.dec_start_token_id < 0 || (int64_t) hparams.dec_start_token_id >= n_vocab)) {
        throw std::runtime_error(format("invalid t5gemma2 decoder_start_token_id: %d", hparams.dec_start_token_id));
    }

    const int64_t n_embd_q = n_head * n_embd_head_k;

    tok_embd        = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"), {n_embd, n_vocab}, 0);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_norm     = create_tensor(tn(LLM_TENSOR_DEC_OUTPUT_NORM, "weight"), {n_embd}, 0);

    const auto * w_dec_key    = ml.get_weight(tn(LLM_TENSOR_DECISION_KEY,        "weight").str().c_str());
    const auto * w_dec_query  = ml.get_weight(tn(LLM_TENSOR_DECISION_QUERY,      "weight").str().c_str());
    const auto * w_dec_scorer = ml.get_weight(tn(LLM_TENSOR_DECISION_SCORER_OUT, "weight").str().c_str());

    const bool any_dec = w_dec_key != nullptr || w_dec_query != nullptr || w_dec_scorer != nullptr;
    const bool all_dec = w_dec_key != nullptr && w_dec_query != nullptr && w_dec_scorer != nullptr;
    if (any_dec && !all_dec) {
        throw std::runtime_error("incomplete t5gemma2 bongard decision head tensors");
    }

    if (all_dec) {
        const int64_t head_rank = w_dec_key->tensor->ne[1];
        if (head_rank <= 0 || head_rank > 65536) {
            throw std::runtime_error(format("invalid t5gemma2 bongard head rank: %lld", (long long) head_rank));
        }

        decision_key        = create_tensor(tn(LLM_TENSOR_DECISION_KEY,        "weight"), {n_embd, head_rank}, 0);
        decision_query      = create_tensor(tn(LLM_TENSOR_DECISION_QUERY,      "weight"), {n_embd, head_rank}, 0);
        decision_scorer_out = create_tensor(tn(LLM_TENSOR_DECISION_SCORER_OUT, "weight"), {n_embd, 1},         0);

        const std::string cand_label = decision_labels.size() >= 1 ? decision_labels[0] : "<unused0>";
        const std::string dec_label  = decision_labels.size() >= 2 ? decision_labels[1] : "<unused1>";
        decision_candidate_end = vocab.text_to_token(cand_label);
        decision_end           = vocab.text_to_token(dec_label);
        if (decision_candidate_end == LLAMA_TOKEN_NULL || decision_end == LLAMA_TOKEN_NULL ||
                decision_candidate_end == decision_end) {
            throw std::runtime_error(format("invalid t5gemma2 bongard marker tokens: '%s', '%s'",
                    cand_label.c_str(), dec_label.c_str()));
        }
    } else {
        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
        if (output == NULL) {
            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm_enc      = create_tensor(tn(LLM_TENSOR_ENC_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_q_norm_enc    = create_tensor(tn(LLM_TENSOR_ENC_ATTN_Q_NORM,    "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm_enc    = create_tensor(tn(LLM_TENSOR_ENC_ATTN_K_NORM,    "weight", i), {n_embd_head_k}, 0);
        layer.attn_post_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.wq_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_Q,   "weight", i), {n_embd, n_embd_q},     0);
        layer.wk_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_OUT, "weight", i), {n_embd_q, n_embd},     0);

        layer.ffn_norm_enc      = create_tensor(tn(LLM_TENSOR_ENC_FFN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_post_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_POST_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate_enc      = create_tensor(tn(LLM_TENSOR_ENC_FFN_GATE,      "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up_enc        = create_tensor(tn(LLM_TENSOR_ENC_FFN_UP,        "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down_enc      = create_tensor(tn(LLM_TENSOR_ENC_FFN_DOWN,      "weight", i), {n_ff, n_embd}, 0);
    }

    const int dec_n_layer = hparams.dec_n_layer;
    for (int i = 0; i < dec_n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_DEC_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_DEC_ATTN_Q_NORM,    "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm    = create_tensor(tn(LLM_TENSOR_DEC_ATTN_K_NORM,    "weight", i), {n_embd_head_k}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_DEC_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_DEC_ATTN_Q,   "weight", i), {n_embd, n_embd_q},     0);
        layer.wk = create_tensor(tn(LLM_TENSOR_DEC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_DEC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_DEC_ATTN_OUT, "weight", i), {n_embd_q, n_embd},     0);

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_DEC_FFN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_DEC_FFN_POST_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate      = create_tensor(tn(LLM_TENSOR_DEC_FFN_GATE,      "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up        = create_tensor(tn(LLM_TENSOR_DEC_FFN_UP,        "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down      = create_tensor(tn(LLM_TENSOR_DEC_FFN_DOWN,      "weight", i), {n_ff, n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_t5gemma2::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph_enc>(params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
                return std::make_unique<graph_dec<true>>(params);
            }
            return std::make_unique<graph_dec<false>>(params);
        default:
            GGML_ABORT("invalid graph type");
    }
}

class llama_model_t5gemma2::input_chunk_pool : public llm_graph_input_i {
public:
    explicit input_chunk_pool(uint32_t chunk_size) : chunk_size(chunk_size) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(weights != nullptr);
        const int64_t n_tokens = weights->ne[0];
        const int64_t n_chunks = weights->ne[1];
        std::vector<float> buf((size_t) (n_tokens * n_chunks), 0.0f);

        for (int64_t chunk = 0; chunk < n_chunks; ++chunk) {
            const int64_t begin = chunk * chunk_size;
            const int64_t end   = std::min<int64_t>(begin + chunk_size, n_tokens);
            int64_t valid = 0;
            for (int64_t i = begin; i < end; ++i) {
                if (!ubatch->token || ubatch->token[i] != 0) {
                    valid++;
                }
            }
            if (valid == 0) {
                continue;
            }
            const float scale = 1.0f / (float) valid;
            for (int64_t i = begin; i < end; ++i) {
                if (!ubatch->token || ubatch->token[i] != 0) {
                    buf[(size_t) (chunk * n_tokens + i)] = scale;
                }
            }
        }

        ggml_backend_tensor_set(weights, buf.data(), 0, ggml_nbytes(weights));
    }

    bool can_reuse(const llm_graph_params & params) override {
        return params.cparams.encoder_chunk_size == chunk_size;
    }

    uint32_t chunk_size;
    ggml_tensor * weights = nullptr;
};

llama_model_t5gemma2::graph_enc::graph_enc(const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_k();

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    if (cparams.encoder_chunk_size > 1) {
        const float hf_embd_scale = ggml_bf16_to_fp32(ggml_fp32_to_bf16(hparams.f_embedding_scale));
        inpL = ggml_scale(ctx0, inpL, hf_embd_scale / hparams.f_embedding_scale);
    }

    ggml_tensor * inp_pos = build_inp_pos();

    // encoder uses bidirectional half-open SWA while hparams.swa_type stays STANDARD for decoder KV
    auto * inp_attn = build_attn_inp_no_cache(/*swa_bidir_half_open=*/true);

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    input_chunk_pool * inp_pool = nullptr;
    if (cparams.encoder_chunk_size > 1) {
        const int64_t n_chunks = (n_tokens + cparams.encoder_chunk_size - 1) / cparams.encoder_chunk_size;
        auto pool_ptr = std::make_unique<input_chunk_pool>(cparams.encoder_chunk_size);
        pool_ptr->weights = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, n_chunks);
        ggml_set_input(pool_ptr->weights);
        inp_pool = static_cast<input_chunk_pool *>(res->add_input(std::move(pool_ptr)));
    }

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * cur = build_norm(inpL, layer.attn_norm_enc, nullptr, LLM_NORM_RMS, il);
        cb(cur, "enc_attn_norm", il);

        {
            ggml_tensor * Qcur = build_lora_mm(layer.wq_enc, cur);
            ggml_tensor * Kcur = build_lora_mm(layer.wk_enc, cur);
            ggml_tensor * Vcur = build_lora_mm(layer.wv_enc, cur);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = build_norm(Qcur, layer.attn_q_norm_enc, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "enc_Qcur_normed", il);
            Qcur = build_rope(Qcur, inp_pos, il);

            Kcur = build_norm(Kcur, layer.attn_k_norm_enc, nullptr, LLM_NORM_RMS, il);
            cb(Kcur, "enc_Kcur_normed", il);
            Kcur = build_rope(Kcur, inp_pos, il);

            cb(Qcur, "enc_Qcur", il);
            cb(Kcur, "enc_Kcur", il);
            cb(Vcur, "enc_Vcur", il);

            cur = build_attn(inp_attn,
                    layer.wo_enc, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    hparams.f_attention_scale, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        cur = build_norm(cur, layer.attn_post_norm_enc, nullptr, LLM_NORM_RMS, il);
        cb(cur, "enc_attn_post_norm", il);

        ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
        cb(sa_out, "enc_sa_out", il);

        cur = build_norm(sa_out, layer.ffn_norm_enc, nullptr, LLM_NORM_RMS, il);
        cb(cur, "enc_ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up_enc,   nullptr, nullptr,
                layer.ffn_gate_enc, nullptr, nullptr,
                layer.ffn_down_enc, nullptr, nullptr,
                nullptr,
                LLM_FFN_GELU, LLM_FFN_PAR, il);
        cb(cur, "enc_ffn_out", il);

        cur = build_norm(cur, layer.ffn_post_norm_enc, nullptr, LLM_NORM_RMS, il);
        cb(cur, "enc_ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, sa_out);
        cur = build_cvec(cur, il);
        cb(cur, "enc_l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm_enc, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    if (inp_pool != nullptr) {
        cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)), inp_pool->weights);
        cb(cur, "enc_final_pooled", -1);
    }

    res->t_embd = cur;
}

struct bongard_rows {
    std::vector<int32_t> options;
    std::vector<int32_t> queries;
    bool valid          = false;
    bool explicit_input = false;
};

// extracts (candidate_end, decision_end) output rows from decision_order or Bongard marker tokens
static bongard_rows bongard_get_rows(
        const llama_ubatch & ubatch,
        llama_token candidate_end_tok,
        llama_token decision_end_tok) {
    bongard_rows res;

    const int32_t n_tokens = ubatch.n_tokens;
    std::vector<int32_t> rows(n_tokens, -1);
    int32_t n_outputs = 0;
    for (int32_t i = 0; i < n_tokens; ++i) {
        if (ubatch.output[i]) {
            rows[i] = n_outputs++;
        }
    }

    bool ok = n_outputs > 0;
    std::vector<std::pair<llama_seq_id, int32_t>> opt_list;
    std::map<llama_seq_id, int32_t> seq_query;
    std::map<llama_seq_id, int32_t> seq_opt_count;

    if (ubatch.decision_order != nullptr) {
        res.explicit_input = true;
        for (int32_t i = 0; ok && i < n_tokens;) {
            const int32_t order = ubatch.decision_order[i];
            const llama_seq_id seq = ubatch.n_seq_id[i] > 0 ? ubatch.seq_id[i][0] : -1;
            int32_t end = i + 1;
            while (end < n_tokens &&
                    ubatch.decision_order[end] == order &&
                    (ubatch.n_seq_id[end] > 0 ? ubatch.seq_id[end][0] : -1) == seq) {
                end++;
            }
            if (order != LLAMA_DECISION_ORDER_NONE) {
                if (ubatch.n_seq_id[end - 1] != 1 || rows[end - 1] < 0) {
                    ok = false;
                } else {
                    switch (order) {
                        case LLAMA_DECISION_ORDER_QUESTION_NOUL:
                        case LLAMA_DECISION_ORDER_QUESTION_CHOICE:
                        case LLAMA_DECISION_ORDER_QUESTION_SCORE:
                            if (!seq_query.emplace(seq, rows[end - 1]).second) {
                                ok = false;
                            }
                            break;
                        case LLAMA_DECISION_ORDER_OPTION:
                            opt_list.emplace_back(seq, rows[end - 1]);
                            seq_opt_count[seq]++;
                            break;
                        default:
                            ok = false;
                            break;
                    }
                }
            }
            i = end;
        }
    } else if (ubatch.token != nullptr && candidate_end_tok != LLAMA_TOKEN_NULL && decision_end_tok != LLAMA_TOKEN_NULL) {
        for (int32_t i = 0; ok && i < n_tokens; ++i) {
            const llama_token tok = ubatch.token[i];
            if (tok == candidate_end_tok) {
                res.explicit_input = true;
                if (ubatch.n_seq_id[i] != 1 || rows[i] < 0) {
                    ok = false;
                } else {
                    const llama_seq_id seq = ubatch.seq_id[i][0];
                    opt_list.emplace_back(seq, rows[i]);
                    seq_opt_count[seq]++;
                }
            } else if (tok == decision_end_tok) {
                res.explicit_input = true;
                if (ubatch.n_seq_id[i] != 1 || rows[i] < 0) {
                    ok = false;
                } else {
                    const llama_seq_id seq = ubatch.seq_id[i][0];
                    if (!seq_query.emplace(seq, rows[i]).second) {
                        ok = false;
                    }
                }
            }
        }
    }

    if (ok && !opt_list.empty() && seq_query.size() == seq_opt_count.size()) {
        res.options.reserve(opt_list.size());
        res.queries.reserve(opt_list.size());
        for (const auto & item : opt_list) {
            const auto it = seq_query.find(item.first);
            if (it == seq_query.end()) {
                ok = false;
                break;
            }
            res.options.push_back(item.second);
            res.queries.push_back(it->second);
        }
    } else {
        ok = false;
    }

    res.valid = ok && !res.options.empty() && (int32_t) res.options.size() <= n_outputs;
    if (!res.valid) {
        res.options.assign(1, 0);
        res.queries.assign(1, 0);
    }
    return res;
}

class llama_model_t5gemma2::input_decision : public llm_graph_input_i {
public:
    input_decision(const llama_ubatch & ubatch, int64_t n_outputs, llama_token candidate_end, llama_token decision_end) :
        candidate_end_tok(candidate_end),
        decision_end_tok(decision_end),
        n_outputs(n_outputs),
        n_options(bongard_get_rows(ubatch, candidate_end, decision_end).options.size()) {}

    void set_input(const llama_ubatch * ubatch) override {
        const auto rows = bongard_get_rows(*ubatch, candidate_end_tok, decision_end_tok);
        GGML_ASSERT(rows.options.size() == n_options);

        // signal NaN on malformed explicit markers/decision_order; keep 0 on dummy reserve batches
        const float status_data = (rows.valid || !rows.explicit_input) ? 0.0f : NAN;

        ggml_backend_tensor_set(options, rows.options.data(), 0, ggml_nbytes(options));
        ggml_backend_tensor_set(queries, rows.queries.data(), 0, ggml_nbytes(queries));
        ggml_backend_tensor_set(status,  &status_data,        0, ggml_nbytes(status));
    }

    bool can_reuse(const llm_graph_params & params) override {
        if (params.n_outputs != n_outputs) {
            return false;
        }
        return bongard_get_rows(params.ubatch, candidate_end_tok, decision_end_tok).options.size() == n_options;
    }

    const llama_token candidate_end_tok;
    const llama_token decision_end_tok;
    const int64_t     n_outputs;
    const size_t      n_options;

    ggml_tensor * options = nullptr; // I32 [n_options]
    ggml_tensor * queries = nullptr; // I32 [n_options]
    ggml_tensor * status  = nullptr; // F32 [1]
};

template <bool iswa>
llama_model_t5gemma2::graph_dec<iswa>::graph_dec(const llm_graph_params & params) : llm_graph_context(params) {
    const auto & t5g2 = static_cast<const llama_model_t5gemma2 &>(model);
    const int64_t n_embd_head = hparams.n_embd_head_k();

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    if (cparams.encoder_chunk_size > 1) {
        const float hf_embd_scale = ggml_bf16_to_fp32(ggml_fp32_to_bf16(hparams.f_embedding_scale));
        inpL = ggml_scale(ctx0, inpL, hf_embd_scale / hparams.f_embedding_scale);
    }

    ggml_tensor * inp_pos  = build_inp_pos();
    ggml_tensor * embd_enc = build_inp_cross_embd();

    const int64_t n_enc = embd_enc->ne[1];

    using inp_attn_type = std::conditional_t<iswa, llm_graph_input_attn_kv_iswa, llm_graph_input_attn_kv>;
    inp_attn_type * inp_attn_self = nullptr;
    if constexpr (iswa) {
        inp_attn_self = build_attn_inp_kv_iswa();
    } else {
        inp_attn_self = build_attn_inp_kv();
    }

    auto * inp_attn_cross = build_attn_inp_cross();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    input_decision * inp_decision = nullptr;
    if (t5g2.has_decision_head() && n_outputs > 0) {
        auto inp_ptr = std::make_unique<input_decision>(ubatch, n_outputs, t5g2.decision_candidate_end, t5g2.decision_end);
        inp_decision = inp_ptr.get();
        inp_decision->options = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, inp_decision->n_options);
        inp_decision->queries = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, inp_decision->n_options);
        inp_decision->status  = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
        for (ggml_tensor * t : { inp_decision->options, inp_decision->queries, inp_decision->status }) {
            ggml_set_input(t);
        }
        res->add_input(std::move(inp_ptr));
    }

    const int dec_n_layer = hparams.dec_n_layer;

    for (int il = 0; il < dec_n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * cur = build_norm(inpL, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // merged self-attention and cross-attention
        {
            auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);
            Qcur = build_rope(Qcur, inp_pos, il);

            Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);
            Kcur = build_rope(Kcur, inp_pos, il);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // cross K uses k_norm with no RoPE
            ggml_tensor * Kcross = build_lora_mm(layer.wk, embd_enc, layer.wk_s);
            ggml_tensor * Vcross = build_lora_mm(layer.wv, embd_enc, layer.wv_s);
            Kcross = ggml_reshape_3d(ctx0, Kcross, n_embd_head, n_head_kv, n_enc);
            Vcross = ggml_reshape_3d(ctx0, Vcross, n_embd_head, n_head_kv, n_enc);
            Kcross = build_norm(Kcross, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
            cb(Kcross, "Kcross_normed", il);
            cb(Vcross, "Vcross", il);

            const llama_kv_cache_context * mctx_cur = nullptr;
            ggml_tensor * k_idxs       = nullptr;
            ggml_tensor * v_idxs       = nullptr;
            ggml_tensor * self_kq_mask = nullptr;
            ggml_tensor * k_rot        = nullptr;
            ggml_tensor * v_rot        = nullptr;

            if constexpr (iswa) {
                const bool is_swa = hparams.is_swa(il);
                mctx_cur     = is_swa ? inp_attn_self->mctx->get_swa()       : inp_attn_self->mctx->get_base();
                k_idxs       = is_swa ? inp_attn_self->get_k_idxs_swa()      : inp_attn_self->get_k_idxs();
                v_idxs       = is_swa ? inp_attn_self->get_v_idxs_swa()      : inp_attn_self->get_v_idxs();
                self_kq_mask = is_swa ? inp_attn_self->self_attn_mask_swa.raw : inp_attn_self->self_attn_mask.raw;
                k_rot        = is_swa ? inp_attn_self->self_k_rot_swa[il]    : inp_attn_self->self_k_rot[il];
                v_rot        = is_swa ? inp_attn_self->self_v_rot_swa[il]    : inp_attn_self->self_v_rot[il];
            } else {
                mctx_cur     = inp_attn_self->mctx;
                k_idxs       = inp_attn_self->self_k_idxs;
                v_idxs       = inp_attn_self->self_v_idxs;
                self_kq_mask = inp_attn_self->self_attn_mask.raw;
                k_rot        = inp_attn_self->self_k_rot[il];
                v_rot        = inp_attn_self->self_v_rot[il];
            }

            if (k_rot) {
                Qcur   = llama_mul_mat_hadamard(ctx0, Qcur,   k_rot);
                Kcur   = llama_mul_mat_hadamard(ctx0, Kcur,   k_rot);
                Kcross = llama_mul_mat_hadamard(ctx0, Kcross, k_rot);
            }
            if (v_rot) {
                Vcur   = llama_mul_mat_hadamard(ctx0, Vcur,   v_rot);
                Vcross = llama_mul_mat_hadamard(ctx0, Vcross, v_rot);
            }

            if (ggml_tensor * k_scale = mctx_cur->get_k_scale(il)) {
                Qcur   = ggml_mul(ctx0, Qcur,   k_scale);
                Kcur   = ggml_div(ctx0, Kcur,   k_scale);
                Kcross = ggml_div(ctx0, Kcross, k_scale);
            }
            if (ggml_tensor * v_scale = mctx_cur->get_v_scale(il)) {
                Vcur   = ggml_div(ctx0, Vcur,   v_scale);
                Vcross = ggml_div(ctx0, Vcross, v_scale);
            }

            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, Vcur);

            const bool use_kv_cur = cparams.training;
            if (!use_kv_cur) {
                ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, k_idxs, il));
                ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur, v_idxs, il));
            }

            ggml_tensor * k_self = use_kv_cur ? Kcur : mctx_cur->get_k(ctx0, il);
            ggml_tensor * v_self = use_kv_cur ? Vcur : mctx_cur->get_v(ctx0, il);

            // undo transposed V layout before concat so build_attn_mha sees a uniform layout
            if (v_self->nb[1] > v_self->nb[2]) {
                v_self = ggml_cont(ctx0, ggml_permute(ctx0, v_self, 2, 1, 0, 3));
            }

            ggml_tensor * cross_kq_mask = inp_attn_cross->cross_attn_mask.raw;
            const int64_t n_stream = k_self->ne[3];
            if (n_stream > 1) {
                Kcross = ggml_repeat_4d(ctx0, Kcross, Kcross->ne[0], Kcross->ne[1], Kcross->ne[2], n_stream);
                Vcross = ggml_repeat_4d(ctx0, Vcross, Vcross->ne[0], Vcross->ne[1], Vcross->ne[2], n_stream);
                cross_kq_mask = ggml_reshape_4d(ctx0, cross_kq_mask, cross_kq_mask->ne[0], cross_kq_mask->ne[1]/n_stream, 1, n_stream);
            }

            if (Kcross->type != k_self->type) {
                Kcross = ggml_cast(ctx0, Kcross, k_self->type);
            }
            if (Vcross->type != v_self->type) {
                Vcross = ggml_cast(ctx0, Vcross, v_self->type);
            }

            ggml_tensor * k = ggml_concat(ctx0, k_self, Kcross, 2);
            ggml_tensor * v = ggml_concat(ctx0, v_self, Vcross, 2);
            ggml_tensor * mask = ggml_concat(ctx0, self_kq_mask, cross_kq_mask, 0);

            cur = build_attn_mha(Qcur, k, v, nullptr, mask, nullptr, nullptr, nullptr, 0, hparams.f_attention_scale, il);

            if (ggml_tensor * v_scale = mctx_cur->get_v_scale(il)) {
                cur = ggml_mul(ctx0, cur, v_scale);
            }
            if (v_rot) {
                cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
            }

            cb(cur, "kqv_out", il);

            cur = build_lora_mm(layer.wo, cur, layer.wo_s);
        }

        if (il == dec_n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
        cb(sa_out, "sa_out", il);

        cur = build_norm(sa_out, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   nullptr, nullptr,
                layer.ffn_gate, nullptr, nullptr,
                layer.ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_GELU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = build_norm(cur, layer.ffn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, sa_out);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    if (t5g2.has_decision_head()) {
        if (inp_decision != nullptr) {
            cur = build_decision_head(cur, inp_decision, t5g2);
            if (n_outputs > cur->ne[1]) {
                cur = ggml_pad(ctx0, cur, 0, n_outputs - cur->ne[1], 0, 0);
            }
            cur = ggml_add(ctx0, cur, inp_decision->status);
            cb(cur, "result_decision", -1);
        }
        res->t_embd = cur;
        return;
    }

    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);

    if (hparams.f_final_logit_softcapping > 0.0f) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

template <bool iswa>
ggml_tensor * llama_model_t5gemma2::graph_dec<iswa>::build_decision_head(
        ggml_tensor * hidden, input_decision * inp, const llama_model_t5gemma2 & t5g2) {
    ggml_tensor * options = ggml_get_rows(ctx0, hidden, inp->options); // [n_embd, n_options]
    ggml_tensor * queries = ggml_get_rows(ctx0, hidden, inp->queries); // [n_embd, n_options]

    ggml_tensor * linear   = build_lora_mm(t5g2.decision_scorer_out, options); // [1, n_options]
    ggml_tensor * cand     = build_lora_mm(t5g2.decision_key,        options); // [head_rank, n_options]
    ggml_tensor * glob     = build_lora_mm(t5g2.decision_query,      queries); // [head_rank, n_options]
    ggml_tensor * bilinear = ggml_sum_rows(ctx0, ggml_mul(ctx0, cand, glob));
    bilinear = ggml_scale(ctx0, bilinear, 1.0f / std::sqrt((float) cand->ne[0])); // [1, n_options]

    ggml_tensor * scores = ggml_add(ctx0, linear, bilinear);
    cb(scores, "decision_scores", -1);
    return scores;
}

template struct llama_model_t5gemma2::graph_dec<false>;
template struct llama_model_t5gemma2::graph_dec<true>;
