#include "common.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::ordered_json;

constexpr int32_t ENCODER_MAX_TOKENS = 1024;
constexpr int32_t QUERY_MAX_TOKENS = 512;
constexpr uint32_t ENCODER_CHUNK_SIZE = 4;
constexpr llama_token PAD_TOKEN_ID = 0;
constexpr llama_token BOS_TOKEN_ID = 2;
constexpr llama_token YES_TOKEN_ID = 4443;
constexpr llama_token NO_TOKEN_ID = 1904;

constexpr const char * DEFAULT_INSTRUCTION =
    "Given a query, retrieve documents that answer the query.";
constexpr const char * SYSTEM_INSTRUCTION =
    "Judge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. Note that the answer can only be \"yes\" or \"no\".";

struct options {
    std::string model;
    std::string query;
    std::string passage;
    std::string instruction = DEFAULT_INSTRUCTION;
    std::string jsonl;
    std::string output;
    int32_t n_gpu_layers = -1;
    int32_t n_threads = 0;
    uint32_t n_ctx = 1024;
    bool query_set = false;
    bool passage_set = false;
    bool instruction_set = false;
    bool dump_tokens = false;
    bool require_gpu = false;
};

struct input_row {
    std::string case_id;
    std::string qid;
    std::string pid;
    std::string query;
    std::string passage;
    std::string instruction;
    int32_t line = 0;
    bool has_ids = false;
};

struct prepared_input {
    std::vector<llama_token> encoder_tokens;
    std::vector<llama_token> decoder_tokens;
    bool encoder_truncated = false;
    bool query_truncated = false;
};

struct model_deleter {
    void operator()(llama_model * value) const {
        llama_model_free(value);
    }
};

struct context_deleter {
    void operator()(llama_context * value) const {
        llama_free(value);
    }
};

struct batch_owner {
    explicit batch_owner(int32_t capacity) :
        batch(llama_batch_init(capacity, 0, 1)) {}

    batch_owner(const batch_owner &) = delete;
    batch_owner & operator=(const batch_owner &) = delete;

    batch_owner(batch_owner && other) noexcept :
        batch(other.batch) {
        other.batch = {};
    }

    ~batch_owner() {
        if (batch.token != nullptr || batch.embd != nullptr) {
            llama_batch_free(batch);
        }
    }

    llama_batch batch;
};

void print_usage(const char * argv0) {
    std::fprintf(
        stderr,
        "usage:\n"
        "  %s -m MODEL [runtime options] --query TEXT --passage TEXT [--instruction TEXT]\n"
        "  %s -m MODEL [runtime options] --jsonl FILE [--out FILE]\n\n"
        "runtime options:\n"
        "  -ngl, --n-gpu-layers N  layers to offload (-1 means all; default: -1)\n"
        "  -t, --threads N          CPU worker threads (default: llama.cpp default)\n"
        "  -c, --ctx-size N         context/batch size, at least 1024 (default: 1024)\n"
        "  --dump-tokens            include unpadded input token IDs in JSON output\n"
        "  --require-gpu            fail if no GPU backend is visible or -ngl is 0\n",
        argv0, argv0);
}

std::string take_value(int argc, char ** argv, int & index) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[index]);
    }
    return argv[++index];
}

int32_t parse_i32(const std::string & value, const char * name) {
    size_t consumed = 0;
    long parsed = 0;
    try {
        parsed = std::stol(value, &consumed);
    } catch (const std::exception &) {
        throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    if (consumed != value.size() ||
        parsed < std::numeric_limits<int32_t>::min() ||
        parsed > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument(std::string(name) + " must be a 32-bit integer");
    }
    return (int32_t) parsed;
}

options parse_options(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (arg == "-m" || arg == "--model") {
            result.model = take_value(argc, argv, i);
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            result.n_gpu_layers = parse_i32(take_value(argc, argv, i), "--n-gpu-layers");
        } else if (arg == "-t" || arg == "--threads") {
            result.n_threads = parse_i32(take_value(argc, argv, i), "--threads");
        } else if (arg == "-c" || arg == "--ctx-size") {
            const int32_t value = parse_i32(take_value(argc, argv, i), "--ctx-size");
            if (value < ENCODER_MAX_TOKENS) {
                throw std::invalid_argument("--ctx-size must be at least 1024");
            }
            result.n_ctx = (uint32_t) value;
        } else if (arg == "--query") {
            result.query = take_value(argc, argv, i);
            result.query_set = true;
        } else if (arg == "--passage") {
            result.passage = take_value(argc, argv, i);
            result.passage_set = true;
        } else if (arg == "--instruction") {
            result.instruction = take_value(argc, argv, i);
            result.instruction_set = true;
        } else if (arg == "--jsonl") {
            result.jsonl = take_value(argc, argv, i);
        } else if (arg == "--out") {
            result.output = take_value(argc, argv, i);
        } else if (arg == "--dump-tokens") {
            result.dump_tokens = true;
        } else if (arg == "--require-gpu") {
            result.require_gpu = true;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (result.model.empty()) {
        throw std::invalid_argument("--model is required");
    }
    if (result.n_threads < 0) {
        throw std::invalid_argument("--threads must be positive");
    }
    if (result.require_gpu && result.n_gpu_layers == 0) {
        throw std::invalid_argument("--require-gpu is incompatible with -ngl 0");
    }

    const bool single_mode = result.query_set || result.passage_set;
    const bool jsonl_mode = !result.jsonl.empty();
    if (single_mode == jsonl_mode) {
        throw std::invalid_argument("select exactly one input mode: --query/--passage or --jsonl");
    }
    if (single_mode && (!result.query_set || !result.passage_set)) {
        throw std::invalid_argument("single mode requires both --query and --passage");
    }
    if (jsonl_mode && result.instruction_set) {
        throw std::invalid_argument("--instruction is only valid in single mode; use the JSONL field");
    }
    return result;
}

std::string required_string(
        const json & value,
        const char * field,
        const std::string & source,
        int32_t line) {
    const auto iterator = value.find(field);
    if (iterator == value.end() || !iterator->is_string()) {
        throw std::invalid_argument(
            source + ":" + std::to_string(line) + ": field '" + field + "' must be a string");
    }
    return iterator->get<std::string>();
}

std::vector<input_row> read_jsonl(const std::string & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open JSONL input: " + path);
    }

    std::vector<input_row> rows;
    std::set<std::pair<std::string, std::string>> ids;
    std::string line;
    int32_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }

        json value;
        try {
            value = json::parse(line);
        } catch (const json::exception & error) {
            throw std::invalid_argument(
                path + ":" + std::to_string(line_number) + ": invalid JSON: " + error.what());
        }
        if (!value.is_object()) {
            throw std::invalid_argument(
                path + ":" + std::to_string(line_number) + ": expected a JSON object");
        }

        input_row row;
        row.qid = required_string(value, "qid", path, line_number);
        row.pid = required_string(value, "pid", path, line_number);
        row.query = required_string(value, "query", path, line_number);
        row.passage = required_string(value, "passage", path, line_number);
        row.instruction = value.contains("instruction")
            ? required_string(value, "instruction", path, line_number)
            : DEFAULT_INSTRUCTION;
        row.case_id = row.qid + "/" + row.pid;
        row.line = line_number;
        row.has_ids = true;

        if (!ids.emplace(row.qid, row.pid).second) {
            throw std::invalid_argument(
                path + ":" + std::to_string(line_number) + ": duplicate qid/pid pair " + row.case_id);
        }
        rows.push_back(std::move(row));
    }
    if (rows.empty()) {
        throw std::invalid_argument(path + ": JSONL input contains no rows");
    }
    return rows;
}

std::string model_metadata(
        const llama_model * model,
        const char * key) {
    char small[128];
    const int32_t length = llama_model_meta_val_str(model, key, small, sizeof(small));
    if (length < 0) {
        return {};
    }
    if (length < (int32_t) sizeof(small)) {
        return std::string(small, length);
    }
    std::vector<char> buffer((size_t) length + 1);
    const int32_t check = llama_model_meta_val_str(model, key, buffer.data(), buffer.size());
    if (check != length) {
        throw std::runtime_error(std::string("failed to read model metadata: ") + key);
    }
    return std::string(buffer.data(), length);
}

void validate_model(const llama_model * model) {
    const std::string architecture = model_metadata(model, "general.architecture");
    if (architecture != "t5gemma2") {
        throw std::runtime_error("expected general.architecture=t5gemma2, got '" + architecture + "'");
    }
    if (!llama_model_has_encoder(model) || !llama_model_has_decoder(model)) {
        throw std::runtime_error("KaLM reranker requires both encoder and decoder capabilities");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (llama_vocab_pad(vocab) != PAD_TOKEN_ID) {
        throw std::runtime_error("KaLM reranker requires PAD token id 0");
    }
    if (llama_vocab_bos(vocab) != BOS_TOKEN_ID ||
        llama_model_decoder_start_token(model) != BOS_TOKEN_ID) {
        throw std::runtime_error("KaLM reranker requires BOS/decoder start token id 2");
    }
    if (llama_vocab_n_tokens(vocab) <= std::max(YES_TOKEN_ID, NO_TOKEN_ID)) {
        throw std::runtime_error("vocabulary does not contain yes/no token ids");
    }

    const auto yes = common_tokenize(vocab, "yes", false, true);
    const auto no = common_tokenize(vocab, "no", false, true);
    if (yes != std::vector<llama_token>{YES_TOKEN_ID} ||
        no != std::vector<llama_token>{NO_TOKEN_ID}) {
        throw std::runtime_error("answer token mismatch: expected yes=[4443], no=[1904]");
    }
}

bool contains_pad(const std::vector<llama_token> & tokens) {
    return std::find(tokens.begin(), tokens.end(), PAD_TOKEN_ID) != tokens.end();
}

prepared_input prepare_input(
        const llama_vocab * vocab,
        const input_row & row,
        uint32_t n_ctx) {
    prepared_input result;

    const std::string encoder_text = "<Document>: " + row.passage;
    result.encoder_tokens = common_tokenize(vocab, encoder_text, false, true);
    result.encoder_truncated = result.encoder_tokens.size() > ENCODER_MAX_TOKENS;
    if (result.encoder_truncated) {
        result.encoder_tokens.resize(ENCODER_MAX_TOKENS);
    }

    auto query_tokens = common_tokenize(vocab, row.query, false, true);
    result.query_truncated = query_tokens.size() > QUERY_MAX_TOKENS;
    if (result.query_truncated) {
        query_tokens.resize(QUERY_MAX_TOKENS);
    }
    const std::string query = common_detokenize(vocab, query_tokens, true);
    const std::string decoder_text =
        std::string("<bos><start_of_turn>user\n") +
        SYSTEM_INSTRUCTION + "\n\n" +
        "<Instruct>: " + row.instruction + "\n" +
        "<Query>: " + query + "<end_of_turn>\n" +
        "<start_of_turn>model\n\n\n\n";
    result.decoder_tokens = common_tokenize(vocab, decoder_text, false, true);

    if (result.encoder_tokens.empty() || result.decoder_tokens.empty()) {
        throw std::runtime_error("tokenized prompt is empty");
    }
    if (contains_pad(result.encoder_tokens) || contains_pad(result.decoder_tokens)) {
        throw std::runtime_error("input text tokenized to reserved PAD token id 0");
    }
    if (result.decoder_tokens.front() != BOS_TOKEN_ID) {
        throw std::runtime_error("decoder prompt does not begin with explicit BOS token id 2");
    }

    const size_t decoder_padded = (result.decoder_tokens.size() + 7)/8*8;
    if (result.encoder_tokens.size() > n_ctx || decoder_padded > n_ctx) {
        throw std::runtime_error(
            "complete encoder/decoder sequence exceeds --ctx-size " + std::to_string(n_ctx));
    }
    return result;
}

batch_owner make_batch(
        const std::vector<llama_token> & tokens,
        int32_t output_index) {
    batch_owner owner((int32_t) tokens.size());
    for (int32_t i = 0; i < (int32_t) tokens.size(); ++i) {
        owner.batch.token[i]     = tokens[i];
        owner.batch.pos[i]       = i;
        owner.batch.n_seq_id[i]  = 1;
        owner.batch.seq_id[i][0] = 0;
        owner.batch.logits[i]    = output_index < 0 || i == output_index;
    }
    owner.batch.n_tokens = (int32_t) tokens.size();
    return owner;
}

double stable_sigmoid(double value) {
    if (value >= 0.0) {
        const double scale = std::exp(-value);
        return 1.0/(1.0 + scale);
    }
    const double scale = std::exp(value);
    return scale/(1.0 + scale);
}

json score_one(
        llama_context * context,
        const input_row & row,
        const prepared_input & input,
        bool dump_tokens) {
    llama_memory_clear(llama_get_memory(context), true);

    auto encoder = make_batch(input.encoder_tokens, -1);
    if (llama_encode(context, encoder.batch) != 0) {
        throw std::runtime_error("llama_encode failed");
    }

    const int32_t scoring_index = (int32_t) input.decoder_tokens.size() - 1;
    auto decoder = make_batch(input.decoder_tokens, scoring_index);
    if (llama_decode(context, decoder.batch) != 0) {
        throw std::runtime_error("llama_decode failed");
    }
    const float * logits = llama_get_logits_ith(context, scoring_index);
    if (logits == nullptr) {
        throw std::runtime_error("scoring logits are unavailable");
    }

    const double yes_logit = logits[YES_TOKEN_ID];
    const double no_logit = logits[NO_TOKEN_ID];
    const double margin = yes_logit - no_logit;
    json output = {
        {"schema_version", 1},
        {"case_id", row.case_id},
        {"yes_token_id", YES_TOKEN_ID},
        {"no_token_id", NO_TOKEN_ID},
        {"yes_logit", yes_logit},
        {"no_logit", no_logit},
        {"margin", margin},
        {"score", stable_sigmoid(margin)},
        {"encoder_tokens", input.encoder_tokens.size()},
        {"pooled_encoder_tokens", (input.encoder_tokens.size() + ENCODER_CHUNK_SIZE - 1) / ENCODER_CHUNK_SIZE},
        {"decoder_tokens", input.decoder_tokens.size()},
        {"encoder_truncated", input.encoder_truncated},
        {"query_truncated", input.query_truncated},
    };
    if (row.has_ids) {
        output["qid"] = row.qid;
        output["pid"] = row.pid;
    }
    if (dump_tokens) {
        output["encoder_input_ids"] = input.encoder_tokens;
        output["decoder_input_ids"] = input.decoder_tokens;
    }
    return output;
}

bool gpu_backend_available() {
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(index)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return true;
        }
    }
    return false;
}

void validate_gguf_architecture(const std::string & path) {
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * metadata = gguf_init_from_file(path.c_str(), params);
    if (metadata == nullptr) {
        throw std::runtime_error("failed to read GGUF metadata: " + path);
    }
    const int64_t key = gguf_find_key(metadata, "general.architecture");
    const bool valid =
        key >= 0 &&
        gguf_get_kv_type(metadata, key) == GGUF_TYPE_STRING &&
        std::strcmp(gguf_get_val_str(metadata, key), "t5gemma2") == 0;
    const std::string actual =
        key >= 0 && gguf_get_kv_type(metadata, key) == GGUF_TYPE_STRING
            ? gguf_get_val_str(metadata, key)
            : "<missing-or-non-string>";
    gguf_free(metadata);
    if (!valid) {
        throw std::runtime_error("expected general.architecture=t5gemma2, got '" + actual + "'");
    }
}

} // namespace

int main(int argc, char ** argv) {
    std::filesystem::path temporary_output;
    try {
        const options opts = parse_options(argc, argv);
        std::vector<input_row> rows;
        if (!opts.jsonl.empty()) {
            rows = read_jsonl(opts.jsonl);
        } else {
            rows.push_back(input_row{
                "single", "", "", opts.query, opts.passage, opts.instruction, 0, false,
            });
        }

        validate_gguf_architecture(opts.model);
        ggml_backend_load_all();
        if (opts.require_gpu && !gpu_backend_available()) {
            throw std::runtime_error("--require-gpu requested but no GPU backend is available");
        }

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = opts.n_gpu_layers;
        model_params.check_tensors = true;
        std::unique_ptr<llama_model, model_deleter> model(
            llama_model_load_from_file(opts.model.c_str(), model_params));
        if (!model) {
            throw std::runtime_error("failed to load model: " + opts.model);
        }
        validate_model(model.get());
        const llama_vocab * vocab = llama_model_get_vocab(model.get());

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = opts.n_ctx;
        context_params.n_batch = opts.n_ctx;
        context_params.n_ubatch = opts.n_ctx;
        context_params.n_seq_max = 1;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context_params.encoder_chunk_size = ENCODER_CHUNK_SIZE;
        context_params.no_perf = true;
        if (opts.n_threads > 0) {
            context_params.n_threads = opts.n_threads;
            context_params.n_threads_batch = opts.n_threads;
        }
        std::unique_ptr<llama_context, context_deleter> context(
            llama_init_from_model(model.get(), context_params));
        if (!context) {
            throw std::runtime_error("failed to create KaLM reranker context");
        }

        std::ofstream file_output;
        std::ostream * output = &std::cout;
        if (!opts.output.empty()) {
            const std::filesystem::path final_path = opts.output;
            if (!final_path.parent_path().empty()) {
                std::filesystem::create_directories(final_path.parent_path());
            }
            const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
            temporary_output = final_path.string() + ".tmp." + std::to_string(stamp);
            file_output.open(temporary_output);
            if (!file_output) {
                throw std::runtime_error("failed to open temporary output: " + temporary_output.string());
            }
            output = &file_output;
        }

        const auto started = std::chrono::steady_clock::now();
        for (const auto & row : rows) {
            try {
                const auto prepared = prepare_input(vocab, row, opts.n_ctx);
                *output << score_one(context.get(), row, prepared, opts.dump_tokens).dump() << '\n';
            } catch (const std::exception & error) {
                const std::string location = row.line > 0
                    ? opts.jsonl + ":" + std::to_string(row.line) + " (" + row.case_id + ")"
                    : row.case_id;
                throw std::runtime_error(location + ": " + error.what());
            }
        }
        output->flush();
        if (!*output) {
            throw std::runtime_error("failed to write scoring output");
        }

        if (!opts.output.empty()) {
            file_output.close();
            std::filesystem::rename(temporary_output, opts.output);
            temporary_output.clear();
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::fprintf(
            stderr,
            "scored %zu pair(s) in %.3f s (%.3f pairs/s)\n",
            rows.size(), seconds, seconds > 0.0 ? rows.size()/seconds : 0.0);
        return EXIT_SUCCESS;
    } catch (const std::invalid_argument & error) {
        if (!temporary_output.empty()) {
            std::error_code ignored;
            std::filesystem::remove(temporary_output, ignored);
        }
        std::fprintf(stderr, "input error: %s\n", error.what());
        print_usage(argv[0]);
        return 2;
    } catch (const std::exception & error) {
        if (!temporary_output.empty()) {
            std::error_code ignored;
            std::filesystem::remove(temporary_output, ignored);
        }
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
