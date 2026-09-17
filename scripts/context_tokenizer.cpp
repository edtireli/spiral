// Vocabulary-only batch measurement. No weights, KV cache, context or generation.
// Input: repeated decimal UTF-8 byte lengths + newline + exactly that many bytes.
// Output: one JSON count per input, then a terminal receipt. No source echo.
#include "llama.h"
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

static void quiet(enum ggml_log_level, const char *, void *) {}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    llama_log_set(quiet, nullptr);
    auto options = llama_model_default_params();
    options.vocab_only = true;
    auto *model = llama_model_load_from_file(argv[1], options);
    if (!model) return 3;
    const auto *vocabulary = llama_model_get_vocab(model);
    std::string line;
    std::size_t total = 0, index = 0;
    int status = 0;
    while (std::getline(std::cin, line)) {
        if (line.empty() || line.size() > 10 || line.find_first_not_of("0123456789") != std::string::npos) {
            status = 4; break;
        }
        std::size_t size;
        try { size = std::stoull(line); } catch (...) { status = 4; break; }
        if (size > 2 * 1024 * 1024 || total + size > 32 * 1024 * 1024 || index >= 4096) {
            status = 5; break;
        }
        std::string text(size, '\0');
        if (!std::cin.read(text.data(), static_cast<std::streamsize>(size))) { status = 6; break; }
        total += size;
        const int32_t count = llama_tokenize(vocabulary, text.data(), static_cast<int32_t>(size),
                                              nullptr, 0, false, false);
        if (count > 0 || count == std::numeric_limits<int32_t>::min()) { status = 7; break; }
        std::cout << "{\"index\":" << index++ << ",\"tokens\":" << -count << "}\n";
    }
    if (status == 0 && !std::cin.eof()) status = 6;
    if (status == 0) {
        std::cout << "{\"schema\":\"spiral.tokenizer.v1\",\"scope\":\"raw_text_no_bos_no_special\",\"count\":"
                  << index << "}\n";
    }
    llama_model_free(model);
    return status;
}
