
#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tinyml/gpt2_embedding.hpp"

namespace tinyml {

inline std::vector<float> read_gpt2_weights(const std::string& path, std::size_t expected_count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open GPT-2 weights: " + path);

    const auto bytes = file.tellg();
    const auto expected_bytes = static_cast<std::streamoff>(expected_count * sizeof(float));
    if (bytes != expected_bytes) throw std::runtime_error("GPT-2 weight file size mismatch: " + path);

    file.seekg(0);
    std::vector<float> weights(expected_count);
    if (!file.read(reinterpret_cast<char*>(weights.data()), expected_bytes)) throw std::runtime_error("Failed to read GPT-2 weights: " + path);

    return weights;
}

inline void load_gpt2_weights(sycl::queue& queue, GPT2Embedding<float>& embedding, const std::string& token_path, const std::string& position_path) {
    if (embedding.token_weights().shape() != std::vector<std::size_t>{50257, 768}) throw std::invalid_argument("Unexpected GPT-2 token embedding shape");
    if (embedding.position_weights().shape() != std::vector<std::size_t>{1024, 768}) throw std::invalid_argument("Unexpected GPT-2 positional embedding shape");

    auto token_host = read_gpt2_weights(token_path, 50257 * 768);
    auto position_host = read_gpt2_weights(position_path, 1024 * 768);

    queue.memcpy(embedding.token_weights().data(), token_host.data(), token_host.size() * sizeof(float)).wait();
    queue.memcpy(embedding.position_weights().data(), position_host.data(), position_host.size() * sizeof(float)).wait();
}

} // namespace tinyml
