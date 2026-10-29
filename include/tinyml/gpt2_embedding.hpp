
#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "tinyml/tensor.hpp"
#include "tinyml/embedding.hpp"
#include "tinyml/positional_embedding.hpp"

namespace tinyml {

template <typename T>
class GPT2Embedding {
public:
    GPT2Embedding(sycl::queue& queue, std::size_t vocab_size, std::size_t max_seq_len, std::size_t embedding_dim)
        : queue_(queue), max_seq_len_(max_seq_len), embedding_dim_(embedding_dim), token_embedding_(queue, vocab_size, embedding_dim), positional_embedding_(queue, max_seq_len, embedding_dim)
    {
        if (max_seq_len == 0 || vocab_size == 0 || embedding_dim == 0) {
            throw std::invalid_argument("GPT2Embedding dimensions must be greater than zero");
        }
    }

    Tensor<T>& token_weights() { return token_embedding_.weights(); }
    Tensor<T>& position_weights() { return positional_embedding_.weights(); }

    sycl::event forward(const Tensor<std::int32_t>& token_ids, Tensor<T>& token_vectors, Tensor<T>& output, std::size_t group_size = 256)
    {
        const auto ids_shape = token_ids.shape();
        const auto vectors_shape = token_vectors.shape();
        const auto out_shape = output.shape();

        if (ids_shape.size() != 2 || vectors_shape.size() != 3 || out_shape.size() != 3) {
            throw std::invalid_argument("GPT2Embedding tensor rank mismatch");
        }

        const std::size_t B = ids_shape[0];
        const std::size_t L = ids_shape[1];

        if (B == 0 || L == 0) {
            throw std::invalid_argument("GPT2Embedding received empty token IDs");
        }

        if (L > max_seq_len_) {
            throw std::invalid_argument("GPT2Embedding sequence exceeds max_seq_len");
        }

        if (vectors_shape != out_shape || out_shape[0] != B || out_shape[1] != L || out_shape[2] != embedding_dim_) {
            throw std::invalid_argument("GPT2Embedding tensor shape mismatch");
        }

        if (group_size == 0) {
            throw std::invalid_argument("GPT2Embedding group_size must be greater than zero");
        }

        auto token_event = token_embedding_.forward(token_ids, token_vectors, group_size);
        return positional_embedding_.forward(token_vectors, output, group_size, {token_event});
    }

private:
    sycl::queue& queue_;
    std::size_t max_seq_len_;
    std::size_t embedding_dim_;
    Embedding<T> token_embedding_;
    PositionalEmbedding<T> positional_embedding_;
};

} // namespace tinyml
