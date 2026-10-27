#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "tinyml/tensor.hpp"

namespace tinyml {

template <typename T>
class Embedding {
public:
    Embedding(sycl::queue& queue, std::size_t vocab_size, std::size_t embedding_dim)
        : queue_(queue), vocab_size_(vocab_size), embedding_dim_(embedding_dim), weights_(queue, {vocab_size, embedding_dim})
    {
        if (vocab_size == 0 || embedding_dim == 0) {
            throw std::invalid_argument("Invalid embedding dimensions");
        }
    }

    Tensor<T>& weights()
    {
        return weights_;
    }

    sycl::event forward(const Tensor<std::int32_t>& token_ids, Tensor<T>& output, std::size_t group_size = 256, const std::vector<sycl::event>& dependencies = {})
    {
        const auto ids_shape = token_ids.shape();
        const auto out_shape = output.shape();

        if (ids_shape.size() != 2 || out_shape.size() != 3) {
            throw std::invalid_argument("Embedding expects token IDs [B,L] and output [B,L,D]");
        }

        const std::size_t B = ids_shape[0];
        const std::size_t L = ids_shape[1];

        if (out_shape[0] != B || out_shape[1] != L || out_shape[2] != embedding_dim_ || token_ids.size() != B * L) {
            throw std::invalid_argument("Embedding tensor shape mismatch");
        }

        if (group_size == 0) {
            throw std::invalid_argument("Embedding group_size must be greater than zero");
        }

        const std::size_t total = B * L * embedding_dim_;
        const std::size_t embedding_dim = embedding_dim_;
        const std::size_t vocab_size = vocab_size_;
        const std::size_t global_size = ((total + group_size - 1) / group_size) * group_size;

        const auto* ids = token_ids.data();
        const auto* weights = weights_.data();
        auto* out = output.data();

        return queue_.submit([&](sycl::handler& h) {
            h.depends_on(dependencies);

            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(group_size)), [=](sycl::nd_item<1> item) {
                const std::size_t i = item.get_global_linear_id();

                if (i >= total) {
                    return;
                }

                const std::size_t token_position = i / embedding_dim;
                const std::size_t feature = i % embedding_dim;
                const std::int32_t token_id = ids[token_position];

                if (token_id >= 0 && static_cast<std::size_t>(token_id) < vocab_size) {
                    out[i] = weights[static_cast<std::size_t>(token_id) * embedding_dim + feature];
                } else {
                    out[i] = T{0};
                }
            });
        });
    }

private:
    sycl::queue& queue_;
    std::size_t vocab_size_;
    std::size_t embedding_dim_;
    Tensor<T> weights_;
};

} // namespace tinyml
