#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "tinyml/tensor.hpp"

namespace tinyml {

    template <typename T>
    class PositionalEmbedding {
    public:
        PositionalEmbedding(sycl::queue& queue, std::size_t max_seq_len, std::size_t embedding_dim): queue_(queue), max_seq_len_(max_seq_len), embedding_dim_(embedding_dim), weights_(queue, {max_seq_len, embedding_dim})
        {
            if (max_seq_len == 0 || embedding_dim == 0) {
                throw std::invalid_argument("Invalid embedding dimensions");
            }
        }
        Tensor<T>& weights()
        {
            return weights_;
        }
        sycl::event forward(const Tensor<T>& input, Tensor<T>& output, std::size_t group_size = 256, const std::vector<sycl::event>& dependencies = {})
        {
            const auto input_shape = input.shape();
            const auto output_shape = output.shape();

            if (input_shape.size() != 3 || output_shape.size() != 3) {
                throw std::invalid_argument("PositionalEmbedding expects input and output shapes [B,L,D]");
            }

            const std::size_t B = input_shape[0];
            const std::size_t L = input_shape[1];
            const std::size_t D = input_shape[2];

            if (L > max_seq_len_ || D != embedding_dim_ || output_shape != input_shape) {
                throw std::invalid_argument("PositionalEmbedding tensor shape mismatch");
            }

            if (group_size == 0) {
                throw std::invalid_argument("PositionalEmbedding group_size must be greater than zero");
            }

            const std::size_t total = B * L * D;
            const std::size_t max_seq_len = max_seq_len_;
            const std::size_t embedding_dim = embedding_dim_;
            const std::size_t global_size = ((total + group_size - 1) / group_size) * group_size;

            const auto* in = input.data();
            const auto* weights = weights_.data();
            auto* out = output.data();

            return queue_.submit([&](sycl::handler& h) {
                h.depends_on(dependencies);

                h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(group_size)), [=](sycl::nd_item<1> item) {
                    const std::size_t i = item.get_global_linear_id();

                    if (i >= total) {
                        return;
                    }

                    const std::size_t position = (i / embedding_dim) % L;
                    const std::size_t feature = i % embedding_dim;

                    out[i] = in[i] + weights[position * embedding_dim + feature];
                });
            });
        }
    private:
        sycl::queue& queue_;
        std::size_t max_seq_len_;
        std::size_t embedding_dim_;
        Tensor<T> weights_;
    };

} // namespace tinyml
