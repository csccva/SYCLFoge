#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include "tinyml/tensor.hpp"
#include "tinyml/ops.hpp"

namespace tinyml {

template <typename T>
class GPT2LayerNorm {
public:
    GPT2LayerNorm(sycl::queue& queue, std::size_t embedding_dim, T epsilon = static_cast<T>(1.0e-5)): queue_(queue), embedding_dim_(embedding_dim), epsilon_(epsilon), gamma_(queue, {embedding_dim}), beta_(queue, {embedding_dim}) {
        if (embedding_dim == 0) throw std::invalid_argument("GPT2LayerNorm embedding_dim must be greater than zero");
        if (epsilon <= static_cast<T>(0)) throw std::invalid_argument("GPT2LayerNorm epsilon must be greater than zero");
        queue_.fill(gamma_.data(), static_cast<T>(1), embedding_dim_).wait();
        queue_.fill(beta_.data(), static_cast<T>(0), embedding_dim_).wait();
    }

    Tensor<T>& gamma() { return gamma_; }
    Tensor<T>& beta() { return beta_; }
    const Tensor<T>& gamma() const { return gamma_; }
    const Tensor<T>& beta() const { return beta_; }

    sycl::event forward(const Tensor<T>& input, Tensor<T>& output, std::size_t group_size = 256, const std::vector<sycl::event>& dependencies = {}) {
        const auto shape = input.shape();
        if (shape.size() != 3 || output.shape() != shape) throw std::invalid_argument("GPT2LayerNorm expects matching rank-3 tensors [B,L,D]");
        if (shape[0] == 0 || shape[1] == 0 || shape[2] != embedding_dim_) throw std::invalid_argument("GPT2LayerNorm input dimensions are invalid");
        if (group_size == 0 || (group_size & (group_size - 1)) != 0) throw std::invalid_argument("GPT2LayerNorm group_size must be a power of two");
        const std::size_t M = shape[0] * shape[1];
        const std::size_t N = embedding_dim_;
        return ops::layer_norm_parallel(queue_, input.data(), gamma_.data(), beta_.data(), output.data(), M, N, group_size, epsilon_, dependencies);
    }

private:
    sycl::queue& queue_;
    std::size_t embedding_dim_;
    T epsilon_;
    Tensor<T> gamma_;
    Tensor<T> beta_;
};

}
