
#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "tinyml/tensor.hpp"
#include "tinyml/transformer_block.hpp"
#include "tinyml/gpt2_weights.hpp"

namespace tinyml {

inline void load_gpt2_block_weights(sycl::queue& queue, TransformerBlock<float>& block, const std::string& directory) {
    auto load = [&](Tensor<float>& tensor, const std::string& filename, const std::vector<std::size_t>& expected_shape) {
        if (tensor.shape() != expected_shape) throw std::invalid_argument("Unexpected tensor shape for GPT-2 weight: " + filename);
        const std::string path = directory + "/" + filename;
        auto weights = read_gpt2_weights(path, tensor.size());
        queue.memcpy(tensor.data(), weights.data(), weights.size() * sizeof(float)).wait_and_throw();
    };

    load(block.ln1_gamma(), "ln1_gamma.bin", {768});
    load(block.ln1_beta(), "ln1_beta.bin", {768});

    load(block.q_proj().weight(), "q_weight.bin", {768, 768});
    load(block.q_proj().bias(), "q_bias.bin", {768});
    load(block.k_proj().weight(), "k_weight.bin", {768, 768});
    load(block.k_proj().bias(), "k_bias.bin", {768});
    load(block.v_proj().weight(), "v_weight.bin", {768, 768});
    load(block.v_proj().bias(), "v_bias.bin", {768});

    load(block.o_proj().weight(), "attn_out_weight.bin", {768, 768});
    load(block.o_proj().bias(), "attn_out_bias.bin", {768});

    load(block.ln2_gamma(), "ln2_gamma.bin", {768});
    load(block.ln2_beta(), "ln2_beta.bin", {768});

    load(block.ff1().weight(), "ff1_weight.bin", {768, 3072});
    load(block.ff1().bias(), "ff1_bias.bin", {3072});
    load(block.ff2().weight(), "ff2_weight.bin", {3072, 768});
    load(block.ff2().bias(), "ff2_bias.bin", {768});
}

}
