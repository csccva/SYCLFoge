
#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tinyml/tensor.hpp"
#include "tinyml/gpt2_embedding.hpp"
#include "tinyml/gpt2_layer_norm.hpp"
#include "tinyml/transformer_block.hpp"
#include "tinyml/gpt2_block_weights.hpp"
#include "tinyml/gpt2_weights.hpp"
#include "tinyml/ops.hpp"

namespace tinyml {

class GPT2Model {
public:
    static constexpr std::size_t VOCAB_SIZE = 50257;
    static constexpr std::size_t MAX_SEQ_LEN = 1024;
    static constexpr std::size_t D_MODEL = 768;
    static constexpr std::size_t N_HEADS = 12;
    static constexpr std::size_t D_FF = 3072;
    static constexpr std::size_t N_LAYERS = 12;
    static constexpr std::size_t GROUP_SIZE = 256;
    static constexpr std::size_t TILE_SIZE = 16;

    explicit GPT2Model(sycl::queue& queue, const std::string& model_dir = "models/gpt2")
        : queue_(queue),
          model_dir_(model_dir),
          embedding_(queue, VOCAB_SIZE, MAX_SEQ_LEN, D_MODEL),
          final_ln_(queue, D_MODEL) {
        load_gpt2_weights(
            queue_,
            embedding_,
            model_dir_ + "/token_embeddings.bin",
            model_dir_ + "/position_embeddings.bin");

        auto gamma = read_gpt2_weights(model_dir_ + "/final_ln_gamma.bin", D_MODEL);
        auto beta = read_gpt2_weights(model_dir_ + "/final_ln_beta.bin", D_MODEL);

        queue_.memcpy(final_ln_.gamma().data(), gamma.data(), D_MODEL * sizeof(float)).wait_and_throw();
        queue_.memcpy(final_ln_.beta().data(), beta.data(), D_MODEL * sizeof(float)).wait_and_throw();

        blocks_.reserve(N_LAYERS);

        for (std::size_t i = 0; i < N_LAYERS; ++i) {
            auto block = std::make_unique<TransformerBlock<float>>(queue_, D_MODEL, N_HEADS, D_FF);
            load_gpt2_block_weights(queue_, *block, model_dir_ + "/block" + std::to_string(i));
            blocks_.push_back(std::move(block));
        }
    }

    sycl::event forward(const Tensor<std::int32_t>& token_ids, Tensor<float>& logits) {
        const auto ids_shape = token_ids.shape();

        if (ids_shape.size() != 2 || ids_shape[0] == 0 || ids_shape[1] == 0) {
            throw std::invalid_argument("GPT2Model expects non-empty token IDs with shape [B,L]");
        }

        const std::size_t B = ids_shape[0];
        const std::size_t L = ids_shape[1];

        if (L > MAX_SEQ_LEN) {
            throw std::invalid_argument("GPT2Model sequence length exceeds 1024");
        }

        if (logits.shape() != std::vector<std::size_t>{B, L, VOCAB_SIZE}) {
            throw std::invalid_argument("GPT2Model logits must have shape [B,L,50257]");
        }

        // The forward API has no dependency parameter for the token-ID producer.
        // Wait for earlier work on this queue before starting the embedding lookup.
        queue_.wait_and_throw();

        Tensor<float> token_vectors(queue_, {B, L, D_MODEL});
        Tensor<float> embedded(queue_, {B, L, D_MODEL});

        // Produce token embeddings plus positional embeddings.
        auto e_embedding = embedding_.forward(token_ids, token_vectors, embedded, GROUP_SIZE);
        e_embedding.wait_and_throw();

        Tensor<float> buffer(queue_, {B, L, D_MODEL});
        Tensor<float> X_norm(queue_, {B, L, D_MODEL});
        Tensor<float> Q(queue_, {B, L, D_MODEL});
        Tensor<float> K(queue_, {B, L, D_MODEL});
        Tensor<float> V(queue_, {B, L, D_MODEL});
        Tensor<float> A(queue_, {B, L, D_MODEL});
        Tensor<float> R(queue_, {B, L, D_MODEL});
        Tensor<float> R_norm(queue_, {B, L, D_MODEL});
        Tensor<float> F1(queue_, {B, L, D_FF});
        Tensor<float> F1_gelu(queue_, {B, L, D_FF});
        Tensor<float> F2(queue_, {B, L, D_MODEL});
        Tensor<float> final_hidden(queue_, {B, L, D_MODEL});

        Tensor<float>* current = &embedded;
        Tensor<float>* next = &buffer;

        // Execute all 12 pretrained Transformer blocks in sequence.
        for (std::size_t i = 0; i < N_LAYERS; ++i) {
            auto e_block = blocks_[i]->forward(
                *current,
                *next,
                X_norm,
                Q,
                K,
                V,
                A,
                R,
                R_norm,
                F1,
                F1_gelu,
                F2,
                GROUP_SIZE,
                TILE_SIZE);

            e_block.wait_and_throw();
            std::swap(current, next);
        }

        // GPT-2 applies a final LayerNorm after the last Transformer block.
        auto e_norm = final_ln_.forward(*current, final_hidden, GROUP_SIZE);

        // The language-model head shares weights with the token embedding matrix.
        // final_hidden is [B*L, 768], and the stored embedding matrix is [50257, 768].
        const std::size_t rows = B * L;

        auto e_logits = ops::matmul_tiled_transposed(
            queue_,
            final_hidden.data(),
            embedding_.token_weights().data(),
            logits.data(),
            rows,
            D_MODEL,
            VOCAB_SIZE,
            TILE_SIZE,
            {e_norm});

        e_logits.wait_and_throw();
        return e_logits;
    }

    GPT2Embedding<float>& embedding() {
        return embedding_;
    }

private:
    sycl::queue& queue_;
    std::string model_dir_;
    GPT2Embedding<float> embedding_;
    GPT2LayerNorm<float> final_ln_;
    std::vector<std::unique_ptr<TransformerBlock<float>>> blocks_;
};

}
