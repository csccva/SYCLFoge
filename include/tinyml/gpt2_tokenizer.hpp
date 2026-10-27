
#pragma once

#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "tokenizers_cpp.h"

namespace tinyml {

class GPT2Tokenizer {
public:
    explicit GPT2Tokenizer(const std::string& tokenizer_path)
    {
        std::ifstream file(tokenizer_path, std::ios::binary);
        if (!file) {
            throw std::runtime_error(
                "Cannot open GPT-2 tokenizer: " + tokenizer_path);
        }

        const std::string blob{
            std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>()
        };

        tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(blob);

        if (!tokenizer_) {
            throw std::runtime_error(
                "Failed to load GPT-2 tokenizer from: " + tokenizer_path);
        }
    }

    std::vector<std::int32_t> encode(const std::string& text) const
    {
        return tokenizer_->Encode(text);
    }

    std::string decode(const std::vector<std::int32_t>& ids) const
    {
        return tokenizer_->Decode(ids);
    }

    std::size_t vocab_size() const
    {
        return tokenizer_->GetVocabSize();
    }

private:
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};

} // namespace tinyml
