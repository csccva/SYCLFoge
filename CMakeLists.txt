cmake_minimum_required(VERSION 3.20)

project(tinyml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(external/tokenizers-cpp EXCLUDE_FROM_ALL)

option(TINYML_BUILD_TESTS "Build tinyml tests" OFF)

add_executable(tinyml
    src/main.cpp
)

target_include_directories(tinyml PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(tinyml PRIVATE
    -fsycl
    -fsycl-targets=spir64
)

target_link_options(tinyml PRIVATE
    -fsycl
    -fsycl-targets=spir64
)

if(TINYML_BUILD_TESTS)
    add_executable(test_core tests/test_core.cpp)
    target_include_directories(test_core PRIVATE include)
    target_compile_options(test_core PRIVATE -fsycl -fsycl-targets=spir64)
    target_link_options(test_core PRIVATE -fsycl -fsycl-targets=spir64)

    add_executable(test_nn tests/test_nn.cpp)
    target_include_directories(test_nn PRIVATE include)
    target_compile_options(test_nn PRIVATE -fsycl -fsycl-targets=spir64)
    target_link_options(test_nn PRIVATE -fsycl -fsycl-targets=spir64)

    add_executable(test_sdpa tests/test_sdpa.cpp)
    target_include_directories(test_sdpa PRIVATE include)
    target_compile_options(test_sdpa PRIVATE -fsycl -fsycl-targets=spir64)
    target_link_options(test_sdpa PRIVATE -fsycl -fsycl-targets=spir64)

    add_executable(test_attention tests/test_attention.cpp)
    target_include_directories(test_attention PRIVATE include)
    target_compile_options(test_attention PRIVATE -fsycl -fsycl-targets=spir64)
    target_link_options(test_attention PRIVATE -fsycl -fsycl-targets=spir64)

    add_executable(benchmark_ops tests/benchmark_ops.cpp)
    target_include_directories(benchmark_ops PRIVATE include)
    target_compile_options(benchmark_ops PRIVATE -fsycl -fsycl-targets=spir64)
    target_link_options(benchmark_ops PRIVATE -fsycl -fsycl-targets=spir64)

    add_executable(test_flash_attention tests/test_flash_attention.cpp)
    target_include_directories(test_flash_attention PRIVATE include)
    target_compile_options(test_flash_attention PRIVATE -fsycl -fsycl-targets=spir64)
    target_link_options(test_flash_attention PRIVATE -fsycl -fsycl-targets=spir64)

    add_executable(benchmark_flash_attention
        tests/benchmark_flash_attention.cpp
    )
    target_include_directories(benchmark_flash_attention PRIVATE include)
    target_compile_options(benchmark_flash_attention PRIVATE -fsycl -fsycl-targets=spir64)
    target_link_options(benchmark_flash_attention PRIVATE -fsycl -fsycl-targets=spir64)

    add_executable(test_transformer_block
        tests/test_transformer_block.cpp
    )
    target_include_directories(test_transformer_block PRIVATE include)
    target_compile_options(test_transformer_block PRIVATE -fsycl -fsycl-targets=spir64)
    target_link_options(test_transformer_block PRIVATE -fsycl -fsycl-targets=spir64)

    add_executable(test_embedding tests/test_embedding.cpp)
    target_include_directories(test_embedding PRIVATE include)
    target_compile_options(test_embedding PRIVATE -fsycl -fsycl-targets=spir64)
    target_link_options(test_embedding PRIVATE -fsycl -fsycl-targets=spir64)

    add_executable(test_positional_embedding
        tests/test_positional_embedding.cpp
    )
    target_include_directories(test_positional_embedding PRIVATE include)
    target_compile_options(test_positional_embedding PRIVATE -fsycl -fsycl-targets=spir64)
    target_link_options(test_positional_embedding PRIVATE -fsycl -fsycl-targets=spir64)

    add_executable(test_gpt2_embedding tests/test_gpt2_embedding.cpp)
    target_include_directories(test_gpt2_embedding PRIVATE include)
    target_compile_options(test_gpt2_embedding PRIVATE -fsycl -fsycl-targets=spir64)
    target_link_options(test_gpt2_embedding PRIVATE -fsycl -fsycl-targets=spir64)

    add_executable(test_gpt2_embedding_validation
        tests/test_gpt2_embedding_validation.cpp
    )
    target_include_directories(test_gpt2_embedding_validation PRIVATE include)
    target_compile_options(test_gpt2_embedding_validation PRIVATE -fsycl -fsycl-targets=spir64)
    target_link_options(test_gpt2_embedding_validation PRIVATE -fsycl -fsycl-targets=spir64)
endif()

option(TINYML_BUILD_GPT2_WEIGHTS_TEST
       "Build the GPT-2 pretrained weights loading test" OFF)

if(TINYML_BUILD_GPT2_WEIGHTS_TEST)
    add_executable(test_gpt2_weights tests/test_gpt2_weights.cpp)
    target_include_directories(test_gpt2_weights PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_options(test_gpt2_weights PRIVATE -fsycl -fsycl-targets=spir64)
    target_link_options(test_gpt2_weights PRIVATE -fsycl -fsycl-targets=spir64)
endif()

option(TINYML_BUILD_GPT2_BLOCK_TEST
       "Build GPT-2 Transformer block test" ON)

if(TINYML_BUILD_GPT2_BLOCK_TEST)
    add_executable(test_gpt2_block tests/test_gpt2_block.cpp)
    target_include_directories(test_gpt2_block PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_options(test_gpt2_block PRIVATE -fsycl)
    target_link_options(test_gpt2_block PRIVATE -fsycl)
endif()

option(TINYML_BUILD_GPT2_MODEL_TEST
       "Build GPT-2 full model test" ON)

if(TINYML_BUILD_GPT2_MODEL_TEST)
    add_executable(test_gpt2_model tests/test_gpt2_model.cpp)
    target_include_directories(test_gpt2_model PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_options(test_gpt2_model PRIVATE -fsycl -fsycl-targets=spir64)
    target_link_options(test_gpt2_model PRIVATE -fsycl -fsycl-targets=spir64)
endif()

option(TINYML_BUILD_GPT2_GENERATE
       "Build GPT-2 text generation executable" ON)

if(TINYML_BUILD_GPT2_GENERATE)
    add_executable(gpt2_generate src/gpt2_generate.cpp)

    target_include_directories(gpt2_generate PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/tokenizers-cpp/include
    )

    target_compile_options(gpt2_generate PRIVATE
        -fsycl
        -fsycl-targets=spir64
    )

    target_link_options(gpt2_generate PRIVATE
        -fsycl
        -fsycl-targets=spir64
    )

    target_link_libraries(gpt2_generate PRIVATE tokenizers_cpp)
endif()
option(TINYML_BUILD_GPT2_DUMP_LOGITS
       "Build GPT-2 SYCL logits diagnostic" ON)

if(TINYML_BUILD_GPT2_DUMP_LOGITS)
    add_executable(gpt2_dump_logits src/gpt2_dump_logits.cpp)

    target_include_directories(gpt2_dump_logits PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_compile_options(gpt2_dump_logits PRIVATE
        -fsycl -fsycl-targets=spir64
    )

    target_link_options(gpt2_dump_logits PRIVATE
        -fsycl -fsycl-targets=spir64
    )
endif()
