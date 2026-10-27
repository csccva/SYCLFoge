cmake_minimum_required(VERSION 3.18)
project(test_gpt2_tokenizer C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(
    ../external/tokenizers-cpp
    tokenizers
    EXCLUDE_FROM_ALL
)

add_executable(test_gpt2_tokenizer test_gpt2_tokenizer.cpp)

target_link_libraries(test_gpt2_tokenizer PRIVATE tokenizers_cpp)
target_include_directories(test_gpt2_tokenizer PRIVATE ../include)