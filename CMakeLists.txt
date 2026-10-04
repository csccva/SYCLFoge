cmake_minimum_required(VERSION 3.20)

project(tinyml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TINYML_BUILD_TESTS "Build tinyml tests" OFF)

add_executable(tinyml
    src/main.cpp
)

target_include_directories(tinyml PRIVATE
    include
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
    add_executable(test_core
        tests/test_core.cpp
    )

    target_include_directories(test_core PRIVATE
        include
    )

    target_compile_options(test_core PRIVATE
        -fsycl
        -fsycl-targets=spir64
    )

    target_link_options(test_core PRIVATE
        -fsycl
        -fsycl-targets=spir64
    )
        add_executable(test_nn
        tests/test_nn.cpp
    )

    target_include_directories(test_nn PRIVATE
        include
    )

    target_compile_options(test_nn PRIVATE
        -fsycl
        -fsycl-targets=spir64
    )

    target_link_options(test_nn PRIVATE
        -fsycl
        -fsycl-targets=spir64
    )
        add_executable(test_sdpa
        tests/test_sdpa.cpp
    )

    target_include_directories(test_sdpa PRIVATE
        include
    )

    target_compile_options(test_sdpa PRIVATE
        -fsycl
        -fsycl-targets=spir64
    )

    target_link_options(test_sdpa PRIVATE
        -fsycl
        -fsycl-targets=spir64
    )

        add_executable(test_attention
        tests/test_attention.cpp
    )

    target_include_directories(test_attention PRIVATE
        include
    )
    target_compile_options(test_attention PRIVATE
        -fsycl
        -fsycl-targets=spir64
    )

    target_link_options(test_attention PRIVATE
        -fsycl
        -fsycl-targets=spir64
    )
        add_executable(benchmark_ops
        tests/benchmark_ops.cpp
    )

    target_include_directories(benchmark_ops PRIVATE
        include
    )

    target_compile_options(benchmark_ops PRIVATE
        -fsycl
        -fsycl-targets=spir64
    )

    target_link_options(benchmark_ops PRIVATE
        -fsycl
        -fsycl-targets=spir64
    )
endif()