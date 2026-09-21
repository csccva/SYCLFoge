cmake_minimum_required(VERSION 3.20)

project(tinyml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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