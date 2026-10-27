// rm -rf build/; cmake -S . -B build   -DCMAKE_CXX_COMPILER=/home/cristian/intel/compiler/2025.0/bin/icpx   -DTINYML_BUILD_TESTS=ON; cmake --build build --target benchmark_flash_attention; ./build/benchmark_flash_attention
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>
#include <exception>
#include <stdexcept>

#include "tinyml/tensor.hpp"
#include "tinyml/ops.hpp"
#include "tinyml/flash_attention.hpp"

double event_ms(const sycl::event& e)
{
    const auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    return static_cast<double>(end - start) / 1.0e6;
}

template <typename Function>
double benchmark_ms(Function&& function, std::size_t warmup, std::size_t iterations, sycl::queue& queue)
{
    for (std::size_t i = 0; i < warmup; ++i) {
        auto events = function();
        for (auto& e : events) e.wait();
    }
    queue.wait();

    double total_ms = 0.0;
    for (std::size_t i = 0; i < iterations; ++i) {
        auto events = function();
        for (auto& e : events) e.wait();
        for (const auto& e : events) total_ms += event_ms(e);
    }
    return total_ms / static_cast<double>(iterations);
}

int main()
{
    sycl::queue queue{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";

    const std::vector<std::size_t> sizes = {128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384};
    const std::size_t D = 513, Dv = 513, tile_size = 32, softmax_group_size = 8, warmup = 5;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::cout << "\n=== Attention Scaling Benchmark ===\n";
    std::cout << "D=" << D << ", Dv=" << Dv << ", tile=" << tile_size << ", warm-up=" << warmup << "\n\n";
    std::cout << std::left << std::setw(8) << "Size" << std::setw(8) << "Iters" << std::setw(13) << "Naive ms" << std::setw(13) << "Online ms" << std::setw(13) << "MergePV ms" << std::setw(13) << "Old ms" << std::setw(13) << "OldNew ms" << std::setw(12) << "N/Online" << std::setw(12) << "N/MergePV" << std::setw(12) << "N/Old" << std::setw(12) << "N/OldNew" << "\n";

    for (const std::size_t size : sizes) {
        const std::size_t M = size, N = size;
        const std::size_t iterations = size <= 512 ? 50 : (size <= 1024 ? 30 : 10);
        std::cout << std::left << std::setw(8) << size << std::flush;

        try {
            std::vector<float> h_Q(M * D), h_K(N * D), h_V(N * Dv);
            for (auto& x : h_Q) x = dist(rng);
            for (auto& x : h_K) x = dist(rng);
            for (auto& x : h_V) x = dist(rng);

            tinyml::Tensor<float> Q(queue, {M, D}), K_tensor(queue, {N, D}), V(queue, {N, Dv});
            tinyml::Tensor<float> scores(queue, {M, N}), weights(queue, {M, N});
            tinyml::Tensor<float> output_naive(queue, {M, Dv}), output_online(queue, {M, Dv}), output_merge(queue, {M, Dv}), output_old(queue, {M, Dv}), output_old_new(queue, {M, Dv});

            auto e1 = ops::copy_to_device(queue, Q, h_Q.data());
            auto e2 = ops::copy_to_device(queue, K_tensor, h_K.data());
            auto e3 = ops::copy_to_device(queue, V, h_V.data());
            e1.wait_and_throw();
            e2.wait_and_throw();
            e3.wait_and_throw();

            const double naive_ms = benchmark_ms([&]() {
                auto qk = ops::matmul_tiled_transposed_scaled(queue, Q, K_tensor, scores, tile_size);
                auto sm = ops::softmax_parallel(queue, scores, weights, softmax_group_size, {qk});
                auto pv = ops::matmul_tiled(queue, weights, V, output_naive, tile_size, {sm});
                return std::vector<sycl::event>{qk, sm, pv};
            }, warmup, iterations, queue);

            const double online_ms = benchmark_ms([&]() {
                return std::vector<sycl::event>{ops::attention_scores_online(queue, Q, K_tensor, V, output_online, tile_size)};
            }, warmup, iterations, queue);

            const double merge_ms = benchmark_ms([&]() {
                auto qk = ops::matmul_tiled_transposed_scaled(queue, Q, K_tensor, scores, tile_size);
                auto merged = softmax_matmul_tiled(queue, scores.data(), V.data(), output_merge.data(), M, N, Dv, tile_size, {qk});
                return std::vector<sycl::event>{qk, merged};
            }, warmup, iterations, queue);

            const double old_ms = benchmark_ms([&]() {
                return std::vector<sycl::event>{flash_attention_tiled_stream(queue, Q.data(), K_tensor.data(), V.data(), output_old.data(), M, N, D, Dv, tile_size)};
            }, warmup, iterations, queue);

            const double old_new_ms = benchmark_ms([&]() {
                return std::vector<sycl::event>{flash_attention_tiled_stream(queue, Q.data(), K_tensor.data(), V.data(), output_old_new.data(), M, N, D, Dv, tile_size)};
            }, warmup, iterations, queue);

            std::cout << std::left << std::setw(8) << iterations << std::fixed << std::setprecision(4)
                      << std::setw(13) << naive_ms << std::setw(13) << online_ms << std::setw(13) << merge_ms
                      << std::setw(13) << old_ms << std::setw(13) << old_new_ms
                      << std::setprecision(2)
                      << std::setw(12) << naive_ms / online_ms << std::setw(12) << naive_ms / merge_ms
                      << std::setw(12) << naive_ms / old_ms << std::setw(12) << naive_ms / old_new_ms << "\n";

            queue.wait();
        }
        catch (const sycl::exception& e) {
            std::cout << "\nFAILED at size " << size << "\nSYCL error: " << e.what() << "\n";
            break;
        }
        catch (const std::exception& e) {
            std::cout << "\nFAILED at size " << size << "\nError: " << e.what() << "\n";
            break;
        }
    }

    std::cout << "\n=== Benchmark Finished ===\n";
    return 0;
}
