#include <sycl/sycl.hpp>
#include <iostream>
#include <algorithm>

# include "tinyml/device_buffer.hpp"
# include "tinyml/tensor.hpp"
# include "tinyml/ops.hpp"

int main()
{
    size_t counts= 4096, group_size= 256;
    //std::vector<sycl::event> no_dependencies;

    //sycl::queue queue;
    // sycl::queue queue{sycl::gpu_selector_v};
    sycl::queue queue{sycl::gpu_selector_v,sycl::property_list{sycl::property::queue::enable_profiling{}}};

    //    std::vector<float> float_data_h(counts);
    //tinyml::DeviceBuffer<float> float_buffer(queue, counts);

    std::cout << "Device: "
              << queue.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    
    const std::size_t M = 2024;
    const std::size_t K = 3027;
    const std::size_t N = 2024;
    const std::size_t tile_size = 16;

    std::vector<float> A_h(M * K);
    std::vector<float> B_h(K * N);
    std::vector<float> C_h(M * N);

    for (std::size_t i = 0; i < A_h.size(); i++) {
        A_h[i] = 1.0f;
    }

    for (std::size_t i = 0; i < B_h.size(); i++) {
        B_h[i] = 1.0f;
    }

    tinyml::Tensor<float> A(queue, {M, K});
    tinyml::Tensor<float> B(queue, {K, N});
    tinyml::Tensor<float> C(queue, {M, N});

    sycl::event e_A = ops::copy_to_device(queue, A, A_h.data());

    sycl::event e_B = ops::copy_to_device(queue, B, B_h.data());

    std::vector<sycl::event> dependencies = {e_A, e_B};

    const int warmup_runs = 10;
    const int benchmark_runs = 51;

    std::vector<double> tiled_times;
    std::vector<double> naive_times;

    for (int i = 0; i < warmup_runs; i++) {
        ops::matmul_tiled(queue, A, B, C, tile_size, dependencies).wait();
    }

    for (int i = 0; i < benchmark_runs; i++) {
        sycl::event e = ops::matmul_tiled(queue, A, B, C, tile_size, dependencies);

        e.wait();

        auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();

        auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();

        tiled_times.push_back(
        static_cast<double>(end - start) / 1e6);
    }


    for (int i = 0; i < warmup_runs; i++) {
        ops::matmul(queue, A, B, C, 256, dependencies).wait();
    }

    for (int i = 0; i < benchmark_runs; i++) {
        sycl::event e =
        ops::matmul(queue, A, B, C, 256, dependencies);

        e.wait();

        auto start =
        e.get_profiling_info<sycl::info::event_profiling::command_start>();

        auto end =
        e.get_profiling_info<sycl::info::event_profiling::command_end>();

        naive_times.push_back(
        static_cast<double>(end - start) / 1e6);
    }


    std::sort(tiled_times.begin(), tiled_times.end());
    std::sort(naive_times.begin(), naive_times.end());

    double tiled_median =
    tiled_times[tiled_times.size() / 2];

    double naive_median =
    naive_times[naive_times.size() / 2];

    std::cout << "Tiled median: " << tiled_median << " ms" << std::endl;

    std::cout << "Naive median: "<< naive_median << " ms" << std::endl;

    std::cout << "Speedup: "  << naive_median / tiled_median << "x" << std::endl;
    return 0;
}