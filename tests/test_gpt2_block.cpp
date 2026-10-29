/*
cmake -S . -B build -DCMAKE_CXX_COMPILER=icpx -DTINYML_BUILD_TESTS=ON -DTINYML_BUILD_GPT2_BLOCK_TEST=ON;cmake --build build --target test_gpt2_block -j2; ./build/test_gpt2_block
*/


/*
Configure and build with CMake:
cmake -S . -B build -DCMAKE_CXX_COMPILER=icpx -DTINYML_BUILD_TESTS=ON -DTINYML_BUILD_GPT2_BLOCK_TEST=ON
cmake --build build --target test_gpt2_block -j2

Run one block:
./build/test_gpt2_block 0

Run all 12 blocks after exporting weights and references:
for layer in $(seq 0 11); do ./build/test_gpt2_block "$layer" || break; done
*/

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tinyml/ops.hpp"
#include "tinyml/tensor.hpp"
#include "tinyml/transformer_block.hpp"
#include "tinyml/gpt2_block_weights.hpp"
#include "tinyml/gpt2_weights.hpp"

int main(int argc, char** argv) {
    try {
        const std::size_t layer = argc > 1 ? std::stoul(argv[1]) : 0;
        if (layer >= 12) throw std::invalid_argument("GPT-2 Small layer index must be between 0 and 11");

        sycl::queue queue{sycl::gpu_selector_v};
        constexpr std::size_t B = 1, L = 4, D = 768, D_FF = 3072;
        constexpr std::size_t N_HEADS = 12, GROUP_SIZE = 256, TILE_SIZE = 16;
        constexpr float TOLERANCE = 5.0e-3f;
        const std::string dir = "models/gpt2/block" + std::to_string(layer);

        std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << '\n';
        std::cout << "Testing GPT-2 block: " << layer << '\n';

        auto h_input = tinyml::read_gpt2_weights(dir + "/test_input.bin", B * L * D);
        auto h_reference = tinyml::read_gpt2_weights(dir + "/reference_output.bin", B * L * D);

        tinyml::TransformerBlock<float> block(queue, D, N_HEADS, D_FF);
        tinyml::load_gpt2_block_weights(queue, block, dir);

        tinyml::Tensor<float> X(queue, {B, L, D});
        tinyml::Tensor<float> Y(queue, {B, L, D});
        tinyml::Tensor<float> X_norm(queue, {B, L, D});
        tinyml::Tensor<float> Q(queue, {B, L, D});
        tinyml::Tensor<float> K(queue, {B, L, D});
        tinyml::Tensor<float> V(queue, {B, L, D});
        tinyml::Tensor<float> A(queue, {B, L, D});
        tinyml::Tensor<float> R(queue, {B, L, D});
        tinyml::Tensor<float> R_norm(queue, {B, L, D});
        tinyml::Tensor<float> F1(queue, {B, L, D_FF});
        tinyml::Tensor<float> F1_gelu(queue, {B, L, D_FF});
        tinyml::Tensor<float> F2(queue, {B, L, D});

        auto e_input = ops::copy_to_device(queue, X, h_input.data());

        auto e_output = block.forward(
            X, Y, X_norm, Q, K, V, A, R, R_norm, F1, F1_gelu, F2,
            GROUP_SIZE, TILE_SIZE, {e_input});

        std::vector<float> h_output(B * L * D);
        ops::copy_to_host(queue, h_output.data(), Y, {e_output}).wait_and_throw();

        float max_abs_error = 0.0f;
        double sum_abs_error = 0.0;
        double sum_squared_error = 0.0;
        bool finite = true;
        std::size_t max_error_index = 0;

        for (std::size_t i = 0; i < h_output.size(); ++i) {
            if (!std::isfinite(h_output[i]) || !std::isfinite(h_reference[i])) {
                finite = false;
                continue;
            }

            const float error = std::abs(h_output[i] - h_reference[i]);

            if (error > max_abs_error) {
                max_abs_error = error;
                max_error_index = i;
            }

            sum_abs_error += error;
            sum_squared_error += static_cast<double>(error) * error;
        }

        const double mean_abs_error = sum_abs_error / h_output.size();
        const double rmse = std::sqrt(sum_squared_error / h_output.size());

        std::cout << std::setprecision(10);
        std::cout << "Input shape: [" << B << ',' << L << ',' << D << "]\n";
        std::cout << "Output elements: " << h_output.size() << '\n';
        std::cout << "Maximum absolute error: " << max_abs_error << '\n';
        std::cout << "Mean absolute error: " << mean_abs_error << '\n';
        std::cout << "RMSE: " << rmse << '\n';
        std::cout << "Maximum-error index: " << max_error_index
                  << " (token " << max_error_index / D
                  << ", feature " << max_error_index % D << ")\n";
        std::cout << "SYCL output: " << h_output[max_error_index] << '\n';
        std::cout << "Reference:   " << h_reference[max_error_index] << '\n';

        if (finite && max_abs_error < TOLERANCE) {
            std::cout << "GPT-2 BLOCK " << layer << " TEST PASSED\n";
            return 0;
        }

        std::cout << "GPT-2 BLOCK " << layer << " TEST FAILED\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "GPT-2 block test error: " << e.what() << '\n';
        return 1;
    }
}

