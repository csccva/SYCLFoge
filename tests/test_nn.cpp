#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "tinyml/device_buffer.hpp"
#include "tinyml/tensor.hpp"
#include "tinyml/ops.hpp"
#include "tinyml/linear.hpp"

int main()
{
    sycl::queue queue{sycl::gpu_selector_v};

    bool all_tests_passed=true;
    const float tolerance=1.0e-5f;

    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";

    {
        std::cout << "\n=== ReLU Test ===\n";

        const std::size_t M = 2;
        const std::size_t N = 4;

        std::vector<float> h_in{
            -2.0f, -1.0f, 0.0f, 1.0f,
             2.0f, -3.0f, 4.0f, -5.0f
        };

        tinyml::Tensor<float> input(queue, {M,N});
        tinyml::Tensor<float> output(queue, {M,N});

        sycl::event e_input =
            ops::copy_to_device(
                queue, input, h_in.data());

        sycl::event e_relu =
            ops::relu(
                queue, input, output, 256, {e_input});

        std::vector<float> h_out(M * N);
        std::vector<float> h_reference(M * N);

        ops::copy_to_host(
            queue, h_out.data(), output, {e_relu}
        ).wait_and_throw();

        // -----------------------------------------------------
        // CPU reference
        // -----------------------------------------------------

        for (std::size_t i = 0; i < M * N; i++) {

            h_reference[i] =
                std::max(0.0f, h_in[i]);
        }

        // -----------------------------------------------------
        // Numerical validation
        // -----------------------------------------------------

        float max_diff = 0.0f;
        bool finite_results = true;

        for (std::size_t i = 0; i < M * N; i++) {

            std::cout << h_out[i] << " ";

            if (!std::isfinite(h_out[i])) {

                finite_results = false;
                continue;
            }

            float diff =
                std::abs(h_out[i] - h_reference[i]);

            max_diff = std::max(max_diff, diff);
        }

        std::cout
            << "\nMaximum difference: "
            << max_diff << "\n";

        if (finite_results && max_diff < tolerance) {

            std::cout << "ReLU test PASSED\n";
        }
        else {

            std::cout << "ReLU test FAILED\n";

            all_tests_passed = false;
        }
    }

    {
        std::cout << "\n=== Add Bias Test ===\n";

        const std::size_t M = 3;
        const std::size_t N = 4;

        std::vector<float> h_in{
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f
        };

        std::vector<float> h_bias{
            0.5f, -1.0f, 2.0f, -0.5f
        };

        tinyml::Tensor<float> input(queue, {M,N});
        tinyml::Tensor<float> bias(queue, {N});
        tinyml::Tensor<float> output(queue, {M,N});

        sycl::event e_input =
            ops::copy_to_device(
                queue, input, h_in.data());

        sycl::event e_bias =
            ops::copy_to_device(
                queue, bias, h_bias.data());

        sycl::event e_add_bias =
            ops::add_bias(
                queue,
                input,
                bias,
                output,
                256,
                {e_input, e_bias}
            );

        std::vector<float> h_out(M * N);
        std::vector<float> h_reference(M * N);

        ops::copy_to_host(
            queue, h_out.data(), output, {e_add_bias}
        ).wait_and_throw();

        // -----------------------------------------------------
        // CPU reference
        // -----------------------------------------------------

        for (std::size_t i = 0; i < M; i++) {

            for (std::size_t j = 0; j < N; j++) {

                h_reference[i * N + j] =
                    h_in[i * N + j] + h_bias[j];
            }
        }

        // -----------------------------------------------------
        // Numerical validation
        // -----------------------------------------------------

        float max_diff = 0.0f;
        bool finite_results = true;

        for (std::size_t i = 0; i < M * N; i++) {

            std::cout << h_out[i] << " ";

            if (!std::isfinite(h_out[i])) {

                finite_results = false;
                continue;
            }

            float diff =
                std::abs(h_out[i] - h_reference[i]);

            max_diff = std::max(max_diff, diff);
        }

        std::cout
            << "\nMaximum difference: "
            << max_diff << "\n";

        if (finite_results && max_diff < tolerance) {

            std::cout << "Add Bias test PASSED\n";
        }
        else {

            std::cout << "Add Bias test FAILED\n";

            all_tests_passed = false;
        }
    }
    {
        std::cout << "\n=== Exact GELU Test ===\n";

        const std::size_t M = 3;
        const std::size_t N = 4;

        std::vector<float> h_in{
            -10.0f, -5.0f, -2.0f, -1.0f,
             -0.5f,  0.0f,  0.5f,  1.0f,
              2.0f,  3.0f,  5.0f, 10.0f
        };

        tinyml::Tensor<float> input(queue, {M,N});
        tinyml::Tensor<float> output(queue, {M,N});

        sycl::event e_input =
            ops::copy_to_device(queue, input, h_in.data());

        sycl::event e_gelu =
            ops::gelu_exact(queue, input, output, 256, {e_input});

        std::vector<float> h_out(M * N);
        std::vector<float> h_reference(M * N);

        ops::copy_to_host(queue, h_out.data(), output, {e_gelu}).wait_and_throw();

        // -----------------------------------------------------
        // Independent CPU reference
        // -----------------------------------------------------

        for (std::size_t i = 0; i < M * N; i++) {

            double x = static_cast<double>(h_in[i]);

            double value = 0.5 * x * std::erfc(-x / std::sqrt(2.0));

            h_reference[i] = static_cast<float>(value);
        }

        // -----------------------------------------------------
        // Numerical validation
        // -----------------------------------------------------

        float max_diff = 0.0f;
        bool finite_results = true;

        for (std::size_t i = 0; i < M * N; i++) {

            std::cout << "Input: " << h_in[i]
                      << " GPU: " << h_out[i]
                      << " CPU: " << h_reference[i]
                      << "\n";

            if (!std::isfinite(h_out[i]) || !std::isfinite(h_reference[i])) {

                finite_results = false;
                continue;
            }

            float diff = std::abs(h_out[i] - h_reference[i]);

            max_diff = std::max(max_diff, diff);
        }

        std::cout << "Maximum difference: " << max_diff << "\n";

        if (finite_results && max_diff < tolerance) {

            std::cout << "Exact GELU test PASSED\n";
        }
        else {

            std::cout << "Exact GELU test FAILED\n";

            all_tests_passed = false;
        }
    }
    {
        std::cout << "\n=== Approx GELU Test ===\n";

        const std::size_t M = 3;
        const std::size_t N = 4;
        const float gelu_approx_tolerance = 5.0e-4f;

        std::vector<float> h_in{
            -10.0f, -5.0f, -2.0f, -1.0f,
             -0.5f,  0.0f,  0.5f,  1.0f,
              2.0f,  3.0f,  5.0f, 10.0f
        };

        tinyml::Tensor<float> input(queue, {M,N});
        tinyml::Tensor<float> output(queue, {M,N});

        sycl::event e_input =
            ops::copy_to_device(queue, input, h_in.data());

        sycl::event e_gelu =
            ops::gelu_approx(queue, input, output, 256, {e_input});

        std::vector<float> h_out(M * N);
        std::vector<float> h_reference(M * N);

        ops::copy_to_host(queue, h_out.data(), output, {e_gelu}).wait_and_throw();

        // -----------------------------------------------------
        // Independent CPU reference
        // -----------------------------------------------------

        for (std::size_t i = 0; i < M * N; i++) {

            double x = static_cast<double>(h_in[i]);

            double value = 0.5 * x * std::erfc(-x / std::sqrt(2.0));

            h_reference[i] = static_cast<float>(value);
        }

        // -----------------------------------------------------
        // Numerical validation
        // -----------------------------------------------------

        float max_diff = 0.0f;
        bool finite_results = true;

        for (std::size_t i = 0; i < M * N; i++) {

            std::cout << "Input: " << h_in[i]
                      << " GPU: " << h_out[i]
                      << " CPU: " << h_reference[i]
                      << "\n";

            if (!std::isfinite(h_out[i]) || !std::isfinite(h_reference[i])) {

                finite_results = false;
                continue;
            }

            float diff = std::abs(h_out[i] - h_reference[i]);

            max_diff = std::max(max_diff, diff);
        }

        std::cout << "Maximum difference: " << max_diff << "\n";

        if (finite_results && max_diff < gelu_approx_tolerance) {

            std::cout << "Approx GELU test PASSED\n";
        }
        else {

            std::cout << "Approx GELU test FAILED\n";

            all_tests_passed = false;
        }
    }

    
    {
        std::cout << "\n=== Serial Softmax Test ===\n";

        const std::size_t M = 3;
        const std::size_t N = 4;

        std::vector<float> h_in{
            1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 2.0f, 3.0f, 4.0f,
            4.0f, 3.0f, 2.0f, 1.0f
        };

        tinyml::Tensor<float> input(queue, {M,N});
        tinyml::Tensor<float> output(queue, {M,N});

        sycl::event e_in =
            ops::copy_to_device(
                queue, input, h_in.data());

        sycl::event e_sm =
            ops::softmax(
                queue, input, output, 256, {e_in});

        std::vector<float> h_out(M * N);
        std::vector<float> h_reference(M * N);

        ops::copy_to_host(
            queue, h_out.data(), output, {e_sm}
        ).wait_and_throw();

        // -----------------------------------------------------
        // Independent CPU reference
        // -----------------------------------------------------

        for (std::size_t i = 0; i < M; i++) {

            float row_max = h_in[i * N];

            for (std::size_t j = 1; j < N; j++) {

                row_max = std::max(
                    row_max,
                    h_in[i * N + j]
                );
            }

            double row_sum = 0.0;

            for (std::size_t j = 0; j < N; j++) {

                row_sum += std::exp(
                    static_cast<double>(
                        h_in[i * N + j]
                    ) - row_max
                );
            }

            for (std::size_t j = 0; j < N; j++) {

                h_reference[i * N + j] =
                    static_cast<float>(
                        std::exp(
                            static_cast<double>(
                                h_in[i * N + j]
                            ) - row_max
                        ) / row_sum
                    );
            }
        }

        // -----------------------------------------------------
        // Numerical validation
        // -----------------------------------------------------

        float max_diff = 0.0f;
        bool finite_results = true;

        for (std::size_t i = 0; i < M * N; i++) {

            std::cout << h_out[i] << " ";

            if (!std::isfinite(h_out[i])) {

                finite_results = false;
                continue;
            }

            float diff =
                std::abs(h_out[i] - h_reference[i]);

            max_diff = std::max(max_diff, diff);
        }

        std::cout
            << "\nMaximum difference: "
            << max_diff << "\n";

        if (finite_results && max_diff < tolerance) {

            std::cout << "Serial Softmax test PASSED\n";
        }
        else {

            std::cout << "Serial Softmax test FAILED\n";

            all_tests_passed = false;
        }
    }

     
    {
        std::cout << "\n=== Parallel Softmax Test ===\n";

        const std::size_t M = 3;
        const std::size_t N = 4;

        std::vector<float> h_in{
            1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 2.0f, 3.0f, 4.0f,
            4.0f, 3.0f, 2.0f, 1.0f
        };

        tinyml::Tensor<float> input(queue, {M,N});
        tinyml::Tensor<float> output(queue, {M,N});

        sycl::event e_in =
            ops::copy_to_device(
                queue, input, h_in.data());

        sycl::event e_sm =
            ops::softmax_parallel(
                queue, input, output, 256, {e_in});

        std::vector<float> h_out(M * N);
        std::vector<float> h_reference(M * N);

        ops::copy_to_host(
            queue, h_out.data(), output, {e_sm}
        ).wait_and_throw();

        // -----------------------------------------------------
        // Independent CPU reference
        // -----------------------------------------------------

        for (std::size_t i = 0; i < M; i++) {

            float row_max = h_in[i * N];

            for (std::size_t j = 1; j < N; j++) {

                row_max = std::max(
                    row_max,
                    h_in[i * N + j]
                );
            }

            double row_sum = 0.0;

            for (std::size_t j = 0; j < N; j++) {

                row_sum += std::exp(
                    static_cast<double>(
                        h_in[i * N + j]
                    ) - row_max
                );
            }

            for (std::size_t j = 0; j < N; j++) {

                h_reference[i * N + j] =
                    static_cast<float>(
                        std::exp(
                            static_cast<double>(
                                h_in[i * N + j]
                            ) - row_max
                        ) / row_sum
                    );
            }
        }

        // -----------------------------------------------------
        // Numerical validation
        // -----------------------------------------------------

        float max_diff = 0.0f;
        bool finite_results = true;

        for (std::size_t i = 0; i < M * N; i++) {

            std::cout << h_out[i] << " ";

            if (!std::isfinite(h_out[i])) {

                finite_results = false;
                continue;
            }

            float diff =
                std::abs(h_out[i] - h_reference[i]);

            max_diff = std::max(max_diff, diff);
        }

        std::cout
            << "\nMaximum difference: "
            << max_diff << "\n";

        if (finite_results && max_diff < tolerance) {

            std::cout << "Parallel Softmax test PASSED\n";
        }
        else {

            std::cout << "Parallel Softmax test FAILED\n";

            all_tests_passed = false;
        }
    }

    {
        std::cout
            << "\n=== Softmax Comparison Test ===\n";

        const std::size_t M = 3;
        const std::size_t N = 1000;
        const std::size_t group_size = 256;

        std::vector<float> h_in(M * N);

        for (std::size_t i = 0; i < M; i++) {

            for (std::size_t j = 0; j < N; j++) {

                h_in[i * N + j] =
                    static_cast<float>(j % 17) * 0.1f
                    - static_cast<float>(i);
            }
        }

        tinyml::Tensor<float> input(queue, {M,N});

        tinyml::Tensor<float> output_serial(
            queue, {M,N});

        tinyml::Tensor<float> output_parallel(
            queue, {M,N});

        sycl::event e_input =
            ops::copy_to_device(
                queue, input, h_in.data());

        sycl::event e_serial =
            ops::softmax(
                queue,
                input,
                output_serial,
                group_size,
                {e_input}
            );

        sycl::event e_parallel =
            ops::softmax_parallel(
                queue,
                input,
                output_parallel,
                group_size,
                {e_input}
            );

        std::vector<float> h_serial(M * N);
        std::vector<float> h_parallel(M * N);

        sycl::event e_serial_copy =
            ops::copy_to_host(
                queue,
                h_serial.data(),
                output_serial,
                {e_serial}
            );

        sycl::event e_parallel_copy =
            ops::copy_to_host(
                queue,
                h_parallel.data(),
                output_parallel,
                {e_parallel}
            );

        e_serial_copy.wait_and_throw();
        e_parallel_copy.wait_and_throw();

        // -----------------------------------------------------
        // Independent CPU reference
        // -----------------------------------------------------

        std::vector<float> h_reference(M * N);

        for (std::size_t i = 0; i < M; i++) {

            double row_max = h_in[i * N];

            for (std::size_t j = 1; j < N; j++) {

                row_max = std::max(
                    row_max,
                    static_cast<double>(
                        h_in[i * N + j]
                    )
                );
            }

            double row_sum = 0.0;

            for (std::size_t j = 0; j < N; j++) {

                row_sum += std::exp(
                    static_cast<double>(
                        h_in[i * N + j]
                    ) - row_max
                );
            }

            for (std::size_t j = 0; j < N; j++) {

                h_reference[i * N + j] =
                    static_cast<float>(
                        std::exp(
                            static_cast<double>(
                                h_in[i * N + j]
                            ) - row_max
                        ) / row_sum
                    );
            }
        }

        // -----------------------------------------------------
        // Numerical validation
        // -----------------------------------------------------

        float max_serial_parallel_diff = 0.0f;
        float max_serial_reference_diff = 0.0f;
        float max_parallel_reference_diff = 0.0f;

        bool finite_results = true;
        bool valid_row_sums = true;

        for (std::size_t i = 0; i < M * N; i++) {

            if (!std::isfinite(h_serial[i]) ||
                !std::isfinite(h_parallel[i]))
            {
                finite_results = false;
                continue;
            }

            max_serial_parallel_diff =
                std::max(
                    max_serial_parallel_diff,
                    std::abs(
                        h_serial[i] - h_parallel[i]
                    )
                );

            max_serial_reference_diff =
                std::max(
                    max_serial_reference_diff,
                    std::abs(
                        h_serial[i] - h_reference[i]
                    )
                );

            max_parallel_reference_diff =
                std::max(
                    max_parallel_reference_diff,
                    std::abs(
                        h_parallel[i] - h_reference[i]
                    )
                );
        }

        std::cout
            << "Maximum serial vs parallel difference: "
            << max_serial_parallel_diff << "\n";

        std::cout
            << "Maximum serial vs CPU reference difference: "
            << max_serial_reference_diff << "\n";

        std::cout
            << "Maximum parallel vs CPU reference difference: "
            << max_parallel_reference_diff << "\n";

        // -----------------------------------------------------
        // Row sums
        // -----------------------------------------------------

        for (std::size_t i = 0; i < M; i++) {

            double row_sum = 0.0;

            for (std::size_t j = 0; j < N; j++) {

                row_sum +=
                    static_cast<double>(
                        h_parallel[i * N + j]
                    );
            }

            std::cout
                << "Row " << i
                << " sum: " << row_sum
                << "\n";

            if (!std::isfinite(row_sum) ||
                std::abs(row_sum - 1.0) >= tolerance)
            {
                valid_row_sums = false;
            }
        }

        // -----------------------------------------------------
        // Final validation
        // -----------------------------------------------------

        if (finite_results &&
            valid_row_sums &&
            max_serial_parallel_diff < tolerance &&
            max_serial_reference_diff < tolerance &&
            max_parallel_reference_diff < tolerance)
        {
            std::cout
                << "Softmax Comparison test PASSED\n";
        }
        else {

            std::cout
                << "Softmax Comparison test FAILED\n";

            all_tests_passed = false;
        }
    }

    {
        std::cout << "\n=== LayerNorm Test ===\n";

        std::vector<float> h_in{
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 5.0f, 5.0f, 5.0f
        };

        tinyml::Tensor<float> input(queue, {2,4});
        tinyml::Tensor<float> output(queue, {2,4});

        sycl::event e_in =
            ops::copy_to_device(
                queue, input, h_in.data());

        sycl::event e_ln =
            ops::layer_norm(
                queue, input, output, 256, {e_in});

        std::vector<float> h_out(8);

        ops::copy_to_host(
            queue, h_out.data(), output, {e_ln}
        ).wait_and_throw();

        std::cout << "LayerNorm output:\n";

        for (std::size_t i = 0; i < 2; i++) {

            std::cout << "Row " << i << ": ";

            for (std::size_t j = 0; j < 4; j++) {

                std::cout
                    << h_out[i * 4 + j] << " ";
            }

            std::cout << "\n";
        }

        std::vector<float> expected{
            -1.341635f, -0.447212f,
             0.447212f,  1.341635f,
             0.0f,       0.0f,
             0.0f,       0.0f
        };

        float max_diff = 0.0f;
        bool finite_results = true;

        for (std::size_t i = 0; i < h_out.size(); i++) {

            if (!std::isfinite(h_out[i])) {

                finite_results = false;
                continue;
            }

            float diff =
                std::abs(h_out[i] - expected[i]);

            max_diff = std::max(max_diff, diff);
        }

        std::cout
            << "Maximum difference: "
            << max_diff << "\n";

        if (finite_results && max_diff < tolerance) {

            std::cout << "LayerNorm test PASSED\n";
        }
        else {

            std::cout << "LayerNorm test FAILED\n";

            all_tests_passed = false;
        }
    }

    {
        std::cout
            << "\n=== Parallel LayerNorm Test ===\n";

        const std::size_t M = 3;
        const std::size_t N = 1000;
        const std::size_t group_size = 256;

        const float epsilon = 1.0e-5f;

        // -----------------------------------------------------
        // Input
        // -----------------------------------------------------

        std::vector<float> h_in(M * N);

        for (std::size_t i = 0; i < M; i++) {

            for (std::size_t j = 0; j < N; j++) {

                if (i == 0) {

                    h_in[i * N + j] =
                        static_cast<float>(j % 17)
                        * 0.1f;
                }
                else if (i == 1) {

                    // Constant row

                    h_in[i * N + j] = 1000.0f;
                }
                else {

                    h_in[i * N + j] =
                        static_cast<float>(j % 23)
                        * 0.2f - 2.0f;
                }
            }
        }

        // -----------------------------------------------------
        // Allocate tensors
        // -----------------------------------------------------

        tinyml::Tensor<float> input(queue, {M,N});
        tinyml::Tensor<float> gamma(queue,{N});
        tinyml::Tensor<float> beta(queue,{N});

        tinyml::Tensor<float> output_serial(
            queue, {M,N});

        tinyml::Tensor<float> output_parallel(
            queue, {M,N});

        // -----------------------------------------------------
        // Copy input to GPU
        // -----------------------------------------------------
        std::vector<float> h_gamma(N,1.0f);
        std::vector<float> h_beta(N,0.0f);

        sycl::event e_input = ops::copy_to_device(queue,input,h_in.data());
        sycl::event e_gamma=ops::copy_to_device(queue,gamma,h_gamma.data());
        sycl::event e_beta=ops::copy_to_device(queue,beta,h_beta.data());
        // -----------------------------------------------------
        // Serial LayerNorm
        // -----------------------------------------------------

        sycl::event e_serial =
            ops::layer_norm(
                queue,
                input,
                output_serial,
                group_size,
                {e_input}
            );

        // -----------------------------------------------------
        // Parallel LayerNorm
        // -----------------------------------------------------

        sycl::event e_parallel = ops::layer_norm_parallel(queue,input,gamma,beta,output_parallel,group_size,{e_input,e_gamma,e_beta});

        // -----------------------------------------------------
        // Copy results to CPU
        // -----------------------------------------------------

        std::vector<float> h_serial(M * N);
        std::vector<float> h_parallel(M * N);

        sycl::event e_serial_copy =
            ops::copy_to_host(
                queue,
                h_serial.data(),
                output_serial,
                {e_serial}
            );

        sycl::event e_parallel_copy =
            ops::copy_to_host(
                queue,
                h_parallel.data(),
                output_parallel,
                {e_parallel}
            );

        e_serial_copy.wait_and_throw();
        e_parallel_copy.wait_and_throw();

        // -----------------------------------------------------
        // Independent CPU reference
        // -----------------------------------------------------

        std::vector<float> h_reference(M * N);

        std::vector<double> cpu_mean(M);
        std::vector<double> cpu_variance(M);

        for (std::size_t i = 0; i < M; i++) {

            double sum = 0.0;

            for (std::size_t j = 0; j < N; j++) {

                sum +=
                    static_cast<double>(
                        h_in[i * N + j]
                    );
            }

            double mean =
                sum / static_cast<double>(N);

            double variance = 0.0;

            for (std::size_t j = 0; j < N; j++) {

                double diff =
                    static_cast<double>(
                        h_in[i * N + j]
                    ) - mean;

                variance += diff * diff;
            }

            variance /= static_cast<double>(N);

            cpu_mean[i] = mean;
            cpu_variance[i] = variance;

            double denominator =
                std::sqrt(
                    variance + static_cast<double>(epsilon)
                );

            for (std::size_t j = 0; j < N; j++) {

                h_reference[i * N + j] =
                    static_cast<float>(
                        (static_cast<double>(
                            h_in[i * N + j]
                        ) - mean) / denominator
                    );
            }
        }

        // -----------------------------------------------------
        // Numerical validation
        // -----------------------------------------------------

        float max_serial_parallel_diff = 0.0f;
        float max_serial_reference_diff = 0.0f;
        float max_parallel_reference_diff = 0.0f;

        bool finite_results = true;

        std::size_t max_error_row = 0;
        std::size_t max_error_col = 0;

        for (std::size_t i = 0; i < M * N; i++) {

            if (!std::isfinite(h_serial[i]) ||
                !std::isfinite(h_parallel[i]) ||
                !std::isfinite(h_reference[i]))
            {
                finite_results = false;
                continue;
            }

            float serial_parallel_diff =
                std::abs(
                    h_serial[i] - h_parallel[i]
                );

            float serial_reference_diff =
                std::abs(
                    h_serial[i] - h_reference[i]
                );

            float parallel_reference_diff =
                std::abs(
                    h_parallel[i] - h_reference[i]
                );

            max_serial_parallel_diff =
                std::max(
                    max_serial_parallel_diff,
                    serial_parallel_diff
                );

            max_serial_reference_diff =
                std::max(
                    max_serial_reference_diff,
                    serial_reference_diff
                );

            if (parallel_reference_diff >
                max_parallel_reference_diff)
            {
                max_parallel_reference_diff =
                    parallel_reference_diff;

                max_error_row = i / N;
                max_error_col = i % N;
            }
        }

        std::cout
            << "Maximum serial vs parallel difference: "
            << max_serial_parallel_diff << "\n";

        std::cout
            << "Maximum serial vs CPU reference difference: "
            << max_serial_reference_diff << "\n";

        std::cout
            << "Maximum parallel vs CPU reference difference: "
            << max_parallel_reference_diff << "\n";

        std::cout
            << "Maximum parallel error location: row "
            << max_error_row
            << ", column "
            << max_error_col
            << "\n";

        // -----------------------------------------------------
        // Per-row validation
        // -----------------------------------------------------

        bool nonconstant_rows_passed = true;
        bool constant_row_passed = true;

        for (std::size_t i = 0; i < M; i++) {

            float max_row_diff = 0.0f;

            double row_mean = 0.0;
            double row_variance = 0.0;

            for (std::size_t j = 0; j < N; j++) {

                float diff =
                    std::abs(
                        h_parallel[i * N + j] -
                        h_reference[i * N + j]
                    );

                max_row_diff =
                    std::max(max_row_diff, diff);

                row_mean +=
                    static_cast<double>(
                        h_parallel[i * N + j]
                    );
            }

            row_mean /= static_cast<double>(N);

            for (std::size_t j = 0; j < N; j++) {

                double diff =
                    static_cast<double>(
                        h_parallel[i * N + j]
                    ) - row_mean;

                row_variance += diff * diff;
            }

            row_variance /= static_cast<double>(N);

            double expected_variance =
                cpu_variance[i] /
                (cpu_variance[i] +
                 static_cast<double>(epsilon));

            std::cout
                << "\nRow " << i << "\n";

            std::cout
                << "CPU mean:           "
                << cpu_mean[i] << "\n";

            std::cout
                << "CPU variance:       "
                << cpu_variance[i] << "\n";

            std::cout
                << "Output mean:        "
                << row_mean << "\n";

            std::cout
                << "Output variance:    "
                << row_variance << "\n";

            std::cout
                << "Expected variance:  "
                << expected_variance << "\n";

            std::cout
                << "Maximum row error:  "
                << max_row_diff << "\n";

            if (i == 1) {

                if (max_row_diff < tolerance) {

                    std::cout
                        << "Constant row test PASSED\n";
                }
                else {

                    std::cout
                        << "Constant row test FAILED\n";

                    constant_row_passed = false;
                }
            }
            else {

                if (max_row_diff < tolerance) {

                    std::cout
                        << "Row " << i
                        << " test PASSED\n";
                }
                else {

                    std::cout
                        << "Row " << i
                        << " test FAILED\n";

                    nonconstant_rows_passed = false;
                }
            }
        }

        // -----------------------------------------------------
        // Final validation
        // -----------------------------------------------------

        if (finite_results &&
            nonconstant_rows_passed &&
            constant_row_passed &&
            max_serial_parallel_diff < tolerance &&
            max_serial_reference_diff < tolerance &&
            max_parallel_reference_diff < tolerance)
        {
            std::cout
                << "\nParallel LayerNorm test PASSED\n";
        }
        else {

            std::cout
                << "\nParallel LayerNorm test FAILED\n";

            all_tests_passed = false;
        }
    }

    
    {
        std::cout << "\n--- Linear forward_3d test ---\n";
    
        std::size_t B=2;
        std::size_t L=3;
        std::size_t K=4;
        std::size_t N=5;
        std::size_t group_size=8;
        std::size_t tile_size=2;
    
        tinyml::Tensor<float> X3(queue,{B,L,K});
        tinyml::Tensor<float> Y3(queue,{B,L,N});
    
        tinyml::Tensor<float> X2(queue,{B*L,K});
        tinyml::Tensor<float> Y2(queue,{B*L,N});
    
        tinyml::Linear<float> linear(queue,K,N);
    
        std::vector<float> h_X(B*L*K);
        std::vector<float> h_W(K*N);
        std::vector<float> h_b(N);
        std::vector<float> h_Y3(B*L*N);
        std::vector<float> h_Y2(B*L*N);
    
        for(std::size_t i=0;i<h_X.size();i++){
            h_X[i]=static_cast<float>((static_cast<int>(i)%11)-5)*0.1f;
        }
    
        for(std::size_t i=0;i<h_W.size();i++){
            h_W[i]=static_cast<float>((static_cast<int>(i)%7)-3)*0.1f;
        }
    
        for(std::size_t i=0;i<h_b.size();i++){
            h_b[i]=static_cast<float>(i)*0.05f;
        }
    
        auto e1=ops::copy_to_device(queue,X3,h_X.data());
        auto e2=ops::copy_to_device(queue,X2,h_X.data());
        auto e3=ops::copy_to_device(queue,linear.weight(),h_W.data());
        auto e4=ops::copy_to_device(queue,linear.bias(),h_b.data());
    
        auto e3d=linear.forward_3d(X3,Y3,group_size,tile_size,{e1,e3,e4});
        auto e2d=linear.forward(X2,Y2,group_size,tile_size,{e2,e3,e4});
    
        auto e5=ops::copy_to_host(queue,h_Y3.data(),Y3,{e3d});
        auto e6=ops::copy_to_host(queue,h_Y2.data(),Y2,{e2d});
    
        e5.wait();
        e6.wait();
    
        float max_diff=0.0f;
    
        for(std::size_t i=0;i<h_Y3.size();i++){
            max_diff=std::max(max_diff,std::abs(h_Y3[i]-h_Y2[i]));
        }
    
        std::cout << "X3=[" << B << "," << L << "," << K << "]  Y3=[" << B << "," << L << "," << N << "]\n";
        std::cout << "Equivalent 2D: X2=[" << B*L << "," << K << "]  Y2=[" << B*L << "," << N << "]\n";
        std::cout << "Max difference: " << max_diff << "\n";
    
        if(max_diff<1e-5f){
            std::cout << "LINEAR FORWARD_3D TEST PASSED\n";
        }else{
            std::cout << "LINEAR FORWARD_3D TEST FAILED\n";
        }
    }

    {
        std::cout << "\n--- GPT-2 MLP test ---\n";
        std::size_t B=2;
        std::size_t L=16;
        std::size_t d_model=768;
        std::size_t d_ff=4*d_model;
        std::size_t tile_size=16;
        std::size_t group_size=256;
        tinyml::Tensor<float> X(queue,{B,L,d_model});
        tinyml::Tensor<float> H(queue,{B,L,d_ff});
        tinyml::Tensor<float> H_gelu(queue,{B,L,d_ff});
        tinyml::Tensor<float> Y(queue,{B,L,d_model});
        tinyml::Tensor<float> R(queue,{B,L,d_model});
        tinyml::Linear<float> fc1(queue,d_model,d_ff);
        tinyml::Linear<float> fc2(queue,d_ff,d_model);
        std::vector<float> h_X(B*L*d_model);
        std::vector<float> h_W1(d_model*d_ff);
        std::vector<float> h_b1(d_ff);
        std::vector<float> h_W2(d_ff*d_model);
        std::vector<float> h_b2(d_model);

        for(std::size_t i=0;i<h_X.size();i++){
            h_X[i]=static_cast<float>(static_cast<int>(i%11)-5)*0.1f;
        }
        
        for(std::size_t i=0;i<h_W1.size();i++){
            h_W1[i]=static_cast<float>(static_cast<int>(i%7)-3)*0.01f;
        }
        
        for(std::size_t i=0;i<h_b1.size();i++){
            h_b1[i]=static_cast<float>(static_cast<int>(i%5)-2)*0.01f;
        }
        
        for(std::size_t i=0;i<h_W2.size();i++){
            h_W2[i]=static_cast<float>(static_cast<int>(i%9)-4)*0.01f;
        }
        
        for(std::size_t i=0;i<h_b2.size();i++){
            h_b2[i]=static_cast<float>(static_cast<int>(i%7)-3)*0.01f;
        }

        auto e_X=ops::copy_to_device(queue,X,h_X.data());
        auto e_W1=ops::copy_to_device(queue,fc1.weight(),h_W1.data());
        auto e_b1=ops::copy_to_device(queue,fc1.bias(),h_b1.data());
        auto e_W2=ops::copy_to_device(queue,fc2.weight(),h_W2.data());
        auto e_b2=ops::copy_to_device(queue,fc2.bias(),h_b2.data());
        auto e_fc1=fc1.forward_3d(X,H,group_size,tile_size,{e_X,e_W1,e_b1});
        auto e_gelu=ops::gelu_exact(queue,H,H_gelu,group_size,{e_fc1});
        auto e_fc2=fc2.forward_3d(H_gelu,Y,group_size,tile_size,{e_gelu,e_W2,e_b2});
        auto e_R=ops::add(queue,X,Y,R,group_size,{e_X,e_fc2});
        std::vector<float> h_Y_gpu(B*L*d_model);
        std::vector<float> h_R_gpu(B*L*d_model);
        auto e_Y_host=ops::copy_to_host(queue,h_Y_gpu.data(),Y,{e_fc2});
        auto e_R_host=ops::copy_to_host(queue,h_R_gpu.data(),R,{e_R});
        e_Y_host.wait_and_throw();
        e_R_host.wait_and_throw();
        std::vector<float> h_H_ref(B*L*d_ff,0.0f);

        for(std::size_t row=0;row<B*L;row++){
            for(std::size_t j=0;j<d_ff;j++){
                float value=h_b1[j];

                for(std::size_t k=0;k<d_model;k++){
                    value+=h_X[row*d_model+k]*h_W1[k*d_ff+j];
                }

                h_H_ref[row*d_ff+j]=value;
            }
        }
        std::vector<float> h_H_gelu_ref(B*L*d_ff);

        for(std::size_t i=0;i<h_H_ref.size();i++){
            float x=h_H_ref[i];
            h_H_gelu_ref[i]=0.5f*x*(1.0f+std::erf(x/std::sqrt(2.0f)));
        }

        std::vector<float> h_Y_ref(B*L*d_model,0.0f);

        for(std::size_t row=0;row<B*L;row++){
            for(std::size_t j=0;j<d_model;j++){
                float value=h_b2[j];

                for(std::size_t k=0;k<d_ff;k++){
                    value+=h_H_gelu_ref[row*d_ff+k]*h_W2[k*d_model+j];
                }

                h_Y_ref[row*d_model+j]=value;
            }
        }
        std::vector<float> h_R_ref(B*L*d_model);

        for(std::size_t i=0;i<h_R_ref.size();i++){
            h_R_ref[i]=h_X[i]+h_Y_ref[i];
        }
        float max_diff=0.0f;
        float mean_diff=0.0f;

        for(std::size_t i=0;i<h_Y_gpu.size();i++){
            float diff=std::abs(h_Y_gpu[i]-h_Y_ref[i]);
            max_diff=std::max(max_diff,diff);
            mean_diff+=diff;
        }

        mean_diff/=static_cast<float>(h_Y_gpu.size());

        float max_diff_R=0.0f;
        float mean_diff_R=0.0f;

        for(std::size_t i=0;i<h_R_gpu.size();i++){
            float diff=std::abs(h_R_gpu[i]-h_R_ref[i]);
            max_diff_R=std::max(max_diff_R,diff);
            mean_diff_R+=diff;
        }

        mean_diff_R/=static_cast<float>(h_R_gpu.size());

        std::cout << "MLP mean difference: " << mean_diff << "\n";
        std::cout << "MLP max difference: " << max_diff << "\n";
        std::cout << "MLP residual mean difference: " << mean_diff_R << "\n";
        std::cout << "MLP residual max difference: " << max_diff_R << "\n";

        if(max_diff<5.0e-4f && max_diff_R<5.0e-4f){
            std::cout << "GPT-2 MLP TEST PASSED\n";
        }
        else{
            std::cout << "GPT-2 MLP TEST FAILED\n";
            all_tests_passed=false;
        }
    }

    
    {
        std::cout << "\n=== Neural Network Test ===\n";

        // X [3,2] -> Linear(2,4) -> ReLU -> Linear(4,2)

        tinyml::Linear<float> linear1(queue, 2, 4);
        tinyml::Linear<float> linear2(queue, 4, 2);

        tinyml::Tensor<float> X(queue,  {3,2});
        tinyml::Tensor<float> H1(queue, {3,4});
        tinyml::Tensor<float> H2(queue, {3,4});
        tinyml::Tensor<float> Y(queue,  {3,2});

        std::vector<float> h_X{
            3.2f, 1.5f,
            3.2f, 1.5f,
            3.2f, 1.5f
        };

        std::vector<float> h_W1{
             2.1f,  3.4f, 1.01f, 0.1f,
            -0.1f, -0.3f, 0.88f, 2.7f
        };

        std::vector<float> h_b1{
            0.1f, -0.37f, 0.1f, -7.5f
        };

        std::vector<float> h_W2{
             0.1f,  -0.4f,
             1.31f,  0.91f,
            -0.1f,  -0.3f,
            -0.65f, -2.0f
        };

        std::vector<float> h_b2{
            -0.95f, -9.0f
        };

        sycl::event e_X =
            ops::copy_to_device(
                queue, X, h_X.data());

        sycl::event e_W1 =
            ops::copy_to_device(
                queue, linear1.weight(), h_W1.data());

        sycl::event e_b1 =
            ops::copy_to_device(
                queue, linear1.bias(), h_b1.data());

        sycl::event e_l1 =
            linear1.forward(
                X, H1,
                256, 16,
                {e_X, e_W1, e_b1}
            );

        sycl::event e_relu1 =
            ops::relu(
                queue, H1, H2, 256, {e_l1});

        sycl::event e_W2 =
            ops::copy_to_device(
                queue, linear2.weight(), h_W2.data());

        sycl::event e_b2 =
            ops::copy_to_device(
                queue, linear2.bias(), h_b2.data());

        sycl::event e_l2 =
            linear2.forward(
                H2, Y,
                256, 16,
                {e_relu1, e_W2, e_b2}
            );

        std::vector<float> h_Y(6);

        ops::copy_to_host(
            queue, h_Y.data(), Y, {e_l2}
        ).wait_and_throw();

        // -----------------------------------------------------
        // Independent CPU reference
        // -----------------------------------------------------

        std::vector<float> h_H1(12);
        std::vector<float> h_H2(12);
        std::vector<float> h_reference(6);

        for (std::size_t i = 0; i < 3; i++) {

            for (std::size_t j = 0; j < 4; j++) {

                float value = h_b1[j];

                for (std::size_t k = 0; k < 2; k++) {

                    value +=
                        h_X[i * 2 + k] *
                        h_W1[k * 4 + j];
                }

                h_H1[i * 4 + j] = value;

                h_H2[i * 4 + j] =
                    std::max(0.0f, value);
            }
        }

        for (std::size_t i = 0; i < 3; i++) {

            for (std::size_t j = 0; j < 2; j++) {

                float value = h_b2[j];

                for (std::size_t k = 0; k < 4; k++) {

                    value +=
                        h_H2[i * 4 + k] *
                        h_W2[k * 2 + j];
                }

                h_reference[i * 2 + j] = value;
            }
        }

        // -----------------------------------------------------
        // Numerical validation
        // -----------------------------------------------------

        float max_diff = 0.0f;
        bool finite_results = true;

        for (std::size_t i = 0; i < h_Y.size(); i++) {

            std::cout
                << "GPU: " << h_Y[i]
                << " CPU: " << h_reference[i]
                << "\n";

            if (!std::isfinite(h_Y[i])) {

                finite_results = false;
                continue;
            }

            float diff =
                std::abs(h_Y[i] - h_reference[i]);

            max_diff = std::max(max_diff, diff);
        }

        std::cout
            << "Maximum difference: "
            << max_diff << "\n";

        if (finite_results && max_diff < tolerance) {

            std::cout << "Neural Network test PASSED\n";
        }
        else {

            std::cout << "Neural Network test FAILED\n";

            all_tests_passed = false;
        }
    }

    if(all_tests_passed){
        std::cout << "\nALL NN TESTS PASSED\n";
        return 0;
    }

    std::cout << "\nSOME NN TESTS FAILED\n";
    return 1;
}