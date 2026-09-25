#include <sycl/sycl.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <algorithm>

#include "tinyml/device_buffer.hpp"
#include "tinyml/tensor.hpp"
#include "tinyml/ops.hpp"
#include "tinyml/linear.hpp"


int main()
{
    //sycl::queue queue{sycl::gpu_selector_v};
    sycl::queue queue{sycl::gpu_selector_v,sycl::property::queue::enable_profiling{}};
    //sycl::queue queue{sycl::gpu_selector_v,{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}}};

    std::cout << "Device: "
              << queue.get_device()
                     .get_info<sycl::info::device::name>()
              << "\n";

    const float tolerance = 1.0e-5f;

    bool all_tests_passed = true;

    // =========================================================
    // TEST 1: Neural network
    // =========================================================

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

    // =========================================================
    // TEST 2: Serial Softmax
    // =========================================================

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

    // =========================================================
    // TEST 3: Parallel Softmax
    // =========================================================

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

    // =========================================================
    // TEST 4: Serial vs Parallel Softmax
    // =========================================================

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

    // =========================================================
    // TEST 5: Serial LayerNorm
    // =========================================================

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

    // =========================================================
    // TEST 6: Parallel LayerNorm
    // =========================================================

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

        tinyml::Tensor<float> output_serial(
            queue, {M,N});

        tinyml::Tensor<float> output_parallel(
            queue, {M,N});

        // -----------------------------------------------------
        // Copy input to GPU
        // -----------------------------------------------------

        sycl::event e_input =
            ops::copy_to_device(
                queue,
                input,
                h_in.data()
            );

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

        sycl::event e_parallel =
            ops::layer_norm_parallel(
                queue,
                input,
                output_parallel,
                group_size,
                {e_input}
            );

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

    // =========================================================
    // TEST 7: ReLU
    // =========================================================

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

    // =========================================================
    // TEST 8: Add Bias
    // =========================================================

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

    // =========================================================
    // TEST 9: Matrix Multiplication
    // =========================================================

    {
        std::cout << "\n=== Matrix Multiplication Test ===\n";

        const std::size_t M = 3;
        const std::size_t K = 4;
        const std::size_t N = 2;

        std::vector<float> h_A{
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f
        };

        std::vector<float> h_B{
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
            7.0f, 8.0f
        };

        tinyml::Tensor<float> A(queue, {M,K});
        tinyml::Tensor<float> B(queue, {K,N});
        tinyml::Tensor<float> C(queue, {M,N});

        sycl::event e_A =
            ops::copy_to_device(
                queue, A, h_A.data());

        sycl::event e_B =
            ops::copy_to_device(
                queue, B, h_B.data());

        sycl::event e_matmul =
            ops::matmul(
                queue,
                A,
                B,
                C,
                256,
                {e_A, e_B}
            );

        std::vector<float> h_C(M * N);
        std::vector<float> h_reference(M * N);

        ops::copy_to_host(
            queue, h_C.data(), C, {e_matmul}
        ).wait_and_throw();

        // -----------------------------------------------------
        // CPU reference
        // -----------------------------------------------------

        for (std::size_t i = 0; i < M; i++) {

            for (std::size_t j = 0; j < N; j++) {

                float value = 0.0f;

                for (std::size_t k = 0; k < K; k++) {

                    value +=
                        h_A[i * K + k] *
                        h_B[k * N + j];
                }

                h_reference[i * N + j] = value;
            }
        }

        // -----------------------------------------------------
        // Numerical validation
        // -----------------------------------------------------

        float max_diff = 0.0f;
        bool finite_results = true;

        for (std::size_t i = 0; i < M * N; i++) {

            std::cout << h_C[i] << " ";

            if (!std::isfinite(h_C[i])) {

                finite_results = false;
                continue;
            }

            float diff =
                std::abs(h_C[i] - h_reference[i]);

            max_diff = std::max(max_diff, diff);
        }

        std::cout
            << "\nMaximum difference: "
            << max_diff << "\n";

        if (finite_results && max_diff < tolerance) {

            std::cout << "Matrix Multiplication test PASSED\n";
        }
        else {

            std::cout << "Matrix Multiplication test FAILED\n";

            all_tests_passed = false;
        }
    }

    // =========================================================
    // TEST 10: Tiled Matrix Multiplication
    // =========================================================

    {
        std::cout
            << "\n=== Tiled Matrix Multiplication Test ===\n";

        const std::size_t M = 5;
        const std::size_t K = 7;
        const std::size_t N = 6;

        const std::size_t tile_size = 16;

        std::vector<float> h_A(M * K);
        std::vector<float> h_B(K * N);

        for (std::size_t i = 0; i < M * K; i++) {

            h_A[i] =
                static_cast<float>(i % 11) * 0.1f;
        }

        for (std::size_t i = 0; i < K * N; i++) {

            h_B[i] =
                static_cast<float>(i % 7) * 0.2f;
        }

        tinyml::Tensor<float> A(queue, {M,K});
        tinyml::Tensor<float> B(queue, {K,N});
        tinyml::Tensor<float> C(queue, {M,N});

        sycl::event e_A =
            ops::copy_to_device(
                queue, A, h_A.data());

        sycl::event e_B =
            ops::copy_to_device(
                queue, B, h_B.data());

        sycl::event e_matmul =
            ops::matmul_tiled(
                queue,
                A,
                B,
                C,
                tile_size,
                {e_A, e_B}
            );

        std::vector<float> h_C(M * N);
        std::vector<float> h_reference(M * N);

        ops::copy_to_host(
            queue, h_C.data(), C, {e_matmul}
        ).wait_and_throw();

        // -----------------------------------------------------
        // CPU reference
        // -----------------------------------------------------

        for (std::size_t i = 0; i < M; i++) {

            for (std::size_t j = 0; j < N; j++) {

                float value = 0.0f;

                for (std::size_t k = 0; k < K; k++) {

                    value +=
                        h_A[i * K + k] *
                        h_B[k * N + j];
                }

                h_reference[i * N + j] = value;
            }
        }

        // -----------------------------------------------------
        // Numerical validation
        // -----------------------------------------------------

        float max_diff = 0.0f;
        bool finite_results = true;

        for (std::size_t i = 0; i < M * N; i++) {

            if (!std::isfinite(h_C[i])) {

                finite_results = false;
                continue;
            }

            float diff =
                std::abs(h_C[i] - h_reference[i]);

            max_diff = std::max(max_diff, diff);
        }

        std::cout
            << "Maximum difference: "
            << max_diff << "\n";

        if (finite_results && max_diff < tolerance) {

            std::cout
                << "Tiled Matrix Multiplication test PASSED\n";
        }
        else {

            std::cout
                << "Tiled Matrix Multiplication test FAILED\n";

            all_tests_passed = false;
        }
    }

    // =========================================================
    // TEST 11: Scalar Multiplication
    // =========================================================

    {
        std::cout << "\n=== Scalar Multiplication Test ===\n";

        const std::size_t M = 2;
        const std::size_t N = 4;

        const float scalar = 2.5f;

        std::vector<float> h_in{
            1.0f, -2.0f, 3.0f, -4.0f,
            5.0f, -6.0f, 7.0f, -8.0f
        };

        tinyml::Tensor<float> input(queue, {M,N});
        tinyml::Tensor<float> output(queue, {M,N});

        sycl::event e_input =
            ops::copy_to_device(
                queue, input, h_in.data());

        sycl::event e_mul =
            ops::scalar_mul(
                queue,
                input,
                scalar,
                output,
                256,
                {e_input}
            );

        std::vector<float> h_out(M * N);
        std::vector<float> h_reference(M * N);

        ops::copy_to_host(
            queue, h_out.data(), output, {e_mul}
        ).wait_and_throw();

        // -----------------------------------------------------
        // CPU reference
        // -----------------------------------------------------

        for (std::size_t i = 0; i < M * N; i++) {

            h_reference[i] = h_in[i] * scalar;
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

            std::cout
                << "Scalar Multiplication test PASSED\n";
        }
        else {

            std::cout
                << "Scalar Multiplication test FAILED\n";

            all_tests_passed = false;
        }
    }

    // =========================================================
    // TEST 12: Element-wise Addition
    // =========================================================

    {
        std::cout << "\n=== Element-wise Addition Test ===\n";

        const std::size_t M = 2;
        const std::size_t N = 4;

        std::vector<float> h_A{
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f
        };

        std::vector<float> h_B{
            8.0f, 7.0f, 6.0f, 5.0f,
            4.0f, 3.0f, 2.0f, 1.0f
        };

        tinyml::Tensor<float> A(queue, {M,N});
        tinyml::Tensor<float> B(queue, {M,N});
        tinyml::Tensor<float> C(queue, {M,N});

        sycl::event e_A =
            ops::copy_to_device(
                queue, A, h_A.data());

        sycl::event e_B =
            ops::copy_to_device(
                queue, B, h_B.data());

        sycl::event e_add =
            ops::add(
                queue,
                A,
                B,
                C,
                256,
                {e_A, e_B}
            );

        std::vector<float> h_C(M * N);
        std::vector<float> h_reference(M * N);

        ops::copy_to_host(
            queue, h_C.data(), C, {e_add}
        ).wait_and_throw();

        // -----------------------------------------------------
        // CPU reference
        // -----------------------------------------------------

        for (std::size_t i = 0; i < M * N; i++) {

            h_reference[i] = h_A[i] + h_B[i];
        }

        // -----------------------------------------------------
        // Numerical validation
        // -----------------------------------------------------

        float max_diff = 0.0f;
        bool finite_results = true;

        for (std::size_t i = 0; i < M * N; i++) {

            std::cout << h_C[i] << " ";

            if (!std::isfinite(h_C[i])) {

                finite_results = false;
                continue;
            }

            float diff =
                std::abs(h_C[i] - h_reference[i]);

            max_diff = std::max(max_diff, diff);
        }

        std::cout
            << "\nMaximum difference: "
            << max_diff << "\n";

        if (finite_results && max_diff < tolerance) {

            std::cout << "Addition test PASSED\n";
        }
        else {

            std::cout << "Addition test FAILED\n";

            all_tests_passed = false;
        }
    }

    // =========================================================
    // TEST 13: Element-wise Subtraction
    // =========================================================

    {
        std::cout
            << "\n=== Element-wise Subtraction Test ===\n";

        const std::size_t M = 2;
        const std::size_t N = 4;

        std::vector<float> h_A{
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f
        };

        std::vector<float> h_B{
            8.0f, 7.0f, 6.0f, 5.0f,
            4.0f, 3.0f, 2.0f, 1.0f
        };

        tinyml::Tensor<float> A(queue, {M,N});
        tinyml::Tensor<float> B(queue, {M,N});
        tinyml::Tensor<float> C(queue, {M,N});

        sycl::event e_A =
            ops::copy_to_device(
                queue, A, h_A.data());

        sycl::event e_B =
            ops::copy_to_device(
                queue, B, h_B.data());

        sycl::event e_sub =
            ops::sub(
                queue,
                A,
                B,
                C,
                256,
                {e_A, e_B}
            );

        std::vector<float> h_C(M * N);
        std::vector<float> h_reference(M * N);

        ops::copy_to_host(
            queue, h_C.data(), C, {e_sub}
        ).wait_and_throw();

        // -----------------------------------------------------
        // CPU reference
        // -----------------------------------------------------

        for (std::size_t i = 0; i < M * N; i++) {

            h_reference[i] = h_A[i] - h_B[i];
        }

        // -----------------------------------------------------
        // Numerical validation
        // -----------------------------------------------------

        float max_diff = 0.0f;
        bool finite_results = true;

        for (std::size_t i = 0; i < M * N; i++) {

            std::cout << h_C[i] << " ";

            if (!std::isfinite(h_C[i])) {

                finite_results = false;
                continue;
            }

            float diff =
                std::abs(h_C[i] - h_reference[i]);

            max_diff = std::max(max_diff, diff);
        }

        std::cout
            << "\nMaximum difference: "
            << max_diff << "\n";

        if (finite_results && max_diff < tolerance) {

            std::cout << "Subtraction test PASSED\n";
        }
        else {

            std::cout << "Subtraction test FAILED\n";

            all_tests_passed = false;
        }
    }

    // =========================================================
    // TEST 14: Element-wise Multiplication
    // =========================================================

    {
        std::cout
            << "\n=== Element-wise Multiplication Test ===\n";

        const std::size_t M = 2;
        const std::size_t N = 4;

        std::vector<float> h_A{
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f
        };

        std::vector<float> h_B{
            8.0f, 7.0f, 6.0f, 5.0f,
            4.0f, 3.0f, 2.0f, 1.0f
        };

        tinyml::Tensor<float> A(queue, {M,N});
        tinyml::Tensor<float> B(queue, {M,N});
        tinyml::Tensor<float> C(queue, {M,N});

        sycl::event e_A =
            ops::copy_to_device(
                queue, A, h_A.data());

        sycl::event e_B =
            ops::copy_to_device(
                queue, B, h_B.data());

        sycl::event e_mul =
            ops::mul(
                queue,
                A,
                B,
                C,
                256,
                {e_A, e_B}
            );

        std::vector<float> h_C(M * N);
        std::vector<float> h_reference(M * N);

        ops::copy_to_host(
            queue, h_C.data(), C, {e_mul}
        ).wait_and_throw();

        // -----------------------------------------------------
        // CPU reference
        // -----------------------------------------------------

        for (std::size_t i = 0; i < M * N; i++) {

            h_reference[i] = h_A[i] * h_B[i];
        }

        // -----------------------------------------------------
        // Numerical validation
        // -----------------------------------------------------

        float max_diff = 0.0f;
        bool finite_results = true;

        for (std::size_t i = 0; i < M * N; i++) {

            std::cout << h_C[i] << " ";

            if (!std::isfinite(h_C[i])) {

                finite_results = false;
                continue;
            }

            float diff =
                std::abs(h_C[i] - h_reference[i]);

            max_diff = std::max(max_diff, diff);
        }

        std::cout
            << "\nMaximum difference: "
            << max_diff << "\n";

        if (finite_results && max_diff < tolerance) {

            std::cout << "Multiplication test PASSED\n";
        }
        else {

            std::cout << "Multiplication test FAILED\n";

            all_tests_passed = false;
        }
    }

    // =========================================================
    // TEST 15: Element-wise Division
    // =========================================================

    {
        std::cout
            << "\n=== Element-wise Division Test ===\n";

        const std::size_t M = 2;
        const std::size_t N = 4;

        std::vector<float> h_A{
            2.0f, 4.0f, 6.0f, 8.0f,
            10.0f, 12.0f, 14.0f, 16.0f
        };

        std::vector<float> h_B{
            2.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 3.0f, 7.0f, 8.0f
        };

        tinyml::Tensor<float> A(queue, {M,N});
        tinyml::Tensor<float> B(queue, {M,N});
        tinyml::Tensor<float> C(queue, {M,N});

        sycl::event e_A =
            ops::copy_to_device(
                queue, A, h_A.data());

        sycl::event e_B =
            ops::copy_to_device(
                queue, B, h_B.data());

        sycl::event e_div =
            ops::div(
                queue,
                A,
                B,
                C,
                256,
                {e_A, e_B}
            );

        std::vector<float> h_C(M * N);
        std::vector<float> h_reference(M * N);

        ops::copy_to_host(
            queue, h_C.data(), C, {e_div}
        ).wait_and_throw();

        // -----------------------------------------------------
        // CPU reference
        // -----------------------------------------------------

        for (std::size_t i = 0; i < M * N; i++) {

            h_reference[i] = h_A[i] / h_B[i];
        }

        // -----------------------------------------------------
        // Numerical validation
        // -----------------------------------------------------

        float max_diff = 0.0f;
        bool finite_results = true;

        for (std::size_t i = 0; i < M * N; i++) {

            std::cout << h_C[i] << " ";

            if (!std::isfinite(h_C[i])) {

                finite_results = false;
                continue;
            }

            float diff =
                std::abs(h_C[i] - h_reference[i]);

            max_diff = std::max(max_diff, diff);
        }

        std::cout
            << "\nMaximum difference: "
            << max_diff << "\n";

        if (finite_results && max_diff < tolerance) {

            std::cout << "Division test PASSED\n";
        }
        else {

            std::cout << "Division test FAILED\n";

            all_tests_passed = false;
        }
    }

    // =========================================================
    // TEST 16: Fill
    // =========================================================

    {
        std::cout << "\n=== Fill Test ===\n";

        const std::size_t M = 3;
        const std::size_t N = 4;

        const float value = 3.5f;

        tinyml::Tensor<float> output(queue, {M,N});

        sycl::event e_fill =
            ops::fill(
                queue,
                output,
                value,
                256
            );

        std::vector<float> h_out(M * N);

        ops::copy_to_host(
            queue, h_out.data(), output, {e_fill}
        ).wait_and_throw();

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
                std::abs(h_out[i] - value);

            max_diff = std::max(max_diff, diff);
        }

        std::cout
            << "\nMaximum difference: "
            << max_diff << "\n";

        if (finite_results && max_diff < tolerance) {

            std::cout << "Fill test PASSED\n";
        }
        else {

            std::cout << "Fill test FAILED\n";

            all_tests_passed = false;
        }
    }

    
    // =========================================================
    // TEST 8: Exact GELU
    // =========================================================

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

        
    // =========================================================
    // TEST 9: Approximate GELU
    // =========================================================

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

    
    // =========================================================
    // gelu execution time comparison
    // =========================================================
    {
        std::vector<float> h_gelu_input(16*1048576);
        std::vector<float> h_gelu_output_exact(16*1048576);
        std::vector<float> h_gelu_output_approx(16*1048576);
        std::size_t group_size=256,warm_up=10, measure=101;
        for(std::size_t i=0;i<h_gelu_input.size();i++){
            h_gelu_input[i]=static_cast<float>(i%1001)*0.01f-5.0f;
        }
        tinyml::Tensor<float> input(queue,{16*1048576}),output_exact(queue,{16*1048576}),output_approx(queue,{16*1048576});
        ops::copy_to_device(queue, input,h_gelu_input.data()).wait_and_throw();
        for(std::size_t i=0;i<warm_up; i++){
            ops::gelu_exact(queue,input,output_exact,group_size).wait_and_throw();
            ops::gelu_approx(queue,input,output_approx,group_size).wait_and_throw();
        }
        std::vector<sycl::event> e_e,e_a;
        e_e.reserve(measure);
        e_a.reserve(measure);
        sycl::event e;
        e=ops::gelu_exact(queue,input,output_exact,group_size);
        e_e.push_back(e);
        for(size_t i=1;i<measure; i++){
             e=ops::gelu_exact(queue,input,output_exact,group_size,{e});
             e_e.push_back(e);
        }
        e.wait_and_throw();
        double t_e=0.0;
        for(size_t i=0;i<measure;i++){
            auto start=e_e[i].get_profiling_info<sycl::info::event_profiling::command_start>();
            auto end=e_e[i].get_profiling_info<sycl::info::event_profiling::command_end>();
            t_e+=end-start;
        }
        t_e/=1.0e+6*static_cast<double>(measure);
        std::cout << "Exact GELU average execution time: " << t_e << " ms\n";

        e=ops::gelu_approx(queue,input,output_approx,group_size);
        e_a.push_back(e);
        for(size_t i=1;i<measure; i++){
             e=ops::gelu_approx(queue,input,output_approx,group_size,{e});
             e_a.push_back(e);
        }
        e.wait_and_throw();
        double t_a=0.0;
        for(size_t i=0;i<measure;i++){
            auto start=e_a[i].get_profiling_info<sycl::info::event_profiling::command_start>();
            auto end=e_a[i].get_profiling_info<sycl::info::event_profiling::command_end>();
            t_a+=end-start;
        }
        t_a/=1.0e+6*static_cast<double>(measure);
        std::cout << "Approx GELU average execution time: " << t_a << " ms\n";
        ops::copy_to_host(queue,h_gelu_output_exact.data(),output_exact).wait_and_throw();
        ops::copy_to_host(queue,h_gelu_output_approx.data(),output_approx).wait_and_throw();
        float max_diff=0.0f;
        float sum_f=0.0f;
        double sum_d=0.0;
        for(size_t i=0;i<h_gelu_input.size();i++){
            float diff=std::abs(h_gelu_output_approx[i]-h_gelu_output_exact[i]);
            max_diff=std::max(max_diff,diff);
            //max_diff=(max_diff>std::abs(h_gelu_output_approx[i]-h_gelu_output_exact[i])?std::abs(h_gelu_output_approx[i]-h_gelu_output_exact[i]):max_diff);
            sum_f+=diff;
            sum_d+=diff;
        }
        std::cout << "max difference between exact and approx gelu "<< max_diff << std::endl;
        std::cout << "Sum of differences in single " << sum_f/h_gelu_input.size() << " and double precision " << sum_d/h_gelu_input.size() << std::endl;
    }

        // =========================================================
    // Tiled MatMul Performance Test
    // =========================================================
    {
        std::cout << "\n=== Tiled MatMul Performance Test ===\n";

        const std::size_t M = 1024, K = 1024, N = 1024;
        const std::size_t tile_size = 16;
        const std::size_t warm_up = 10, measure = 100;

        std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_reference(M * N);

        for (std::size_t i = 0; i < M * K; i++) h_A[i] = static_cast<float>(i % 17) * 0.01f;
        for (std::size_t i = 0; i < K * N; i++) h_B[i] = static_cast<float>(i % 13) * 0.01f;

        tinyml::Tensor<float> A(queue, {M,K}), B(queue, {K,N}), C(queue, {M,N});

        ops::copy_to_device(queue, A, h_A.data()).wait_and_throw();
        ops::copy_to_device(queue, B, h_B.data()).wait_and_throw();

        // Warm-up
        for (std::size_t i = 0; i < warm_up; i++) {
            ops::matmul_tiled(queue, A, B, C, tile_size).wait_and_throw();
        }

        // GPU execution time
        std::vector<sycl::event> events;
        events.reserve(measure);

        sycl::event e = ops::matmul_tiled(queue, A, B, C, tile_size);
        events.push_back(e);

        for (std::size_t i = 1; i < measure; i++) {
            e = ops::matmul_tiled(queue, A, B, C, tile_size, {e});
            events.push_back(e);
        }

        e.wait_and_throw();

        double total_time = 0.0;

        for (const auto& event : events) {
            auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
            total_time += static_cast<double>(end - start);
        }

        double average_ms = total_time / (1.0e6 * static_cast<double>(measure));
        double gflops = (2.0 * static_cast<double>(M) * K * N) / (average_ms * 1.0e6);

        std::cout << "Matrix dimensions: " << M << " x " << K << " x " << N << "\n";
        std::cout << "Tile size: " << tile_size << "\n";
        std::cout << "Average execution time: " << average_ms << " ms\n";
        std::cout << "Performance: " << gflops << " GFLOP/s\n";

        // Copy result to host
        ops::copy_to_host(queue, h_C.data(), C, {e}).wait_and_throw();

        // Independent CPU reference
        for (std::size_t i = 0; i < M; i++) {
            for (std::size_t j = 0; j < N; j++) {
                double value = 0.0;
                for (std::size_t k = 0; k < K; k++) value += static_cast<double>(h_A[i * K + k]) * h_B[k * N + j];
                h_reference[i * N + j] = static_cast<float>(value);
            }
        }

        // Numerical validation
        float max_diff = 0.0f;
        bool finite_results = true;

        for (std::size_t i = 0; i < M * N; i++) {
            if (!std::isfinite(h_C[i])) {
                finite_results = false;
                continue;
            }
            max_diff = std::max(max_diff, std::abs(h_C[i] - h_reference[i]));
        }

        std::cout << "Maximum difference: " << max_diff << "\n";

        if (finite_results && max_diff < 1.0e-4f) {
            std::cout << "Tiled MatMul Performance test PASSED\n";
        }
        else {
            std::cout << "Tiled MatMul Performance test FAILED\n";
            all_tests_passed = false;
        }
    }

    {
        std::cout << "\n=== Tiled Transposed MatMul Test ===\n";
    
        const std::size_t M = 3;
        const std::size_t K = 5;
        const std::size_t N = 4;
        const std::size_t tile_size = 2;
    
        tinyml::Tensor<float> A(queue,{M,K});
        tinyml::Tensor<float> B(queue,{N,K});
        tinyml::Tensor<float> C(queue,{M,N});
    
        std::vector<float> h_A = {
             1,  2,  3,  4,  5,
             6,  7,  8,  9, 10,
            11, 12, 13, 14, 15
        };
    
        std::vector<float> h_B = {
             1,  2,  1,  2,  1,
             2,  1,  2,  1,  2,
             1,  1,  1,  1,  1,
             2,  2,  2,  2,  2
        };
    
        std::vector<float> h_C(M*N);
        std::vector<float> h_ref(M*N,0.0f);
    
        auto e1 = ops::copy_to_device(queue,A,h_A.data());
        auto e2 = ops::copy_to_device(queue,B,h_B.data());
    
        auto e3 = ops::matmul_tiled_transposed(queue,A,B,C,tile_size,{e1,e2});
        auto e4 = ops::copy_to_host(queue,h_C.data(),C,{e3});
        e4.wait();
    
        // CPU reference: C[i,j] = sum_k A[i,k] * B[j,k]
        for(std::size_t i=0; i<M; i++){
            for(std::size_t j=0; j<N; j++){
                for(std::size_t k=0; k<K; k++){
                    h_ref[i*N+j] += h_A[i*K+k]*h_B[j*K+k];
                }
            }
        }
    
        float max_diff=0.0f;
    
        std::cout << "GPU result:\n";
        for(std::size_t i=0; i<M; i++){
            for(std::size_t j=0; j<N; j++){
                std::cout << h_C[i*N+j] << " ";
                max_diff=std::max(max_diff,std::abs(h_C[i*N+j]-h_ref[i*N+j]));
            }
            std::cout << "\n";
        }
    
        std::cout << "CPU reference:\n";
        for(std::size_t i=0; i<M; i++){
            for(std::size_t j=0; j<N; j++){
                std::cout << h_ref[i*N+j] << " ";
            }
            std::cout << "\n";
        }
    
        std::cout << "Maximum difference: " << max_diff << "\n";
    
        if(max_diff < 1e-5f){
            std::cout << "Tiled Transposed MatMul test PASSED\n";
        }
        else{
            std::cout << "Tiled Transposed MatMul test FAILED\n";
        }
    }

    
{
    std::cout << "\n=== Scaled MatMul + Softmax Test ===\n";

    const std::size_t M=4, K=8, N=4;
    const std::size_t tile_size=2, group_size=4;

    std::vector<float> h_Q={
        1,2,3,4,5,6,7,8,
        2,1,2,1,2,1,2,1,
        1,1,1,1,1,1,1,1,
        8,7,6,5,4,3,2,1
    };

    std::vector<float> h_K={
        1,0,1,0,1,0,1,0,
        0,1,0,1,0,1,0,1,
        1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2
    };

    std::vector<float> h_scores(M*N);
    std::vector<float> h_weights(M*N);
    std::vector<float> h_reference(M*N,0.0f);
    std::vector<float> h_reference_weights(M*N,0.0f);

    tinyml::Tensor<float> Q(queue,{M,K});
    tinyml::Tensor<float> K_tensor(queue,{N,K});
    tinyml::Tensor<float> scores(queue,{M,N});
    tinyml::Tensor<float> weights(queue,{M,N});

    // GPU: scaled QK^T followed by row-wise softmax

    auto e1=ops::copy_to_device(queue,Q,h_Q.data());
    auto e2=ops::copy_to_device(queue,K_tensor,h_K.data());

    auto e3=ops::matmul_tiled_transposed_scaled(queue,Q,K_tensor,scores,tile_size,{e1,e2});
    auto e4=ops::softmax_parallel(queue,scores,weights,group_size,{e3});

    auto e5=ops::copy_to_host(queue,h_scores.data(),scores,{e3});
    auto e6=ops::copy_to_host(queue,h_weights.data(),weights,{e4});

    e5.wait();
    e6.wait();

    // CPU reference: scaled QK^T

    float scale=1.0f/std::sqrt(static_cast<float>(K));

    for(std::size_t i=0;i<M;i++){
        for(std::size_t j=0;j<N;j++){
            for(std::size_t k=0;k<K;k++){
                h_reference[i*N+j]+=h_Q[i*K+k]*h_K[j*K+k];
            }
            h_reference[i*N+j]*=scale;
        }
    }

    // CPU reference: row-wise softmax

    for(std::size_t i=0;i<M;i++){

        float row_max=h_reference[i*N];

        for(std::size_t j=1;j<N;j++){
            row_max=std::max(row_max,h_reference[i*N+j]);
        }

        float row_sum=0.0f;

        for(std::size_t j=0;j<N;j++){
            float value=std::exp(h_reference[i*N+j]-row_max);
            h_reference_weights[i*N+j]=value;
            row_sum+=value;
        }

        for(std::size_t j=0;j<N;j++){
            h_reference_weights[i*N+j]/=row_sum;
        }
    }

    // Compare scaled scores

    float max_scores_diff=0.0f;

    for(std::size_t i=0;i<M*N;i++){
        max_scores_diff=std::max(max_scores_diff,std::abs(h_scores[i]-h_reference[i]));
    }

    std::cout << "\nScaled scores maximum difference: " << max_scores_diff << "\n";

    // Compare softmax weights

    float max_weights_diff=0.0f;
    bool row_sums_pass=true;

    std::cout << "\nGPU attention weights:\n";

    for(std::size_t i=0;i<M;i++){

        float row_sum=0.0f;

        for(std::size_t j=0;j<N;j++){
            float gpu=h_weights[i*N+j];
            float cpu=h_reference_weights[i*N+j];

            std::cout << gpu << " ";

            row_sum+=gpu;
            max_weights_diff=std::max(max_weights_diff,std::abs(gpu-cpu));
        }

        std::cout << " | Row sum: " << row_sum << "\n";

        if(!std::isfinite(row_sum) || std::abs(row_sum-1.0f)>1e-5f){
            row_sums_pass=false;
        }
    }

    std::cout << "\nCPU attention weights:\n";

    for(std::size_t i=0;i<M;i++){
        for(std::size_t j=0;j<N;j++){
            std::cout << h_reference_weights[i*N+j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nMaximum scores difference: " << max_scores_diff << "\n";
    std::cout << "Maximum weights difference: " << max_weights_diff << "\n";

    if(std::isfinite(max_scores_diff) && std::isfinite(max_weights_diff) &&
       max_scores_diff<1e-5f && max_weights_diff<1e-5f && row_sums_pass){
        std::cout << "Scaled MatMul + Softmax test PASSED\n";
    }
    else{
        std::cout << "Scaled MatMul + Softmax test FAILED\n";
    }
}
     
{
    std::cout << "\n=== Single-Head Scaled Dot-Product Attention Test ===\n";

    const std::size_t M=3, N=4, K=8, Dv=5;
    const std::size_t tile_size=2, group_size=4;

    // Q [M,K], K_tensor [N,K], V [N,Dv]

    std::vector<float> h_Q={
        1,2,3,4,5,6,7,8,
        2,1,2,1,2,1,2,1,
        1,1,1,1,1,1,1,1
    };

    std::vector<float> h_K={
        1,0,1,0,1,0,1,0,
        0,1,0,1,0,1,0,1,
        1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2
    };

    std::vector<float> h_V={
        1,2,3,4,5,
        2,3,4,5,6,
        3,4,5,6,7,
        4,5,6,7,8
    };

    // Host output buffers

    std::vector<float> h_scores(M*N);
    std::vector<float> h_weights(M*N);
    std::vector<float> h_output(M*Dv);

    // CPU reference buffers

    std::vector<float> ref_scores(M*N,0.0f);
    std::vector<float> ref_weights(M*N,0.0f);
    std::vector<float> ref_output(M*Dv,0.0f);

    // Device tensors

    tinyml::Tensor<float> Q(queue,{M,K});
    tinyml::Tensor<float> K_tensor(queue,{N,K});
    tinyml::Tensor<float> V(queue,{N,Dv});

    tinyml::Tensor<float> scores(queue,{M,N});
    tinyml::Tensor<float> weights(queue,{M,N});
    tinyml::Tensor<float> output(queue,{M,Dv});

    // --------------------------------------------------
    // GPU: Copy Q, K and V to device
    // --------------------------------------------------

    auto e1=ops::copy_to_device(queue,Q,h_Q.data());
    auto e2=ops::copy_to_device(queue,K_tensor,h_K.data());
    auto e3=ops::copy_to_device(queue,V,h_V.data());

    // --------------------------------------------------
    // GPU: Step 1 - Scaled QK^T
    // --------------------------------------------------

    auto e4=ops::matmul_tiled_transposed_scaled(
        queue,Q,K_tensor,scores,tile_size,{e1,e2}
    );

    // --------------------------------------------------
    // GPU: Step 2 - Row-wise softmax
    // --------------------------------------------------

    auto e5=ops::softmax_parallel(
        queue,scores,weights,group_size,{e4}
    );

    // --------------------------------------------------
    // GPU: Step 3 - Attention weights multiplied by V
    // --------------------------------------------------

    auto e6=ops::matmul_tiled(
        queue,weights,V,output,tile_size,{e5,e3}
    );

    // --------------------------------------------------
    // Copy intermediate and final results to host
    // --------------------------------------------------

    auto e7=ops::copy_to_host(queue,h_scores.data(),scores,{e4});
    auto e8=ops::copy_to_host(queue,h_weights.data(),weights,{e5});
    auto e9=ops::copy_to_host(queue,h_output.data(),output,{e6});

    e7.wait();
    e8.wait();
    e9.wait();

    // --------------------------------------------------
    // CPU reference: Step 1 - Scaled QK^T
    // --------------------------------------------------

    float scale=1.0f/std::sqrt(static_cast<float>(K));

    for(std::size_t i=0;i<M;i++){
        for(std::size_t j=0;j<N;j++){
            for(std::size_t k=0;k<K;k++){
                ref_scores[i*N+j]+=h_Q[i*K+k]*h_K[j*K+k];
            }
            ref_scores[i*N+j]*=scale;
        }
    }

    // --------------------------------------------------
    // CPU reference: Step 2 - Row-wise softmax
    // --------------------------------------------------

    for(std::size_t i=0;i<M;i++){

        float row_max=ref_scores[i*N];

        for(std::size_t j=1;j<N;j++){
            row_max=std::max(row_max,ref_scores[i*N+j]);
        }

        float row_sum=0.0f;

        for(std::size_t j=0;j<N;j++){
            float value=std::exp(ref_scores[i*N+j]-row_max);
            ref_weights[i*N+j]=value;
            row_sum+=value;
        }

        for(std::size_t j=0;j<N;j++){
            ref_weights[i*N+j]/=row_sum;
        }
    }

    // --------------------------------------------------
    // CPU reference: Step 3 - Attention weights * V
    // --------------------------------------------------

    for(std::size_t i=0;i<M;i++){
        for(std::size_t j=0;j<Dv;j++){
            for(std::size_t k=0;k<N;k++){
                ref_output[i*Dv+j]+=ref_weights[i*N+k]*h_V[k*Dv+j];
            }
        }
    }

    // --------------------------------------------------
    // Compare GPU and CPU results
    // --------------------------------------------------

    float max_scores_diff=0.0f;
    float max_weights_diff=0.0f;
    float max_output_diff=0.0f;

    for(std::size_t i=0;i<M*N;i++){
        max_scores_diff=std::max(max_scores_diff,std::abs(h_scores[i]-ref_scores[i]));
        max_weights_diff=std::max(max_weights_diff,std::abs(h_weights[i]-ref_weights[i]));
    }

    for(std::size_t i=0;i<M*Dv;i++){
        max_output_diff=std::max(max_output_diff,std::abs(h_output[i]-ref_output[i]));
    }

    // --------------------------------------------------
    // Print attention weights and verify row sums
    // --------------------------------------------------

    bool row_sums_pass=true;

    std::cout << "\nGPU attention weights:\n";

    for(std::size_t i=0;i<M;i++){

        float row_sum=0.0f;

        for(std::size_t j=0;j<N;j++){
            float value=h_weights[i*N+j];
            std::cout << value << " ";
            row_sum+=value;
        }

        std::cout << " | Row sum: " << row_sum << "\n";

        if(!std::isfinite(row_sum) || std::abs(row_sum-1.0f)>1e-5f){
            row_sums_pass=false;
        }
    }

    // --------------------------------------------------
    // Print final GPU output
    // --------------------------------------------------

    std::cout << "\nGPU attention output:\n";

    for(std::size_t i=0;i<M;i++){
        for(std::size_t j=0;j<Dv;j++){
            std::cout << h_output[i*Dv+j] << " ";
        }
        std::cout << "\n";
    }

    // --------------------------------------------------
    // Print CPU reference output
    // --------------------------------------------------

    std::cout << "\nCPU reference output:\n";

    for(std::size_t i=0;i<M;i++){
        for(std::size_t j=0;j<Dv;j++){
            std::cout << ref_output[i*Dv+j] << " ";
        }
        std::cout << "\n";
    }

    // --------------------------------------------------
    // Report errors
    // --------------------------------------------------

    std::cout << "\nMaximum scores difference: " << max_scores_diff << "\n";
    std::cout << "Maximum weights difference: " << max_weights_diff << "\n";
    std::cout << "Maximum output difference: " << max_output_diff << "\n";

    if(std::isfinite(max_scores_diff) &&
       std::isfinite(max_weights_diff) &&
       std::isfinite(max_output_diff) &&
       max_scores_diff<1e-5f &&
       max_weights_diff<1e-5f &&
       max_output_diff<1e-5f &&
       row_sums_pass){

        std::cout << "Single-Head Attention test PASSED\n";
    }
    else{
        std::cout << "Single-Head Attention test FAILED\n";
    }
}


    // =========================================================
    // Final test summary
    // =========================================================

    std::cout
        << "\n========================================\n";

    if (all_tests_passed) {
        std::cout << "ALL TESTS PASSED\n";
    }
    else {

        std::cout << "SOME TESTS FAILED\n";
    }

    std::cout
        << "========================================\n";

    return all_tests_passed ? 0 : 1;
}