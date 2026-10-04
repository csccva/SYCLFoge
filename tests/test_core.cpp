#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "tinyml/device_buffer.hpp"
#include "tinyml/tensor.hpp"
#include "tinyml/ops.hpp"

int main()
{
    sycl::queue queue{sycl::gpu_selector_v};

    bool all_tests_passed=true;
    const float tolerance=1.0e-5f;

    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";

    // Tests will go here.

    
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
    
    if(all_tests_passed){
        std::cout << "\nALL CORE TESTS PASSED\n";
        return 0;
    }

    std::cout << "\nSOME CORE TESTS FAILED\n";
    return 1;
}