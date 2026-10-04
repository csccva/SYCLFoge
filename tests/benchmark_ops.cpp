#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "tinyml/device_buffer.hpp"
#include "tinyml/tensor.hpp"
#include "tinyml/ops.hpp"




void test_attention_max_size(sycl::queue& queue)
{
    const std::size_t K=128;
    const std::size_t tile_size=16;

    std::vector<std::size_t> sizes={12288,16384,24576,32768,49152,65536};

    std::cout << "\n=== Attention Maximum Size Test ===\n";
    std::cout << "K=" << K << ", tile=" << tile_size << "\n";

    // ============================================================
    // UNFUSED
    // ============================================================

    std::cout << "\n--- UNFUSED ---\n";

    for(std::size_t N:sizes){
        std::size_t M=N;

        double intermediate_gib=2.0*static_cast<double>(M)*static_cast<double>(N)*sizeof(float)/(1024.0*1024.0*1024.0);

        std::cout << "\nSize " << N << " x " << N << "\n";
        std::cout << "scores + weights = " << intermediate_gib << " GiB\n";

        try{
            tinyml::Tensor<float> Q(queue,{M,K});
            tinyml::Tensor<float> K_tensor(queue,{N,K});
            tinyml::Tensor<float> V(queue,{N,K});
            tinyml::Tensor<float> scores(queue,{M,N});
            tinyml::Tensor<float> weights(queue,{M,N});
            tinyml::Tensor<float> output(queue,{M,K});

            auto e1=ops::fill(queue,Q,0.01f,256);
            auto e2=ops::fill(queue,K_tensor,0.02f,256);
            auto e3=ops::fill(queue,V,0.03f,256);

            auto e4=ops::matmul_tiled_transposed_scaled(queue,Q,K_tensor,scores,tile_size,{e1,e2});
            auto e5=ops::softmax_parallel(queue,scores,weights,256,{e4});
            auto e6=ops::matmul_tiled(queue,weights,V,output,tile_size,{e5,e3});

            e6.wait_and_throw();

            std::cout << "SUCCESS\n";
        }
        catch(const sycl::exception& e){
            std::cout << "FAILED\n";
            std::cout << "SYCL error: " << e.what() << "\n";
            break;
        }
        catch(const std::exception& e){
            std::cout << "FAILED\n";
            std::cout << "Error: " << e.what() << "\n";
            break;
        }
    }

    // Make sure everything from the unfused test is finished
    // and its tensors have gone out of scope before testing fused.
    queue.wait_and_throw();

    // ============================================================
    // FUSED
    // ============================================================

    std::cout << "\n--- FUSED ---\n";

    for(std::size_t N:sizes){
        std::size_t M=N;

        std::cout << "\nSize " << N << " x " << N << "\n";

        try{
            tinyml::Tensor<float> Q(queue,{M,K});
            tinyml::Tensor<float> K_tensor(queue,{N,K});
            tinyml::Tensor<float> V(queue,{N,K});
            tinyml::Tensor<float> output(queue,{M,K});

            auto e1=ops::fill(queue,Q,0.01f,256);
            auto e2=ops::fill(queue,K_tensor,0.02f,256);
            auto e3=ops::fill(queue,V,0.03f,256);

            auto e4=ops::attention_scores_online(queue,Q,K_tensor,V,output,tile_size,{e1,e2,e3});

            e4.wait_and_throw();

            std::cout << "SUCCESS\n";
        }
        catch(const sycl::exception& e){
            std::cout << "FAILED\n";
            std::cout << "SYCL error: " << e.what() << "\n";
            break;
        }
        catch(const std::exception& e){
            std::cout << "FAILED\n";
            std::cout << "Error: " << e.what() << "\n";
            break;
        }
    }

    std::cout << "\n=== Maximum Size Test Finished ===\n";
}

int main()
{
    sycl::queue queue{sycl::gpu_selector_v,sycl::property::queue::enable_profiling{}};

    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
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
            //all_tests_passed = false;
        }
    }
    
    {
        std::cout << "\n=== Attention Performance Benchmark ===\n";

        const std::size_t M=127, N=259, K=128;
        const std::size_t tile_size=16;
        const std::size_t group_size=16;
        const std::size_t warmup=10;
        const std::size_t iterations=100;

        std::vector<float> h_Q(M*K);
        std::vector<float> h_K(N*K);
        std::vector<float> h_V(N*K);

        for(std::size_t i=0;i<M*K;i++){
            h_Q[i]=static_cast<float>(static_cast<int>(i%17)-8)/8.0f;
        }

        for(std::size_t i=0;i<N*K;i++){
            h_K[i]=static_cast<float>(static_cast<int>(i%13)-6)/6.0f;
            h_V[i]=static_cast<float>(static_cast<int>(i%19)-9)/9.0f;
        }

        tinyml::Tensor<float> Q(queue,{M,K});
        tinyml::Tensor<float> K_tensor(queue,{N,K});
        tinyml::Tensor<float> V(queue,{N,K});

        tinyml::Tensor<float> scores(queue,{M,N});
        tinyml::Tensor<float> weights(queue,{M,N});
        tinyml::Tensor<float> output_unfused(queue,{M,K});
        tinyml::Tensor<float> output_fused(queue,{M,K});

        auto e1=ops::copy_to_device(queue,Q,h_Q.data());
        auto e2=ops::copy_to_device(queue,K_tensor,h_K.data());
        auto e3=ops::copy_to_device(queue,V,h_V.data());

        e1.wait();
        e2.wait();
        e3.wait();

        // Warm-up unfused
        for(std::size_t i=0;i<warmup;i++){
            auto u1=ops::matmul_tiled_transposed_scaled(queue,Q,K_tensor,scores,tile_size);
            auto u2=ops::softmax_parallel(queue,scores,weights,group_size,{u1});
            auto u3=ops::matmul_tiled(queue,weights,V,output_unfused,tile_size,{u2});
            u3.wait();
        }

        // Warm-up fused
        for(std::size_t i=0;i<warmup;i++){
            auto f=ops::attention_scores_online(queue,Q,K_tensor,V,output_fused,tile_size);
            f.wait();
        }

        double qk_time=0.0;
        double softmax_time=0.0;
        double sv_time=0.0;
        double fused_time=0.0;

        // Benchmark unfused
        for(std::size_t i=0;i<iterations;i++){
            auto u1=ops::matmul_tiled_transposed_scaled(queue,Q,K_tensor,scores,tile_size);
            auto u2=ops::softmax_parallel(queue,scores,weights,group_size,{u1});
            auto u3=ops::matmul_tiled(queue,weights,V,output_unfused,tile_size,{u2});

            u3.wait();

            auto qk_start=u1.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto qk_end=u1.get_profiling_info<sycl::info::event_profiling::command_end>();

            auto sm_start=u2.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto sm_end=u2.get_profiling_info<sycl::info::event_profiling::command_end>();

            auto sv_start=u3.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto sv_end=u3.get_profiling_info<sycl::info::event_profiling::command_end>();

            qk_time+=static_cast<double>(qk_end-qk_start);
            softmax_time+=static_cast<double>(sm_end-sm_start);
            sv_time+=static_cast<double>(sv_end-sv_start);
        }

        // Benchmark fused
        for(std::size_t i=0;i<iterations;i++){
            auto f=ops::attention_scores_online(queue,Q,K_tensor,V,output_fused,tile_size);

            f.wait();

            auto start=f.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto end=f.get_profiling_info<sycl::info::event_profiling::command_end>();

            fused_time+=static_cast<double>(end-start);
        }

        // Profiling timestamps are in nanoseconds
        qk_time/=iterations*1.0e6;
        softmax_time/=iterations*1.0e6;
        sv_time/=iterations*1.0e6;
        fused_time/=iterations*1.0e6;

        double unfused_time=qk_time+softmax_time+sv_time;

        std::cout << "M=" << M << ", N=" << N << ", K=" << K << ", tile=" << tile_size << "\n";
        std::cout << "Iterations: " << iterations << "\n\n";

        std::cout << "Unfused:\n";
        std::cout << "  QK^T + scale: " << qk_time << " ms\n";
        std::cout << "  Softmax:      " << softmax_time << " ms\n";
        std::cout << "  S*V:          " << sv_time << " ms\n";
        std::cout << "  Total:        " << unfused_time << " ms\n\n";

        std::cout << "Fused:\n";
        std::cout << "  Total:        " << fused_time << " ms\n\n";

        std::cout << "Unfused / Fused: " << unfused_time/fused_time << "x\n";
    }

     
    {
        std::cout << "\n=== Attention Scaling Benchmark ===\n";
    
        const std::vector<std::size_t> sizes={128,256,512,1024,2048};
        const std::size_t K=128;
        const std::size_t tile_size=16;
        const std::size_t group_size=16;
        const std::size_t warmup=5;
        const std::size_t iterations=50;
    
        std::cout << "K=" << K << ", tile=" << tile_size << "\n";
        std::cout << "Iterations=" << iterations << "\n\n";
    
        std::cout << "Size\tUnfused(ms)\tFused(ms)\tUnfused/Fused\n";
    
        for(std::size_t size:sizes){
            const std::size_t M=size;
            const std::size_t N=size;
    
            std::vector<float> h_Q(M*K);
            std::vector<float> h_K(N*K);
            std::vector<float> h_V(N*K);
    
            for(std::size_t i=0;i<M*K;i++){
                h_Q[i]=static_cast<float>(static_cast<int>(i%17)-8)/8.0f;
            }
    
            for(std::size_t i=0;i<N*K;i++){
                h_K[i]=static_cast<float>(static_cast<int>(i%13)-6)/6.0f;
                h_V[i]=static_cast<float>(static_cast<int>(i%19)-9)/9.0f;
            }
    
            tinyml::Tensor<float> Q(queue,{M,K});
            tinyml::Tensor<float> K_tensor(queue,{N,K});
            tinyml::Tensor<float> V(queue,{N,K});
    
            tinyml::Tensor<float> scores(queue,{M,N});
            tinyml::Tensor<float> weights(queue,{M,N});
            tinyml::Tensor<float> output_unfused(queue,{M,K});
            tinyml::Tensor<float> output_fused(queue,{M,K});
    
            auto e1=ops::copy_to_device(queue,Q,h_Q.data());
            auto e2=ops::copy_to_device(queue,K_tensor,h_K.data());
            auto e3=ops::copy_to_device(queue,V,h_V.data());
    
            e1.wait();
            e2.wait();
            e3.wait();
    
            // Warm-up unfused
            for(std::size_t i=0;i<warmup;i++){
                auto u1=ops::matmul_tiled_transposed_scaled(queue,Q,K_tensor,scores,tile_size);
                auto u2=ops::softmax_parallel(queue,scores,weights,group_size,{u1});
                auto u3=ops::matmul_tiled(queue,weights,V,output_unfused,tile_size,{u2});
                u3.wait();
            }
    
            // Warm-up fused
            for(std::size_t i=0;i<warmup;i++){
                auto f=ops::attention_scores_online(queue,Q,K_tensor,V,output_fused,tile_size);
                f.wait();
            }
    
            double unfused_time=0.0;
            double fused_time=0.0;
    
            // Unfused benchmark
            for(std::size_t i=0;i<iterations;i++){
                auto u1=ops::matmul_tiled_transposed_scaled(queue,Q,K_tensor,scores,tile_size);
                auto u2=ops::softmax_parallel(queue,scores,weights,group_size,{u1});
                auto u3=ops::matmul_tiled(queue,weights,V,output_unfused,tile_size,{u2});
    
                u3.wait();
    
                auto qk_start=u1.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto qk_end=u1.get_profiling_info<sycl::info::event_profiling::command_end>();
    
                auto sm_start=u2.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto sm_end=u2.get_profiling_info<sycl::info::event_profiling::command_end>();
    
                auto sv_start=u3.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto sv_end=u3.get_profiling_info<sycl::info::event_profiling::command_end>();
    
                unfused_time+=static_cast<double>(qk_end-qk_start);
                unfused_time+=static_cast<double>(sm_end-sm_start);
                unfused_time+=static_cast<double>(sv_end-sv_start);
            }
    
            // Fused benchmark
            for(std::size_t i=0;i<iterations;i++){
                auto f=ops::attention_scores_online(queue,Q,K_tensor,V,output_fused,tile_size);
    
                f.wait();
    
                auto start=f.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto end=f.get_profiling_info<sycl::info::event_profiling::command_end>();
    
                fused_time+=static_cast<double>(end-start);
            }
    
            unfused_time/=static_cast<double>(iterations)*1.0e6;
            fused_time/=static_cast<double>(iterations)*1.0e6;
    
            std::cout << size << "\t"
                      << unfused_time << "\t\t"
                      << fused_time << "\t\t"
                      << unfused_time/fused_time << "\n";
        }
    }
    {
        std::cout << "\n=== Attention Large Scaling Benchmark ===\n";
    
        const std::vector<std::size_t> sizes={1024,2048,3072,4096,6144,8192};
        const std::size_t K=128;
        const std::size_t tile_size=16;
        const std::size_t group_size=16;
    
        std::cout << "K=" << K << ", tile=" << tile_size << "\n\n";
        std::cout << "Size\tIterations\tUnfused(ms)\tFused(ms)\tUnfused/Fused\n";
    
        for(std::size_t size:sizes){
            const std::size_t M=size;
            const std::size_t N=size;
    
            std::size_t iterations;
            if(size<=2048) iterations=30;
            else if(size<=4096) iterations=15;
            else iterations=5;
    
            const std::size_t warmup=3;
    
            std::vector<float> h_Q(M*K);
            std::vector<float> h_K(N*K);
            std::vector<float> h_V(N*K);
    
            for(std::size_t i=0;i<M*K;i++){
                h_Q[i]=static_cast<float>(static_cast<int>(i%17)-8)/8.0f;
            }
    
            for(std::size_t i=0;i<N*K;i++){
                h_K[i]=static_cast<float>(static_cast<int>(i%13)-6)/6.0f;
                h_V[i]=static_cast<float>(static_cast<int>(i%19)-9)/9.0f;
            }
    
            tinyml::Tensor<float> Q(queue,{M,K});
            tinyml::Tensor<float> K_tensor(queue,{N,K});
            tinyml::Tensor<float> V(queue,{N,K});
    
            tinyml::Tensor<float> scores(queue,{M,N});
            tinyml::Tensor<float> weights(queue,{M,N});
            tinyml::Tensor<float> output_unfused(queue,{M,K});
            tinyml::Tensor<float> output_fused(queue,{M,K});
    
            auto e1=ops::copy_to_device(queue,Q,h_Q.data());
            auto e2=ops::copy_to_device(queue,K_tensor,h_K.data());
            auto e3=ops::copy_to_device(queue,V,h_V.data());
    
            e1.wait();
            e2.wait();
            e3.wait();
    
            // Warm-up unfused
            for(std::size_t i=0;i<warmup;i++){
                auto u1=ops::matmul_tiled_transposed_scaled(queue,Q,K_tensor,scores,tile_size);
                auto u2=ops::softmax_parallel(queue,scores,weights,group_size,{u1});
                auto u3=ops::matmul_tiled(queue,weights,V,output_unfused,tile_size,{u2});
                u3.wait();
            }
    
            // Warm-up fused
            for(std::size_t i=0;i<warmup;i++){
                auto f=ops::attention_scores_online(queue,Q,K_tensor,V,output_fused,tile_size);
                f.wait();
            }
    
            double unfused_time=0.0;
            double fused_time=0.0;
    
            for(std::size_t i=0;i<iterations;i++){
                auto u1=ops::matmul_tiled_transposed_scaled(queue,Q,K_tensor,scores,tile_size);
                auto u2=ops::softmax_parallel(queue,scores,weights,group_size,{u1});
                auto u3=ops::matmul_tiled(queue,weights,V,output_unfused,tile_size,{u2});
    
                u3.wait();
    
                auto qk_start=u1.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto qk_end=u1.get_profiling_info<sycl::info::event_profiling::command_end>();
    
                auto sm_start=u2.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto sm_end=u2.get_profiling_info<sycl::info::event_profiling::command_end>();
    
                auto sv_start=u3.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto sv_end=u3.get_profiling_info<sycl::info::event_profiling::command_end>();
    
                unfused_time+=static_cast<double>(qk_end-qk_start);
                unfused_time+=static_cast<double>(sm_end-sm_start);
                unfused_time+=static_cast<double>(sv_end-sv_start);
            }
    
            for(std::size_t i=0;i<iterations;i++){
                auto f=ops::attention_scores_online(queue,Q,K_tensor,V,output_fused,tile_size);
    
                f.wait();
    
                auto start=f.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto end=f.get_profiling_info<sycl::info::event_profiling::command_end>();
    
                fused_time+=static_cast<double>(end-start);
            }
    
            unfused_time/=static_cast<double>(iterations)*1.0e6;
            fused_time/=static_cast<double>(iterations)*1.0e6;
    
            std::cout << size << "\t"
                      << iterations << "\t\t"
                      << unfused_time << "\t\t"
                      << fused_time << "\t\t"
                      << unfused_time/fused_time << "\n";
        }
    }


    return 0;
} 