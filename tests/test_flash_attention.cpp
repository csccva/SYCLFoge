#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <sycl/sycl.hpp>
#include "tinyml/flash_attention.hpp"
#include "tinyml/ops.hpp"

// rm -rf build/; cmake -S . -B build   -DCMAKE_CXX_COMPILER=/home/cristian/intel/compiler/2025.0/bin/icpx   -DTINYML_BUILD_TESTS=ON; cmake --build build --target test_flash_attention; ./build/test_flash_attention

template <typename T>
void cpu_attention(const std::vector<T>& Q, const std::vector<T>& K, const std::vector<T>& V, std::vector<T>& O, size_t M, size_t N, size_t D, size_t Dv)
{
    std::vector<T> S(N);
    std::vector<T> P(N);

    const T scale = static_cast<T>(1) / std::sqrt(static_cast<T>(D));

    // Compute S = Q * K^T / sqrt(D) and softmax row by row
    for(std::size_t i=0; i<M; i++){
        T local_max = std::numeric_limits<T>::lowest();

        for(std::size_t j=0; j<N; j++){
            T local_S = static_cast<T>(0);
            for(std::size_t k=0; k<D; k++){
                local_S += Q[i*D+k] * K[j*D+k];
            }

            local_S *= scale;
            S[j] = local_S;
            local_max = (local_S > local_max ? local_S : local_max);
        }

        T local_sum = static_cast<T>(0);
        for(size_t j=0; j<N; j++){
            P[j] = std::exp(S[j] - local_max);
            local_sum += P[j];
        }

        for(std::size_t j=0; j<N; j++){
            P[j] /= local_sum;
        }    

        // Compute O = P * V
        for(std::size_t j=0; j<Dv; j++){
            T local_O = static_cast<T>(0);
            for(size_t k=0; k<N; k++){
                local_O += P[k] * V[k*Dv+j];
            }
            O[i*Dv+j] = local_O;
        }
    }
    return;
}

int main()
{
    const std::size_t M = 2377, N = 1531, D = 513, Dv = 213; //const std::size_t M = 3377, N = 2531, D = 1299, Dv = 2179;

    std::vector<float> Q(M * D);
    std::vector<float> K(N * D);
    std::vector<float> V(N * Dv);
    std::vector<float> O(M * Dv);
/*     
    for(size_t i=0; i<Q.size(); i++)
        Q[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0f;
    
    for(size_t i=0; i<K.size(); i++)
        K[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 6.0f;
    
    for(size_t i=0; i<V.size(); i++)
        V[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 9.0f;
 */ 
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (auto& x : Q) x = dist(rng);
    for (auto& x : K) x = dist(rng);
    for (auto& x : V) x = dist(rng);

    cpu_attention(Q, K, V, O, M, N, D, Dv);

    // std::cout << "CPU attention output:\n";
    // for(size_t i=0; i<M; i++){
    //     for(size_t j=0; j<Dv; j++){
    //         // TODO: print O[i*Dv+j]
    //         std::cout<< O[i*Dv+j] << " ";
    //     }
    //     // TODO: print a newline after each row
    //     std::cout << std::endl;
    // }

    
    {
        sycl::queue queue{sycl::gpu_selector_v};

        tinyml::Tensor<float> Q_gpu(queue,{M,D});
        tinyml::Tensor<float> K_gpu(queue,{N,D});
        tinyml::Tensor<float> V_gpu(queue,{N,Dv});
        tinyml::Tensor<float> S_gpu(queue,{M,N});
        tinyml::Tensor<float> P_gpu(queue,{M,N});
        tinyml::Tensor<float> O_gpu(queue,{M,Dv});

        auto e1=ops::copy_to_device(queue,Q_gpu,Q.data());
        auto e2=ops::copy_to_device(queue,K_gpu,K.data());
        auto e3=ops::copy_to_device(queue,V_gpu,V.data());

        auto e4=ops::matmul_tiled_transposed_scaled(queue,Q_gpu,K_gpu,S_gpu,16,{e1,e2});
        auto e5=ops::softmax_parallel(queue,S_gpu,P_gpu,64,{e4});
        auto e6=ops::matmul_tiled(queue,P_gpu,V_gpu,O_gpu,32,{e5,e3});

        std::vector<float> h_O_gpu(M*Dv);
        ops::copy_to_host(queue,h_O_gpu.data(),O_gpu,{e6}).wait_and_throw();

        float max_diff=0.0f;
        for(std::size_t i=0;i<M*Dv;i++){
            max_diff=std::max(max_diff,std::abs(O[i]-h_O_gpu[i]));
        }
        const bool passed = max_diff < 1e-4f;
        std::cout << "Maximum CPU/GPU output difference: " << max_diff << "\n";
        std::cout << (passed ? "Naive GPU attention PASSED\n" : "Naive GPU attention FAILED\n");
        //return passed ? 0 : 1;
    }
    {
        sycl::queue queue{sycl::gpu_selector_v};

        tinyml::Tensor<float> Q_gpu(queue,{M,D});
        tinyml::Tensor<float> K_gpu(queue,{N,D});
        tinyml::Tensor<float> V_gpu(queue,{N,Dv});
        tinyml::Tensor<float> S_gpu(queue,{M,N});
        tinyml::Tensor<float> P_gpu(queue,{M,N});
        tinyml::Tensor<float> O_gpu(queue,{M,Dv});

        auto e1=ops::copy_to_device(queue,Q_gpu,Q.data());
        auto e2=ops::copy_to_device(queue,K_gpu,K.data());
        auto e3=ops::copy_to_device(queue,V_gpu,V.data());

        auto e4=ops::matmul_tiled_transposed_scaled(queue,Q_gpu,K_gpu,S_gpu,16,{e1,e2});
        auto e5=softmax_matmul_tiled(queue,S_gpu.data(),V_gpu.data(),O_gpu.data(),M,N,Dv,16,{e4,e3});

        std::vector<float> h_O_gpu(M*Dv);
        ops::copy_to_host(queue,h_O_gpu.data(),O_gpu,{e5}).wait_and_throw();

        float max_diff=0.0f;
        for(std::size_t i=0;i<M*Dv;i++){
            max_diff=std::max(max_diff,std::abs(O[i]-h_O_gpu[i]));
        }
        const bool passed = max_diff < 1e-4f;
        std::cout << "Maximum CPU/GPU output difference: " << max_diff << "\n";
        std::cout << (passed ? "Softmax merge with PV GPU attention PASSED\n" : "Softmax merge with PV GPU attention FAILED\n");
        //return passed ? 0 : 1;

    }
    
    {
        sycl::queue queue{sycl::gpu_selector_v};

        tinyml::Tensor<float> Q_gpu(queue,{M,D});
        tinyml::Tensor<float> K_gpu(queue,{N,D});
        tinyml::Tensor<float> V_gpu(queue,{N,Dv});
        tinyml::Tensor<float> S_gpu(queue,{M,N});
        tinyml::Tensor<float> P_gpu(queue,{M,N});
        tinyml::Tensor<float> O_gpu(queue,{M,Dv});

        auto e1=ops::copy_to_device(queue,Q_gpu,Q.data());
        auto e2=ops::copy_to_device(queue,K_gpu,K.data());
        auto e3=ops::copy_to_device(queue,V_gpu,V.data());

        auto e5=flash_attention_tiled_stream(queue,Q_gpu.data(),K_gpu.data(),V_gpu.data(),O_gpu.data(),M,N,D,Dv,8,{e1,e2,e3});

        std::vector<float> h_O_gpu(M*Dv);
        ops::copy_to_host(queue,h_O_gpu.data(),O_gpu,{e5}).wait_and_throw();

        float max_diff=0.0f;
        for(std::size_t i=0;i<M*Dv;i++){
            max_diff=std::max(max_diff,std::abs(O[i]-h_O_gpu[i]));
        }
        const bool passed = max_diff < 1e-4f;
        std::cout << "Maximum CPU/GPU output difference: " << max_diff << "\n";
        std::cout << (passed ? "All merged on GPU attention PASSED\n" : "All mergedon GPU attention FAILED\n");
        //return passed ? 0 : 1;
    }
    return 0;
}