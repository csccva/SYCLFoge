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

    {
        std::cout << "\n=== Fused Online Attention Test ===\n";
    
        const std::size_t M=3, N=5, K=8;
        const std::size_t tile_size=2;
    
        std::vector<float> h_Q={
            1,2,3,4,5,6,7,8,
            2,1,2,1,2,1,2,1,
            1,1,1,1,1,1,1,1
        };
    
        std::vector<float> h_K={
            1,0,1,0,1,0,1,0,
            0,1,0,1,0,1,0,1,
            1,1,1,1,1,1,1,1,
            2,2,2,2,2,2,2,2,
            1,2,1,2,1,2,1,2
        };
    
        std::vector<float> h_V={
            1,2,3,4,5,6,7,8,
            2,3,4,5,6,7,8,9,
            3,4,5,6,7,8,9,10,
            4,5,6,7,8,9,10,11,
            5,6,7,8,9,10,11,12
        };
    
        tinyml::Tensor<float> Q(queue,{M,K});
        tinyml::Tensor<float> K_tensor(queue,{N,K});
        tinyml::Tensor<float> V(queue,{N,K});
        tinyml::Tensor<float> U(queue,{M,K});
    
        auto e1=ops::copy_to_device(queue,Q,h_Q.data());
        auto e2=ops::copy_to_device(queue,K_tensor,h_K.data());
        auto e3=ops::copy_to_device(queue,V,h_V.data());
    
        auto e4=ops::attention_scores_online(queue,Q.data(),K_tensor.data(),V.data(),U.data(),M,K,N,tile_size,{e1,e2,e3});
    
        std::vector<float> h_U(M*K);
        ops::copy_to_host(queue,h_U.data(),U,{e4}).wait_and_throw();
    
        // CPU reference
    
        std::vector<double> scores(M*N,0.0);
        std::vector<double> weights(M*N,0.0);
        std::vector<double> reference(M*K,0.0);
    
        const double scale=1.0/std::sqrt(static_cast<double>(K));
    
        for(std::size_t i=0;i<M;i++){
            for(std::size_t j=0;j<N;j++){
                for(std::size_t k=0;k<K;k++){
                    scores[i*N+j]+=static_cast<double>(h_Q[i*K+k])*static_cast<double>(h_K[j*K+k]);
                }
                scores[i*N+j]*=scale;
            }
        }
    
        for(std::size_t i=0;i<M;i++){
            double row_max=scores[i*N];
    
            for(std::size_t j=1;j<N;j++){
                row_max=std::max(row_max,scores[i*N+j]);
            }
    
            double row_sum=0.0;
    
            for(std::size_t j=0;j<N;j++){
                weights[i*N+j]=std::exp(scores[i*N+j]-row_max);
                row_sum+=weights[i*N+j];
            }
    
            for(std::size_t j=0;j<N;j++){
                weights[i*N+j]/=row_sum;
            }
        }
    
        for(std::size_t i=0;i<M;i++){
            for(std::size_t k=0;k<K;k++){
                for(std::size_t j=0;j<N;j++){
                    reference[i*K+k]+=weights[i*N+j]*static_cast<double>(h_V[j*K+k]);
                }
            }
        }
    
        // Compare
    
        double max_diff=0.0;
        bool finite_results=true;
    
        std::cout << "Fused GPU output:\n";
    
        for(std::size_t i=0;i<M;i++){
            for(std::size_t k=0;k<K;k++){
                std::cout << h_U[i*K+k] << " ";
    
                if(!std::isfinite(h_U[i*K+k])){
                    finite_results=false;
                }
                else{
                    max_diff=std::max(max_diff,std::abs(static_cast<double>(h_U[i*K+k])-reference[i*K+k]));
                }
            }
            std::cout << "\n";
        }
    
        std::cout << "CPU reference:\n";
    
        for(std::size_t i=0;i<M;i++){
            for(std::size_t k=0;k<K;k++){
                std::cout << reference[i*K+k] << " ";
            }
            std::cout << "\n";
        }
    
        std::cout << "Maximum difference: " << max_diff << "\n";
    
        if(finite_results && max_diff<1e-5){
            std::cout << "Fused Online Attention test PASSED\n";
        }
        else{
            std::cout << "Fused Online Attention test FAILED\n";
            all_tests_passed=false;
        }
    }

    {
        std::cout << "\n=== Fused Online Attention Large Test ===\n";
    
        const std::size_t M=127, N=259, K=128;
        const std::size_t tile_size=16;
    
        std::vector<float> h_Q(M*K);
        std::vector<float> h_K(N*K);
        std::vector<float> h_V(N*K);
    
        // Deterministic bounded input data
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
        tinyml::Tensor<float> U(queue,{M,K});
    
        auto e1=ops::copy_to_device(queue,Q,h_Q.data());
        auto e2=ops::copy_to_device(queue,K_tensor,h_K.data());
        auto e3=ops::copy_to_device(queue,V,h_V.data());
    
        auto e4=ops::attention_scores_online(queue,Q.data(),K_tensor.data(),V.data(),U.data(),M,K,N,tile_size,{e1,e2,e3});
    
        std::vector<float> h_U(M*K);
        ops::copy_to_host(queue,h_U.data(),U,{e4}).wait_and_throw();
    
        // CPU reference
        std::vector<double> scores(M*N,0.0);
        std::vector<double> weights(M*N,0.0);
        std::vector<double> reference(M*K,0.0);
    
        const double scale=1.0/std::sqrt(static_cast<double>(K));
    
        for(std::size_t i=0;i<M;i++){
            for(std::size_t j=0;j<N;j++){
                for(std::size_t k=0;k<K;k++){
                    scores[i*N+j]+=static_cast<double>(h_Q[i*K+k])*static_cast<double>(h_K[j*K+k]);
                }
                scores[i*N+j]*=scale;
            }
        }
    
        for(std::size_t i=0;i<M;i++){
            double row_max=scores[i*N];
    
            for(std::size_t j=1;j<N;j++){
                row_max=std::max(row_max,scores[i*N+j]);
            }
    
            double row_sum=0.0;
    
            for(std::size_t j=0;j<N;j++){
                weights[i*N+j]=std::exp(scores[i*N+j]-row_max);
                row_sum+=weights[i*N+j];
            }
    
            for(std::size_t j=0;j<N;j++){
                weights[i*N+j]/=row_sum;
            }
        }
    
        for(std::size_t i=0;i<M;i++){
            for(std::size_t k=0;k<K;k++){
                for(std::size_t j=0;j<N;j++){
                    reference[i*K+k]+=weights[i*N+j]*static_cast<double>(h_V[j*K+k]);
                }
            }
        }
    
        // Compare
        double max_diff=0.0;
        double mean_diff=0.0;
        std::size_t finite_count=0;
        std::size_t non_finite=0;
    
        for(std::size_t i=0;i<M*K;i++){
            if(!std::isfinite(h_U[i])){
                if(non_finite<10){
                    std::cout << "Non-finite U[" << i << "] = " << h_U[i] << "\n";
                }
                non_finite++;
            }
            else{
                double diff=std::abs(static_cast<double>(h_U[i])-reference[i]);
                max_diff=std::max(max_diff,diff);
                mean_diff+=diff;
                finite_count++;
            }
        }
    
        if(finite_count>0){
            mean_diff/=static_cast<double>(finite_count);
        }
    
        std::cout << "M=" << M << ", N=" << N << ", K=" << K << ", tile=" << tile_size << "\n";
        std::cout << "Maximum difference: " << max_diff << "\n";
        std::cout << "Mean difference:    " << mean_diff << "\n";
        std::cout << "Non-finite results: " << non_finite << "\n";
    
        if(non_finite==0 && max_diff<1e-4){
            std::cout << "Fused Online Attention Large test PASSED\n";
        }
        else{
            std::cout << "Fused Online Attention Large test FAILED\n";
            all_tests_passed=false;
        }
    }

    if(all_tests_passed){
        std::cout << "\nALL SDPA TESTS PASSED\n";
        return 0;
    }

    std::cout << "\nSOME SDPA TESTS FAILED\n";
    return 1;
}