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
        std::cout << "\n--- Multi-head attention test ---\n";
    
        std::size_t B=2;
        std::size_t H=3;
        std::size_t L=4;
        std::size_t D=8;
        std::size_t tile_size=2;
    
        tinyml::Tensor<float> Q(queue,{B,H,L,D});
        tinyml::Tensor<float> K(queue,{B,H,L,D});
        tinyml::Tensor<float> V(queue,{B,H,L,D});
        tinyml::Tensor<float> U(queue,{B,H,L,D});
        tinyml::Tensor<float> U_ref(queue,{B,H,L,D});
    
        std::size_t numel=B*H*L*D;
    
        std::vector<float> h_Q(numel);
        std::vector<float> h_K(numel);
        std::vector<float> h_V(numel);
        std::vector<float> h_U(numel);
        std::vector<float> h_U_ref(numel);
    
        for(std::size_t i=0;i<numel;i++){
            h_Q[i]=static_cast<float>((static_cast<int>(i)%17)-8)*0.1f;
            h_K[i]=static_cast<float>((static_cast<int>(i)%13)-6)*0.1f;
            h_V[i]=static_cast<float>((static_cast<int>(i)%19)-9)*0.1f;
        }
    
        auto e1=ops::copy_to_device(queue,Q,h_Q.data());
        auto e2=ops::copy_to_device(queue,K,h_K.data());
        auto e3=ops::copy_to_device(queue,V,h_V.data());
    
        auto e_multi=ops::multi_heads_attention_scores_online(queue,Q,K,V,U,tile_size,{e1,e2,e3});
    
        std::vector<sycl::event> ref_events;
    
        for(std::size_t b=0;b<B;b++){
            for(std::size_t h=0;h<H;h++){
                std::size_t offset=(b*H+h)*L*D;
    
                float* Q_head=Q.data()+offset;
                float* K_head=K.data()+offset;
                float* V_head=V.data()+offset;
                float* U_head=U_ref.data()+offset;
    
                auto e=ops::attention_scores_online(queue,Q_head,K_head,V_head,U_head,L,D,L,tile_size,{e1,e2,e3});
                ref_events.push_back(e);
            }
        }
    
        auto e4=ops::copy_to_host(queue,h_U.data(),U,{e_multi});
        auto e5=ops::copy_to_host(queue,h_U_ref.data(),U_ref,ref_events);
    
        e4.wait();
        e5.wait();
    
        float max_diff=0.0f;
    
        for(std::size_t i=0;i<numel;i++){
            max_diff=std::max(max_diff,std::abs(h_U[i]-h_U_ref[i]));
        }
    
        std::cout << "B=" << B << " H=" << H << " L=" << L << " D=" << D << "\n";
        std::cout << "Max difference: " << max_diff << "\n";
    
        if(max_diff<1e-5f){
            std::cout << "MULTI-HEAD ATTENTION TEST PASSED\n";
        }else{
            std::cout << "MULTI-HEAD ATTENTION TEST FAILED\n";
        }
    }

    
    {
        std::cout << "\n--- Batched multi-head strided attention test ---\n";
    
        using T=float;
    
        std::size_t B=4;
        std::size_t L=4;
        std::size_t d_model=8;
        std::size_t nheads=2;
        std::size_t d_head=d_model/nheads;
        std::size_t tile_size=2;
    
        tinyml::Tensor<T> Q(queue,{B,L,d_model});
        tinyml::Tensor<T> K(queue,{B,L,d_model});
        tinyml::Tensor<T> V(queue,{B,L,d_model});
        tinyml::Tensor<T> U_multi(queue,{B,L,d_model});
        tinyml::Tensor<T> U_ref(queue,{B,L,d_model});
    
        std::size_t total_size=B*L*d_model;
    
        std::vector<T> h_Q(total_size);
        std::vector<T> h_K(total_size);
        std::vector<T> h_V(total_size);
        std::vector<T> h_U_multi(total_size);
        std::vector<T> h_U_ref(total_size);
    
        for(std::size_t i=0;i<total_size;i++){
            h_Q[i]=static_cast<T>(static_cast<int>(i%13)-6)*0.1f;
            h_K[i]=static_cast<T>(static_cast<int>(i%11)-5)*0.07f;
            h_V[i]=static_cast<T>(static_cast<int>(i%17)-8)*0.05f;
        }
    
        auto e1=ops::copy_to_device(queue,Q,h_Q.data());
        auto e2=ops::copy_to_device(queue,K,h_K.data());
        auto e3=ops::copy_to_device(queue,V,h_V.data());
    
        auto e_multi=ops::multi_heads_attention_scores_online_strided(queue,Q,K,V,U_multi,nheads,tile_size,{e1,e2,e3});
    
        std::vector<sycl::event> ref_events;
    
        for(std::size_t b=0;b<B;b++){
            std::size_t offset=b*L*d_model;
    
            for(std::size_t h=0;h<nheads;h++){
                std::size_t j_start=h*d_head;
    
                auto e=ops::attention_scores_online_strided(queue,Q.data()+offset,K.data()+offset,V.data()+offset,U_ref.data()+offset,L,d_head,L,j_start,d_model,tile_size,{e1,e2,e3});
    
                ref_events.push_back(e);
            }
        }
    
        auto e4=ops::copy_to_host(queue,h_U_multi.data(),U_multi,{e_multi});
        auto e5=ops::copy_to_host(queue,h_U_ref.data(),U_ref,ref_events);
    
        e4.wait();
        e5.wait();
    
        T max_diff=static_cast<T>(0);
    
        for(std::size_t i=0;i<total_size;i++){
            T diff=sycl::fabs(h_U_multi[i]-h_U_ref[i]);
    
            if(diff>max_diff){
                max_diff=diff;
            }
        }
    
        std::cout << "B=" << B << " L=" << L << " d_model=" << d_model << " nheads=" << nheads << " d_head=" << d_head << "\n";
        std::cout << "Max difference: " << max_diff << "\n";
    
        if(max_diff<1.0e-5f){
            std::cout << "BATCHED MULTI-HEAD STRIDED ATTENTION TEST PASSED\n";
        }
        else{
            std::cout << "BATCHED MULTI-HEAD STRIDED ATTENTION TEST FAILED\n";
        }
    }

    {
        std::cout << "\n--- Causal strided attention test ---\n";
    
        std::size_t L=4;
        std::size_t d_model=4;
        std::size_t nheads=1;
        std::size_t tile_size=2;
    
        tinyml::Tensor<float> Q(queue,{L,d_model});
        tinyml::Tensor<float> K(queue,{L,d_model});
        tinyml::Tensor<float> V(queue,{L,d_model});
        tinyml::Tensor<float> U(queue,{L,d_model});
    
        std::vector<float> h_Q(L*d_model);
        std::vector<float> h_K(L*d_model);
        std::vector<float> h_V(L*d_model);
        std::vector<float> h_U(L*d_model);
        std::vector<float> h_ref(L*d_model,0.0f);
    
        for(std::size_t i=0;i<L*d_model;i++){
            h_Q[i]=static_cast<float>(static_cast<int>(i%7)-3)*0.1f;
            h_K[i]=static_cast<float>(static_cast<int>(i%5)-2)*0.2f;
            h_V[i]=static_cast<float>(i+1);
        }
    
        auto e1=ops::copy_to_device(queue,Q,h_Q.data());
        auto e2=ops::copy_to_device(queue,K,h_K.data());
        auto e3=ops::copy_to_device(queue,V,h_V.data());
    
        auto e4=ops::attention_scores_online_strided_causal(queue,Q,K,V,U,nheads,0,tile_size,{e1,e2,e3});
        auto e5=ops::copy_to_host(queue,h_U.data(),U,{e4});
        e5.wait();
    
        float scale=std::sqrt(static_cast<float>(d_model));
    
        for(std::size_t i=0;i<L;i++){
            std::vector<float> scores(i+1);
    
            float row_max=std::numeric_limits<float>::lowest();
    
            for(std::size_t j=0;j<=i;j++){
                float score=0.0f;
                for(std::size_t d=0;d<d_model;d++){
                    score+=h_Q[i*d_model+d]*h_K[j*d_model+d];
                }
                score/=scale;
                scores[j]=score;
                row_max=std::max(row_max,score);
            }
    
            float row_sum=0.0f;
            for(std::size_t j=0;j<=i;j++){
                scores[j]=std::exp(scores[j]-row_max);
                row_sum+=scores[j];
            }
    
            for(std::size_t d=0;d<d_model;d++){
                float value=0.0f;
                for(std::size_t j=0;j<=i;j++){
                    value+=(scores[j]/row_sum)*h_V[j*d_model+d];
                }
                h_ref[i*d_model+d]=value;
            }
        }
    
        float max_diff=0.0f;
        for(std::size_t i=0;i<L*d_model;i++){
            max_diff=std::max(max_diff,std::abs(h_U[i]-h_ref[i]));
        }
    
        std::cout << "L=" << L << " d_model=" << d_model << " nheads=" << nheads << "\n";
        std::cout << "Max difference: " << max_diff << "\n";
    
        if(max_diff<1.0e-5f){
            std::cout << "CAUSAL STRIDED ATTENTION TEST PASSED\n";
        }
        else{
            std::cout << "CAUSAL STRIDED ATTENTION TEST FAILED\n";
        }
    }

    {
        std::cout << "\n--- Batched multi-head causal strided attention test ---\n";
    
        using T=float;
    
        std::size_t B=2;
        std::size_t L=4;
        std::size_t d_model=8;
        std::size_t nheads=2;
        std::size_t d_head=d_model/nheads;
        std::size_t tile_size=2;
    
        tinyml::Tensor<T> Q(queue,{B,L,d_model});
        tinyml::Tensor<T> K(queue,{B,L,d_model});
        tinyml::Tensor<T> V(queue,{B,L,d_model});
        tinyml::Tensor<T> U(queue,{B,L,d_model});
    
        std::size_t total_size=B*L*d_model;
    
        std::vector<T> h_Q(total_size);
        std::vector<T> h_K(total_size);
        std::vector<T> h_V(total_size);
        std::vector<T> h_U(total_size);
        std::vector<T> h_ref(total_size,0.0f);
    
        for(std::size_t i=0;i<total_size;i++){
            h_Q[i]=static_cast<T>(static_cast<int>(i%13)-6)*0.1f;
            h_K[i]=static_cast<T>(static_cast<int>(i%11)-5)*0.07f;
            h_V[i]=static_cast<T>(static_cast<int>(i%17)-8)*0.05f;
        }
    
        auto e1=ops::copy_to_device(queue,Q,h_Q.data());
        auto e2=ops::copy_to_device(queue,K,h_K.data());
        auto e3=ops::copy_to_device(queue,V,h_V.data());
    
        auto e4=ops::multi_heads_attention_scores_online_strided_causal(queue,Q,K,V,U,nheads,tile_size,{e1,e2,e3});
        auto e5=ops::copy_to_host(queue,h_U.data(),U,{e4});
        e5.wait_and_throw();
    
        // CPU reference
        T scale=std::sqrt(static_cast<T>(d_head));
    
        for(std::size_t b=0;b<B;b++){
            for(std::size_t h=0;h<nheads;h++){
                std::size_t j_start=h*d_head;
    
                for(std::size_t i=0;i<L;i++){
                    std::vector<T> scores(i+1);
    
                    T row_max=std::numeric_limits<T>::lowest();
    
                    for(std::size_t j=0;j<=i;j++){
                        T score=static_cast<T>(0);
    
                        for(std::size_t d=0;d<d_head;d++){
                            std::size_t q_index=(b*L+i)*d_model+j_start+d;
                            std::size_t k_index=(b*L+j)*d_model+j_start+d;
    
                            score+=h_Q[q_index]*h_K[k_index];
                        }
    
                        score/=scale;
                        scores[j]=score;
                        row_max=std::max(row_max,score);
                    }
    
                    T row_sum=static_cast<T>(0);
    
                    for(std::size_t j=0;j<=i;j++){
                        scores[j]=std::exp(scores[j]-row_max);
                        row_sum+=scores[j];
                    }
    
                    for(std::size_t d=0;d<d_head;d++){
                        T value=static_cast<T>(0);
    
                        for(std::size_t j=0;j<=i;j++){
                            std::size_t v_index=(b*L+j)*d_model+j_start+d;
                            value+=(scores[j]/row_sum)*h_V[v_index];
                        }
    
                        std::size_t out_index=(b*L+i)*d_model+j_start+d;
                        h_ref[out_index]=value;
                    }
                }
            }
        }
    
        T max_diff=static_cast<T>(0);
        bool finite_results=true;
    
        for(std::size_t i=0;i<total_size;i++){
            if(!std::isfinite(h_U[i])){
                finite_results=false;
                continue;
            }
    
            T diff=std::abs(h_U[i]-h_ref[i]);
            max_diff=std::max(max_diff,diff);
        }
    
        std::cout << "B=" << B << " L=" << L << " d_model=" << d_model << " nheads=" << nheads << " d_head=" << d_head << "\n";
        std::cout << "Max difference: " << max_diff << "\n";
    
        if(finite_results && max_diff<1.0e-5f){
            std::cout << "BATCHED MULTI-HEAD CAUSAL STRIDED ATTENTION TEST PASSED\n";
        }
        else{
            std::cout << "BATCHED MULTI-HEAD CAUSAL STRIDED ATTENTION TEST FAILED\n";
            all_tests_passed=false;
        }
    }

    {
        std::cout << "\n--- Pre-LN + QKV + causal attention + output projection + residual test ---\n";
    
        std::size_t B=2;
        std::size_t L=128;
        std::size_t d_model=768;
        std::size_t nheads=12;
        std::size_t tile_size=16;
        std::size_t group_size=256;
    
        tinyml::Tensor<float> X(queue,{B,L,d_model});
        tinyml::Tensor<float> X_norm(queue,{B,L,d_model});
        tinyml::Tensor<float> gamma(queue,{d_model});
        tinyml::Tensor<float> beta(queue,{d_model});
        tinyml::Tensor<float> gamma2(queue,{d_model});
        tinyml::Tensor<float> beta2(queue,{d_model});
        tinyml::Tensor<float> Q(queue,{B,L,d_model});
        tinyml::Tensor<float> K(queue,{B,L,d_model});
        tinyml::Tensor<float> V(queue,{B,L,d_model});
        tinyml::Tensor<float> U(queue,{B,L,d_model});
        tinyml::Tensor<float> Y(queue,{B,L,d_model});
        tinyml::Tensor<float> R(queue,{B,L,d_model});
        tinyml::Tensor<float> R_norm(queue,{B,L,d_model});
    
        tinyml::Linear<float> Q_proj(queue,d_model,d_model);
        tinyml::Linear<float> K_proj(queue,d_model,d_model);
        tinyml::Linear<float> V_proj(queue,d_model,d_model);
        tinyml::Linear<float> O_proj(queue,d_model,d_model);
    
        std::vector<float> h_X(B*L*d_model);
        std::vector<float> h_gamma(d_model);
        std::vector<float> h_beta(d_model);
        std::vector<float> h_gamma2(d_model);
        std::vector<float> h_beta2(d_model);
        std::vector<float> h_WQ(d_model*d_model);
        std::vector<float> h_bQ(d_model);
        std::vector<float> h_WK(d_model*d_model);
        std::vector<float> h_bK(d_model);
        std::vector<float> h_WV(d_model*d_model);
        std::vector<float> h_bV(d_model);
        std::vector<float> h_WO(d_model*d_model);
        std::vector<float> h_bO(d_model);

        for(std::size_t i=0;i<d_model;i++){
            h_gamma[i]=1.0f+static_cast<float>(i%7)*0.01f;
            h_beta[i]=static_cast<float>(static_cast<int>(i%5)-2)*0.01f;
        }
        for(std::size_t i=0;i<d_model;i++){
            h_gamma2[i]=0.9f+static_cast<float>(i%11)*0.01f;
            h_beta2[i]=static_cast<float>(static_cast<int>(i%7)-3)*0.02f;
        }
    
        for(std::size_t i=0;i<h_X.size();i++){
            h_X[i]=static_cast<float>(static_cast<int>(i%11)-5)*0.1f;
        }
    
        for(std::size_t i=0;i<h_WQ.size();i++){
            h_WQ[i]=static_cast<float>(static_cast<int>(i%7)-3)*0.05f;
        }
    
        for(std::size_t i=0;i<h_bQ.size();i++){
            h_bQ[i]=static_cast<float>(i)*0.01f;
        }
    
        for(std::size_t i=0;i<h_WK.size();i++){
            h_WK[i]=static_cast<float>(static_cast<int>(i%9)-4)*0.04f;
        }
    
        for(std::size_t i=0;i<h_bK.size();i++){
            h_bK[i]=static_cast<float>(i)*0.02f;
        }
    
        for(std::size_t i=0;i<h_WV.size();i++){
            h_WV[i]=static_cast<float>(static_cast<int>(i%5)-2)*0.06f;
        }
    
        for(std::size_t i=0;i<h_bV.size();i++){
            h_bV[i]=static_cast<float>(i)*0.03f;
        }
    
        for(std::size_t i=0;i<h_WO.size();i++){
            h_WO[i]=static_cast<float>(static_cast<int>(i%13)-6)*0.02f;
        }
    
        for(std::size_t i=0;i<h_bO.size();i++){
            h_bO[i]=static_cast<float>(i)*0.005f;
        }
    
        auto e_X=ops::copy_to_device(queue,X,h_X.data());
        auto e_gamma=ops::copy_to_device(queue,gamma,h_gamma.data());
        auto e_beta=ops::copy_to_device(queue,beta,h_beta.data());
        auto e_gamma2=ops::copy_to_device(queue,gamma2,h_gamma2.data());
        auto e_beta2=ops::copy_to_device(queue,beta2,h_beta2.data());
        auto e_norm=ops::layer_norm_parallel(queue,X.data(),gamma.data(),beta.data(),X_norm.data(),B*L,d_model,group_size,{e_X,e_gamma,e_beta});
    
        auto e_Q_w=ops::copy_to_device(queue,Q_proj.weight(),h_WQ.data());
        auto e_Q_b=ops::copy_to_device(queue,Q_proj.bias(),h_bQ.data());
        auto e_K_w=ops::copy_to_device(queue,K_proj.weight(),h_WK.data());
        auto e_K_b=ops::copy_to_device(queue,K_proj.bias(),h_bK.data());
        auto e_V_w=ops::copy_to_device(queue,V_proj.weight(),h_WV.data());
        auto e_V_b=ops::copy_to_device(queue,V_proj.bias(),h_bV.data());
        auto e_O_w=ops::copy_to_device(queue,O_proj.weight(),h_WO.data());
        auto e_O_b=ops::copy_to_device(queue,O_proj.bias(),h_bO.data());
    
        auto e_Q=Q_proj.forward_3d(X_norm,Q,group_size,tile_size,{e_norm,e_Q_w,e_Q_b});
        auto e_K=K_proj.forward_3d(X_norm,K,group_size,tile_size,{e_norm,e_K_w,e_K_b});
        auto e_V=V_proj.forward_3d(X_norm,V,group_size,tile_size,{e_norm,e_V_w,e_V_b});
    
        auto e_attn=ops::multi_heads_attention_scores_online_strided_causal(queue,Q,K,V,U,nheads,tile_size,{e_Q,e_K,e_V});
        auto e_O=O_proj.forward_3d(U,Y,group_size,tile_size,{e_attn,e_O_w,e_O_b});
        auto e_R=ops::add(queue,X,Y,R,group_size,{e_X,e_O});
        auto e_norm2=ops::layer_norm_parallel(queue,R.data(),gamma2.data(),beta2.data(),R_norm.data(),B*L,d_model,group_size,{e_R,e_gamma2,e_beta2});
    
        std::vector<float> h_X_norm_gpu(B*L*d_model);
        std::vector<float> h_R_norm_gpu(B*L*d_model);
        std::vector<float> h_U_gpu(B*L*d_model);
        std::vector<float> h_Y_gpu(B*L*d_model);
        std::vector<float> h_R_gpu(B*L*d_model);
    
        auto e_X_norm_host=ops::copy_to_host(queue,h_X_norm_gpu.data(),X_norm,{e_norm});
        auto e_R_norm_host=ops::copy_to_host(queue,h_R_norm_gpu.data(),R_norm,{e_norm2});
        auto e_U=ops::copy_to_host(queue,h_U_gpu.data(),U,{e_attn});
        auto e_Y=ops::copy_to_host(queue,h_Y_gpu.data(),Y,{e_O});
        auto e_R_host=ops::copy_to_host(queue,h_R_gpu.data(),R,{e_R});
    
        e_X_norm_host.wait_and_throw();
        e_R_norm_host.wait_and_throw();
        e_U.wait_and_throw();
        e_Y.wait_and_throw();
        e_R_host.wait_and_throw();
    
        std::vector<float> h_X_norm(B*L*d_model,0.0f);
    
        for(std::size_t row=0;row<B*L;row++){
            float mean=0.0f;
    
            for(std::size_t j=0;j<d_model;j++){
                mean+=h_X[row*d_model+j];
            }
    
            mean/=static_cast<float>(d_model);
    
            float var=0.0f;
    
            for(std::size_t j=0;j<d_model;j++){
                float diff=h_X[row*d_model+j]-mean;
                var+=diff*diff;
            }
    
            var/=static_cast<float>(d_model);
    
            for(std::size_t j=0;j<d_model;j++){
                float normalized=(h_X[row*d_model+j]-mean)/std::sqrt(var+eps);
                h_X_norm[row*d_model+j]=h_gamma[j]*normalized+h_beta[j];
            }
        }
    
        std::vector<float> h_Q(B*L*d_model,0.0f);
        std::vector<float> h_K(B*L*d_model,0.0f);
        std::vector<float> h_V(B*L*d_model,0.0f);
    
        for(std::size_t row=0;row<B*L;row++){
            for(std::size_t j=0;j<d_model;j++){
                float value=h_bQ[j];
    
                for(std::size_t k=0;k<d_model;k++){
                    value+=h_X_norm[row*d_model+k]*h_WQ[k*d_model+j];
                }
    
                h_Q[row*d_model+j]=value;
            }
        }
    
        for(std::size_t row=0;row<B*L;row++){
            for(std::size_t j=0;j<d_model;j++){
                float value=h_bK[j];
    
                for(std::size_t k=0;k<d_model;k++){
                    value+=h_X_norm[row*d_model+k]*h_WK[k*d_model+j];
                }
    
                h_K[row*d_model+j]=value;
            }
        }
    
        for(std::size_t row=0;row<B*L;row++){
            for(std::size_t j=0;j<d_model;j++){
                float value=h_bV[j];
    
                for(std::size_t k=0;k<d_model;k++){
                    value+=h_X_norm[row*d_model+k]*h_WV[k*d_model+j];
                }
    
                h_V[row*d_model+j]=value;
            }
        }
    
        std::size_t d_head=d_model/nheads;
        float scale=std::sqrt(static_cast<float>(d_head));
        std::vector<float> h_U_ref(B*L*d_model,0.0f);
    
        for(std::size_t b=0;b<B;b++){
            for(std::size_t h=0;h<nheads;h++){
                std::size_t j_start=h*d_head;
    
                for(std::size_t i=0;i<L;i++){
                    std::vector<float> scores(i+1);
                    float row_max=std::numeric_limits<float>::lowest();
    
                    for(std::size_t j=0;j<=i;j++){
                        float score=0.0f;
    
                        for(std::size_t d=0;d<d_head;d++){
                            std::size_t q_index=(b*L+i)*d_model+j_start+d;
                            std::size_t k_index=(b*L+j)*d_model+j_start+d;
                            score+=h_Q[q_index]*h_K[k_index];
                        }
    
                        score/=scale;
                        scores[j]=score;
                        row_max=std::max(row_max,score);
                    }
    
                    float row_sum=0.0f;
    
                    for(std::size_t j=0;j<=i;j++){
                        scores[j]=std::exp(scores[j]-row_max);
                        row_sum+=scores[j];
                    }
    
                    for(std::size_t d=0;d<d_head;d++){
                        float value=0.0f;
    
                        for(std::size_t j=0;j<=i;j++){
                            std::size_t v_index=(b*L+j)*d_model+j_start+d;
                            value+=(scores[j]/row_sum)*h_V[v_index];
                        }
    
                        std::size_t out_index=(b*L+i)*d_model+j_start+d;
                        h_U_ref[out_index]=value;
                    }
                }
            }
        }
    
        std::vector<float> h_Y_ref(B*L*d_model,0.0f);
    
        for(std::size_t row=0;row<B*L;row++){
            for(std::size_t j=0;j<d_model;j++){
                float value=h_bO[j];
    
                for(std::size_t k=0;k<d_model;k++){
                    value+=h_U_ref[row*d_model+k]*h_WO[k*d_model+j];
                }
    
                h_Y_ref[row*d_model+j]=value;
            }
        }
    
        std::vector<float> h_R_ref(B*L*d_model);
        std::vector<float> h_R_norm_ref(B*L*d_model,0.0f);
        for(std::size_t row=0;row<B*L;row++){
            float mean=0.0f;
        
            for(std::size_t j=0;j<d_model;j++){
                mean+=h_R_gpu[row*d_model+j]; //mean+=h_R_ref[row*d_model+j];
            }
        
            mean/=static_cast<float>(d_model);
        
            float var=0.0f;
        
            for(std::size_t j=0;j<d_model;j++){
                float diff=h_R_gpu[row*d_model+j]-mean; //h_R_ref[row*d_model+j]-mean;
                var+=diff*diff;
            }
        
            var/=static_cast<float>(d_model);
        
            for(std::size_t j=0;j<d_model;j++){
                float normalized=(h_R_gpu[row*d_model+j]-mean)/std::sqrt(var+eps); // (h_R_ref[row*d_model+j]-mean)/std::sqrt(var+eps);
                h_R_norm_ref[row*d_model+j]=h_gamma2[j]*normalized+h_beta2[j];
            }
        }
    
        for(std::size_t i=0;i<h_R_ref.size();i++){
            h_R_ref[i]=h_X[i]+h_Y_ref[i];
        }
    
        float max_diff_norm=0.0f;
        float mean_diff_norm=0.0f;
    
        for(std::size_t i=0;i<h_X_norm_gpu.size();i++){
            float diff=std::abs(h_X_norm_gpu[i]-h_X_norm[i]);
            max_diff_norm=std::max(max_diff_norm,diff);
            mean_diff_norm+=diff;
        }
    
        mean_diff_norm/=static_cast<float>(h_X_norm_gpu.size());
    
        float max_diff_U=0.0f;
        float mean_diff_U=0.0f;
    
        for(std::size_t i=0;i<h_U_gpu.size();i++){
            float diff=std::abs(h_U_gpu[i]-h_U_ref[i]);
            max_diff_U=std::max(max_diff_U,diff);
            mean_diff_U+=diff;
        }
    
        mean_diff_U/=static_cast<float>(h_U_gpu.size());
    
        float max_diff_Y=0.0f;
        float mean_diff_Y=0.0f;
    
        for(std::size_t i=0;i<h_Y_gpu.size();i++){
            float diff=std::abs(h_Y_gpu[i]-h_Y_ref[i]);
            max_diff_Y=std::max(max_diff_Y,diff);
            mean_diff_Y+=diff;
        }
    
        mean_diff_Y/=static_cast<float>(h_Y_gpu.size());
    
        float max_diff_R=0.0f;
        float mean_diff_R=0.0f;
    
        for(std::size_t i=0;i<h_R_gpu.size();i++){
            float diff=std::abs(h_R_gpu[i]-h_R_ref[i]);
            max_diff_R=std::max(max_diff_R,diff);
            mean_diff_R+=diff;
        }
    
        mean_diff_R/=static_cast<float>(h_R_gpu.size());

        float max_diff_norm2=0.0f;
        float mean_diff_norm2=0.0f;

        for(std::size_t i=0;i<h_R_norm_gpu.size();i++){
            float diff=std::abs(h_R_norm_gpu[i]-h_R_norm_ref[i]);
            max_diff_norm2=std::max(max_diff_norm2,diff);
            mean_diff_norm2+=diff;
        }

        mean_diff_norm2/=static_cast<float>(h_R_norm_gpu.size());
    
        std::cout << "B=" << B << " L=" << L << " d_model=" << d_model << " nheads=" << nheads << "\n";
        std::cout << "LayerNorm mean difference: " << mean_diff_norm << "\n";
        std::cout << "LayerNorm max difference: " << max_diff_norm << "\n";
        std::cout << "Attention U mean difference: " << mean_diff_U << "\n";
        std::cout << "Attention U max difference: " << max_diff_U << "\n";
        std::cout << "Output Y mean difference: " << mean_diff_Y << "\n";
        std::cout << "Output Y max difference: " << max_diff_Y << "\n";
        std::cout << "Residual R mean difference: " << mean_diff_R << "\n";
        std::cout << "Residual R max difference: " << max_diff_R << "\n";
        std::cout << "LayerNorm 2 mean difference: " << mean_diff_norm2 << "\n";
        std::cout << "LayerNorm 2 max difference: " << max_diff_norm2 << "\n";
    
        if(max_diff_R<5.0e-4f && max_diff_norm2<5.0e-4f){
            std::cout << "PRE-LN + QKV + CAUSAL ATTENTION + OUTPUT PROJECTION + RESIDUAL TEST PASSED\n";
        }
        else{
            std::cout << "PRE-LN + QKV + CAUSAL ATTENTION + OUTPUT PROJECTION + RESIDUAL TEST FAILED\n";
            all_tests_passed=false;
        }
    }
    
    if(all_tests_passed){
        std::cout << "\nALL ATTENTION TESTS PASSED\n";
        return 0;
    }

    std::cout << "\nSOME ATTENTION TESTS FAILED\n";
    return 1;
}