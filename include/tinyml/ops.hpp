#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <vector>
#include <stdexcept>
#include <limits>
#include <numbers>
#include "tinyml/tensor.hpp"

constexpr float eps = 1.0e-5f;
namespace ops{


    template <typename T>
    sycl::event attention_scores_online(sycl::queue& queue, T *Q, T *K, T *V, T *U, std::size_t M, std::size_t K_dim, std::size_t N, std::size_t tile_size, const std::vector<sycl::event>& dependencies={}){  
        std::size_t global_rows = ((M + tile_size - 1) / tile_size) * tile_size;
        std::size_t global_cols = tile_size;
    
        return queue.submit([&](sycl::handler& h) {
            sycl::local_accessor<T, 2> local_A(sycl::range<2>(tile_size, tile_size+1), h);
            sycl::local_accessor<T, 2> local_B(sycl::range<2>(tile_size, tile_size+1), h);
            sycl::local_accessor<T, 2> local_S(sycl::range<2>(tile_size, tile_size), h);
            sycl::local_accessor<T, 2> local_max(sycl::range<2>(tile_size,tile_size), h);
            sycl::local_accessor<T, 2> local_sum(sycl::range<2>(tile_size, tile_size), h);
            sycl::local_accessor<T, 1> running_max(sycl::range<1>(tile_size), h);
            sycl::local_accessor<T, 1> running_sum(sycl::range<1>(tile_size), h);
            sycl::local_accessor<T, 2> local_V(sycl::range<2>(tile_size, tile_size), h);
            h.depends_on(dependencies);
    
            h.parallel_for(sycl::nd_range<2>(sycl::range<2>(global_rows, global_cols),sycl::range<2>(tile_size, tile_size)), [=](sycl::nd_item<2> item)
            {
                std::size_t row = item.get_global_id(0);
                // std::size_t col = item.get_global_id(1);
    
                std::size_t local_row = item.get_local_id(0);
                std::size_t local_col = item.get_local_id(1);
    
                
                if(local_col==0){
                    running_max[local_row]=std::numeric_limits<T>::lowest();
                    running_sum[local_row]=static_cast<T>(0);
                }
                for(std::size_t k_tile=0; k_tile<K_dim; k_tile+=tile_size){
                    std::size_t u_col=k_tile+local_col;
                    if(row<M && u_col<K_dim){
                        U[row*K_dim+u_col]=static_cast<T>(0);
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                for(std::size_t j_tile=0; j_tile<N; j_tile+=tile_size){
                    std::size_t col = j_tile+local_col;
                    std::size_t b_row = col - local_col + local_row;
                    
                    T local_cij=static_cast<T>(0);    
                    for (std::size_t tile_start=0; tile_start<K_dim; tile_start+=tile_size)
                    {
                        if(row<M && (tile_start+local_col)<K_dim){
                        local_A[local_row][local_col]=Q[row*K_dim+(tile_start+local_col)];
                        }
                        else{
                            local_A[local_row][local_col]=static_cast<T>(0);
                        }
        
                        if(b_row<N && (tile_start+local_col)<K_dim){
                            local_B[local_row][local_col]=K[b_row*K_dim+(tile_start+local_col)];
                        }
                        else{
                            local_B[local_row][local_col]=static_cast<T>(0);
                        }
        
                        item.barrier(sycl::access::fence_space::local_space);
        
                        for(std::size_t k=0; k<tile_size; k++){
                            local_cij+=local_A[local_row][k]*local_B[local_col][k];
                        }
        
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    T local_score=local_cij/sycl::sqrt(static_cast<T>(K_dim));
                    if(row<M && col<N){
                        local_max[local_row][local_col]=local_score;
                    }
                    else{
                        local_max[local_row][local_col]=std::numeric_limits<T>::lowest();
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    // local_max[local_row][local_col]=local_C[local_row][local_col];
                    // item.barrier(sycl::access::fence_space::local_space);
                    for(std::size_t j=tile_size/2; j>0; j>>=1){
                        if(local_col<j){
                            local_max[local_row][local_col]=(local_max[local_row][local_col]>local_max[local_row][local_col+j]?local_max[local_row][local_col]:local_max[local_row][local_col+j]);
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    //running_max[local_row]=local_max[local_row][0];
                    if(row<M && col<N){
                        local_S[local_row][local_col]=sycl::exp(local_score-local_max[local_row][0]);
                    }
                    else{
                        local_S[local_row][local_col]=static_cast<T>(0);
                    }
                    local_sum[local_row][local_col]=local_S[local_row][local_col];
                    item.barrier(sycl::access::fence_space::local_space);
                    for(std::size_t j=tile_size/2; j>0; j>>=1){
                        if(local_col<j){
                            local_sum[local_row][local_col]+=local_sum[local_row][local_col+j];
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    T m_tile=local_max[local_row][0];
                    T m_old= running_max[local_row];
                    T m_new= (m_tile>m_old?m_tile:m_old);
                    T d_old=running_sum[local_row];
                    T d_tile=local_sum[local_row][0];
                    T scale_old=sycl::exp(m_old-m_new);
                    T scale_tile=sycl::exp(m_tile-m_new);
                    T d_new=d_old*scale_old+d_tile*scale_tile;
                    if(local_col==0){
                        running_max[local_row]=m_new;
                        running_sum[local_row]=d_new;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    for(std::size_t k_tile=0; k_tile<K_dim; k_tile+=tile_size){
                        std::size_t v_col=k_tile+local_col;
                    
                        if(b_row<N && v_col<K_dim){
                            local_V[local_row][local_col]=V[b_row*K_dim+v_col];
                        }
                        else{
                            local_V[local_row][local_col]=static_cast<T>(0);
                        }
                    
                        item.barrier(sycl::access::fence_space::local_space);
                    
                        // next: local_S × local_V for this output tile
                        T local_u=static_cast<T>(0);

                        for(std::size_t j=0; j<tile_size; j++){
                            local_u+=local_S[local_row][j]*local_V[j][local_col];
                        }                      
                        std::size_t u_col=k_tile+local_col;
                        if(row<M && u_col<K_dim){
                            U[row*K_dim+u_col]=U[row*K_dim+u_col]*scale_old+local_u*scale_tile;
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                }
                for(std::size_t k_tile=0; k_tile<K_dim; k_tile+=tile_size){
                    std::size_t u_col=k_tile+local_col;
                    if(row<M && u_col<K_dim){
                        U[row*K_dim+u_col]/=running_sum[local_row];
                    }
                }
            });
        });
        
    }
 
    template <typename T>
    sycl::event attention_scores_online(sycl::queue& queue, tinyml::Tensor<T>& Q, tinyml::Tensor<T>& K, tinyml::Tensor<T>& V, tinyml::Tensor<T>& U, std::size_t tile_size, const std::vector<sycl::event>& dependencies={})
    {
        if(Q.shape().size()!=2 || K.shape().size()!=2 || V.shape().size()!=2 || U.shape().size()!=2 ||
        Q.shape()[0]==0 || Q.shape()[1]==0 || K.shape()[0]==0 ||
        Q.shape()[1]!=K.shape()[1] ||
        K.shape()!=V.shape() ||
        U.shape()!=Q.shape()){
            throw std::invalid_argument("attention_scores_online: tensor shapes must match");
        }

        if(tile_size==0 || (tile_size & (tile_size-1))!=0){
            throw std::invalid_argument("attention_scores_online: tile_size must be a power of 2");
        }

        std::size_t M=Q.shape()[0];
        std::size_t K_dim=Q.shape()[1];
        std::size_t N=K.shape()[0];

        return attention_scores_online(queue,Q.data(),K.data(),V.data(),U.data(),M,K_dim,N,tile_size,dependencies);
    }

    template <typename T>
    sycl::event matmul_tiled_transposed_scaled(sycl::queue& queue,T *a, T *b, T *c,std::size_t M,std::size_t K,std::size_t N, std::size_t tile_size, const std::vector<sycl::event>& dependencies={})
    {
        std::size_t global_rows = ((M + tile_size - 1) / tile_size) * tile_size;
        std::size_t global_cols = ((N + tile_size - 1) / tile_size) * tile_size;
    
        return queue.submit([&](sycl::handler& h) {
            sycl::local_accessor<T, 2> local_A(sycl::range<2>(tile_size, tile_size+1), h);
            sycl::local_accessor<T, 2> local_B(sycl::range<2>(tile_size, tile_size+1), h);
            h.depends_on(dependencies);
    
            h.parallel_for(sycl::nd_range<2>(sycl::range<2>(global_rows, global_cols),sycl::range<2>(tile_size, tile_size)), [=](sycl::nd_item<2> item)
            {
                std::size_t row = item.get_global_id(0);
                std::size_t col = item.get_global_id(1);
    
                std::size_t local_row = item.get_local_id(0);
                std::size_t local_col = item.get_local_id(1);
    
                std::size_t b_row = col - local_col + local_row;
    
                T local_cij=static_cast<T>(0);
    
                for (std::size_t tile_start=0; tile_start<K; tile_start+=tile_size)
                {
                    if(row<M && (tile_start+local_col)<K){
                        local_A[local_row][local_col]=a[row*K+(tile_start+local_col)];
                    }
                    else{
                        local_A[local_row][local_col]=static_cast<T>(0);
                    }
    
                    if(b_row<N && (tile_start+local_col)<K){
                        local_B[local_row][local_col]=b[b_row*K+(tile_start+local_col)];
                    }
                    else{
                        local_B[local_row][local_col]=static_cast<T>(0);
                    }
    
                    item.barrier(sycl::access::fence_space::local_space);
    
                    for(std::size_t k=0; k<tile_size; k++){
                        local_cij+=local_A[local_row][k]*local_B[local_col][k];
                    }
    
                    item.barrier(sycl::access::fence_space::local_space);
                }
    
                if(row<M && col<N){
                    c[row*N+col]=local_cij/sycl::sqrt(static_cast<T>(K));
                }
            });
        });
    }
    
    
    template <typename T>
    sycl::event matmul_tiled_transposed_scaled(sycl::queue& queue,tinyml::Tensor<T>& a, tinyml::Tensor<T>& b, tinyml::Tensor<T>& c, std::size_t tile_size, const std::vector<sycl::event>& dependencies={})
    {
        if(a.shape().size()!=2 || b.shape().size()!=2 || c.shape().size()!=2 ||
           a.shape()[1]!=b.shape()[1] ||
           c.shape()[0]!=a.shape()[0] ||
           c.shape()[1]!=b.shape()[0]){
            throw std::invalid_argument("matmul_tiled_transposed: tensor shapes must match");
        }
    
        std::size_t M=a.shape()[0];
        std::size_t K=a.shape()[1];
        std::size_t N=b.shape()[0];
    
        return matmul_tiled_transposed_scaled(queue,a.data(),b.data(),c.data(),M,K,N,tile_size,dependencies);
    }
  

    template <typename T>
    sycl::event matmul_tiled_transposed(sycl::queue& queue,T *a, T *b, T *c,std::size_t M,std::size_t K,std::size_t N, std::size_t tile_size, const std::vector<sycl::event>& dependencies={})
    {
        std::size_t global_rows = ((M + tile_size - 1) / tile_size) * tile_size;
        std::size_t global_cols = ((N + tile_size - 1) / tile_size) * tile_size;
    
        return queue.submit([&](sycl::handler& h) {
            sycl::local_accessor<T, 2> local_A(sycl::range<2>(tile_size, tile_size+1), h);
            sycl::local_accessor<T, 2> local_B(sycl::range<2>(tile_size, tile_size+1), h);
            h.depends_on(dependencies);
    
            h.parallel_for(sycl::nd_range<2>(sycl::range<2>(global_rows, global_cols),sycl::range<2>(tile_size, tile_size)), [=](sycl::nd_item<2> item)
            {
                std::size_t row = item.get_global_id(0);
                std::size_t col = item.get_global_id(1);
    
                std::size_t local_row = item.get_local_id(0);
                std::size_t local_col = item.get_local_id(1);
    
                std::size_t b_row = col - local_col + local_row;
    
                T local_cij=static_cast<T>(0);
    
                for (std::size_t tile_start=0; tile_start<K; tile_start+=tile_size)
                {
                    if(row<M && (tile_start+local_col)<K){
                        local_A[local_row][local_col]=a[row*K+(tile_start+local_col)];
                    }
                    else{
                        local_A[local_row][local_col]=static_cast<T>(0);
                    }
    
                    if(b_row<N && (tile_start+local_col)<K){
                        local_B[local_row][local_col]=b[b_row*K+(tile_start+local_col)];
                    }
                    else{
                        local_B[local_row][local_col]=static_cast<T>(0);
                    }
    
                    item.barrier(sycl::access::fence_space::local_space);
    
                    for(std::size_t k=0; k<tile_size; k++){
                        local_cij+=local_A[local_row][k]*local_B[local_col][k];
                    }
    
                    item.barrier(sycl::access::fence_space::local_space);
                }
    
                if(row<M && col<N){
                    c[row*N+col]=local_cij;
                }
            });
        });
    }
    
    
    template <typename T>
    sycl::event matmul_tiled_transposed(sycl::queue& queue,tinyml::Tensor<T>& a, tinyml::Tensor<T>& b, tinyml::Tensor<T>& c, std::size_t tile_size, const std::vector<sycl::event>& dependencies={})
    {
        if(a.shape().size()!=2 || b.shape().size()!=2 || c.shape().size()!=2 ||
           a.shape()[1]!=b.shape()[1] ||
           c.shape()[0]!=a.shape()[0] ||
           c.shape()[1]!=b.shape()[0]){
            throw std::invalid_argument("matmul_tiled_transposed: tensor shapes must match");
        }
    
        std::size_t M=a.shape()[0];
        std::size_t K=a.shape()[1];
        std::size_t N=b.shape()[0];
    
        return matmul_tiled_transposed(queue,a.data(),b.data(),c.data(),M,K,N,tile_size,dependencies);
    }

    
    template <typename T>
    sycl::event gelu_approx(sycl::queue& queue, T *data,T *Activation, std::size_t counts, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){

        return queue.parallel_for(sycl::nd_range<1>(((counts + group_size - 1) / group_size) * group_size,group_size),dependencies, [=](sycl::nd_item<1> item)
        {
            std::size_t index=item.get_global_id(0);
            if (index < counts) {
                T x=data[index];
                T local_Activation=static_cast<T>(0.5)*x*(static_cast<T>(1.0)+sycl::tanh(sycl::sqrt(static_cast<T>(2.0)/std::numbers::pi_v<T>)*(x+static_cast<T>(0.044715)*x*x*x)));
                //T local_Activation=sycl::tanh(x);
                Activation[index]=local_Activation;
            }

        });
    }

    template <typename T>
    sycl::event gelu_approx(sycl::queue& queue, tinyml::Tensor<T>& tensor, tinyml::Tensor<T>& Activation, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        if ( tensor.shape() != Activation.shape() ) {
            throw std::invalid_argument("gelu_approx: tensor shapes must match");
        }
        return gelu_approx(queue,tensor.data(),Activation.data(),tensor.size(),group_size,dependencies);

    }

    template <typename T>
    sycl::event gelu_exact(sycl::queue& queue, T *data,T *Activation, std::size_t counts, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){

        return queue.parallel_for(sycl::nd_range<1>(((counts + group_size - 1) / group_size) * group_size,group_size),dependencies, [=](sycl::nd_item<1> item)
        {
            std::size_t index=item.get_global_id(0);
            if (index < counts) {
                T x=data[index];
                T local_Activation=static_cast<T>(0.5)*x*(static_cast<T>(1.0)+sycl::erf(x/sycl::sqrt(static_cast<T>(2.0))));
                //T local_Activation=sycl::erf(x);
                Activation[index]=local_Activation;
            }

        });
    }

    template <typename T>
    sycl::event gelu_exact(sycl::queue& queue, tinyml::Tensor<T>& tensor, tinyml::Tensor<T>& Activation, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        if ( tensor.shape() != Activation.shape() ) {
            throw std::invalid_argument("gelu_exact: tensor shapes must match");
        }
        return gelu_exact(queue,tensor.data(),Activation.data(),tensor.size(),group_size,dependencies);

    }

    template <typename T>
    sycl::event layer_norm_parallel(sycl::queue& queue,T *input, T *output, std::size_t M, std::size_t N, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        return queue.submit([&](sycl::handler& h) {        
                sycl::local_accessor<T, 1> local_A(sycl::range<1>(group_size), h); // to store the intermidiate max value and later the intermidiate sums
                h.depends_on(dependencies);           

                h.parallel_for(sycl::nd_range<1>(M*group_size,group_size), [=](sycl::nd_item<1> item)
                {
                    std::size_t i_row = item.get_group(0);                
                    std::size_t local_j= item.get_local_id(0);
                                                           
                    T row_sum=static_cast<T>(0);
                    if(local_j<N){
                        row_sum=input[i_row*N+local_j]; //work item maximum
                    }
                    for(std::size_t j=local_j+group_size;j<N;j+=group_size){
                        row_sum+=input[i_row*N+j];
                    }
                    local_A[local_j]=row_sum;
                    item.barrier(sycl::access::fence_space::local_space); // barrier within the group
                    for(std::size_t stride = group_size / 2;stride>0;stride>>=1){
                        if(local_j<stride){
                            local_A[local_j]+=local_A[local_j+stride];
                        }
                        item.barrier(sycl::access::fence_space::local_space); // barrier within the group
                    }        
                    row_sum= local_A[0]/static_cast<T>(N); 

                    item.barrier(sycl::access::fence_space::local_space); // barrier within the group //not needed here
                    T row_var=static_cast<T>(0);
                    if(local_j<N){
                        row_var=(input[i_row*N+local_j]-row_sum)*(input[i_row*N+local_j]-row_sum); //work item maximum
                    }
                    for(std::size_t j=local_j+group_size;j<N;j+=group_size){
                        row_var+=(input[i_row*N+j]-row_sum)*(input[i_row*N+j]-row_sum); //input[i_row*N+j];
                    }
                    local_A[local_j]=row_var;
                    item.barrier(sycl::access::fence_space::local_space); // barrier within the group
                    for(std::size_t stride = group_size / 2;stride>0;stride>>=1){
                        if(local_j<stride){
                            local_A[local_j]+=local_A[local_j+stride];
                        }
                        item.barrier(sycl::access::fence_space::local_space); // barrier within the group
                    }        
                    row_var= local_A[0]/static_cast<T>(N);
                    //item.barrier(sycl::access::fence_space::local_space); // barrier within the group //not needed here
                    
                    for(std::size_t j=local_j;j<N;j+=group_size){
                        output[i_row*N+j]=(input[i_row*N+j]-row_sum)/sycl::sqrt(row_var+eps);
                    }   
                }
            );
        });
    }

    
    template <typename T>
    sycl::event layer_norm_parallel(sycl::queue& queue,tinyml::Tensor<T>& input, tinyml::Tensor<T>& output, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        if (input.shape().size() != 2 || output.shape().size() != 2 || input.shape() != output.shape() ||  input.shape()[0] == 0 ||  input.shape()[1] == 0) {
            throw std::invalid_argument("layer_norm_parallel: tensors must be 2D, non-empty, and have matching shapes");
        }
        if (group_size == 0 || (group_size & (group_size - 1)) != 0) {
            throw std::invalid_argument("layer_norm_parallel: group_size must be a power of 2");
        }
        std::size_t M=input.shape()[0];
        std::size_t N=input.shape()[1];
        return layer_norm_parallel(queue,input.data(), output.data(), M, N, group_size, dependencies);
    }

    template <typename T>
    sycl::event layer_norm(sycl::queue& queue,T *input, T *output, std::size_t M, std::size_t N, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        return queue.parallel_for(sycl::nd_range<1>(((M + group_size - 1) / group_size) * group_size,group_size),dependencies, [=](sycl::nd_item<1> item)
        {
            std::size_t index=item.get_global_id(0);
            std::size_t i_start=index*N;
            if (index < M) {
                T row_sum=input[i_start];
                for(std::size_t j=i_start+1;j<i_start+N;j++){
                    row_sum+=input[j];
                }
                row_sum/=static_cast<T>(N);
                T row_var=(input[i_start]-row_sum)*(input[i_start]-row_sum);
                for(std::size_t j=i_start+1;j<i_start+N;j++){
                    row_var+=(input[j]-row_sum)*(input[j]-row_sum);
                }
                row_var/=static_cast<T>(N);
                for(std::size_t j=i_start;j<i_start+N;j++){
                    output[j]=(input[j]-row_sum)/sycl::sqrt(row_var+eps);
                }
            }
        });
    }
 
    
    template <typename T>
    sycl::event layer_norm(sycl::queue& queue, tinyml::Tensor<T>& input, tinyml::Tensor<T>& output, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        
        if (input.shape().size() != 2 || output.shape().size() != 2 || input.shape() != output.shape() ||  input.shape()[0] == 0 || input.shape()[1] == 0) {
            throw std::invalid_argument("layer_norm: tensors must be 2D, non-empty, and have matching shapes");
        }
        if (group_size == 0) {
            throw std::invalid_argument("layer_norm: group_size must be larger than");
        }
        std::size_t M=input.shape()[0];
        std::size_t N=input.shape()[1];
        return layer_norm(queue,input.data(), output.data(), M, N, group_size, dependencies);
    }

    template <typename T>
    sycl::event softmax_parallel(sycl::queue& queue,T *input, T *output, std::size_t M, std::size_t N, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        return queue.submit([&](sycl::handler& h) {        
                sycl::local_accessor<T, 1> local_A(sycl::range<1>(group_size), h); // to store the intermidiate max value and later the intermidiate sums
                h.depends_on(dependencies);           

                h.parallel_for(sycl::nd_range<1>(M*group_size,group_size), [=](sycl::nd_item<1> item)
                {
                    std::size_t i_row = item.get_group(0);                
                    std::size_t local_j= item.get_local_id(0);
                    
                    T row_max=std::numeric_limits<T>::lowest();
                    if(local_j<N){
                        row_max=input[i_row*N+local_j]; //work item maximum
                    }
                    for(std::size_t j=local_j+group_size;j<N;j+=group_size){
                        row_max=(input[i_row*N+j]>row_max?input[i_row*N+j]:row_max);
                    }
                    local_A[local_j]=row_max;
                    item.barrier(sycl::access::fence_space::local_space); // barrier within the group
                    for(std::size_t stride = group_size / 2;stride>0;stride>>=1){
                        if(local_j<stride){
                            local_A[local_j]=(local_A[local_j+stride]>local_A[local_j]?local_A[local_j+stride]:local_A[local_j]);
                        }
                        item.barrier(sycl::access::fence_space::local_space); // barrier within the group
                    }        
                    row_max = local_A[0]; 
                    item.barrier(sycl::access::fence_space::local_space); // barrier within the group  
                    
                    T row_sum=static_cast<T>(0);
                    if(local_j<N){
                        row_sum=sycl::exp(input[i_row*N+local_j]-row_max); //work item maximum
                    }
                    for(std::size_t j=local_j+group_size;j<N;j+=group_size){
                        row_sum+=sycl::exp(input[i_row*N+j]-row_max);
                    }
                    local_A[local_j]=row_sum;
                    item.barrier(sycl::access::fence_space::local_space); // barrier within the group
                    for(std::size_t stride = group_size / 2;stride>0;stride>>=1){
                        if(local_j<stride){
                            local_A[local_j]+=local_A[local_j+stride];
                        }
                        item.barrier(sycl::access::fence_space::local_space); // barrier within the group
                    }        
                    row_sum= local_A[0]; 
                    // item.barrier(sycl::access::fence_space::local_space); // barrier within the group //not needed here
                    
                    for(std::size_t j=local_j;j<N;j+=group_size){
                        output[i_row*N+j]=sycl::exp(input[i_row*N+j]-row_max)/row_sum;
                    }   
                }
            );
        });
    }

    
    template <typename T>
    sycl::event softmax_parallel(sycl::queue& queue,tinyml::Tensor<T>& input, tinyml::Tensor<T>& output, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        if (input.shape().size() != 2 || output.shape().size() != 2 || input.shape() != output.shape() ||  input.shape()[0] == 0 ||  input.shape()[1] == 0) {
            throw std::invalid_argument("softmax_parllel: tensors must be 2D, non-empty, and have matching shapes");
        }
        if (group_size == 0 || (group_size & (group_size - 1)) != 0) {
            throw std::invalid_argument("softmax_parallel: group_size must be a power of 2");
        }
        std::size_t M=input.shape()[0];
        std::size_t N=input.shape()[1];
        return softmax_parallel(queue,input.data(), output.data(), M, N, group_size, dependencies);
    }

    template <typename T>
    sycl::event softmax(sycl::queue& queue,T *input, T *output, std::size_t M, std::size_t N, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        return queue.parallel_for(sycl::nd_range<1>(((M + group_size - 1) / group_size) * group_size,group_size),dependencies, [=](sycl::nd_item<1> item)
        {
            std::size_t index=item.get_global_id(0);
            std::size_t i_start=index*N;
            if (index < M) {
                T row_max=input[i_start];
                for(std::size_t j=i_start+1;j<i_start+N;j++){
                    row_max=(input[j]>row_max?input[j]:row_max);
                }
                T row_sum=sycl::exp(input[i_start]-row_max);
                for(std::size_t j=i_start+1;j<i_start+N;j++){
                    row_sum+=sycl::exp(input[j]-row_max);
                }
                for(std::size_t j=i_start;j<i_start+N;j++){
                    output[j]=sycl::exp(input[j]-row_max)/row_sum;
                }
            }
        });
    }


    template <typename T>
    sycl::event softmax(sycl::queue& queue, tinyml::Tensor<T>& input, tinyml::Tensor<T>& output, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        
        if (input.shape().size() != 2 || output.shape().size() != 2 || input.shape() != output.shape() ||  input.shape()[0] == 0||  input.shape()[1] == 0) {
            throw std::invalid_argument("softmax: tensors must be 2D, non-empty, and have matching shapes");
        }
        std::size_t M=input.shape()[0];
        std::size_t N=input.shape()[1];
        return softmax(queue,input.data(), output.data(), M, N, group_size, dependencies);
    }

    template <typename T>
    sycl::event relu(sycl::queue& queue, T *data,T *Activation, std::size_t counts, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){

        return queue.parallel_for(sycl::nd_range<1>(((counts + group_size - 1) / group_size) * group_size,group_size),dependencies, [=](sycl::nd_item<1> item)
        {
            std::size_t index=item.get_global_id(0);
            if (index < counts) {
                T x=data[index];
                T local_Activation=static_cast<T>(0);
                if(x>local_Activation){
                    local_Activation=x;
                }
                Activation[index]=local_Activation;
            }

        });
    }

    template <typename T>
    sycl::event relu(sycl::queue& queue, tinyml::Tensor<T>& tensor, tinyml::Tensor<T>& Activation, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        if ( tensor.shape() != Activation.shape() ) {
            throw std::invalid_argument("relu: tensor shapes must match");
        }
        return relu(queue,tensor.data(),Activation.data(),tensor.size(),group_size,dependencies);

    }


    template <typename T>
    sycl::event add_bias(sycl::queue& queue,T *a, T *b, T *c,std::size_t M,std::size_t N,  std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        return queue.parallel_for(sycl::nd_range<1>(((M*N + group_size - 1) / group_size) * group_size,group_size), dependencies, [=](sycl::nd_item<1> item)
        {
            std::size_t index=item.get_global_id(0); 
            std::size_t c_i=index/N;
            std::size_t c_j=index%N;
            if (c_i <M && c_j<N) {
                c[index]=a[index]+b[c_j];
                }
        });    
    }


    template <typename T>
    sycl::event add_bias(sycl::queue& queue,tinyml::Tensor<T>& a, tinyml::Tensor<T>& b, tinyml::Tensor<T>& c, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        if (a.shape().size() != 2 || b.shape().size() !=1 || c.shape().size() != 2 || a.shape() != c.shape() || a.shape()[1] != b.shape()[0]) {
            throw std::invalid_argument("add_bias: tensor shapes must match");
        }
        std:: size_t M=a.shape()[0];
        std:: size_t N=a.shape()[1];
        return add_bias(queue,a.data(), b.data(), c.data(), M,N, group_size, dependencies);
    }

    template <typename T>
    sycl::event matmul_tiled(sycl::queue& queue,T *a, T *b, T *c,std::size_t M,std::size_t K,std::size_t N,  std::size_t tile_size, const std::vector<sycl::event>& dependencies={}){
        std::size_t global_rows = ((M + tile_size - 1) / tile_size) * tile_size;
        std::size_t global_cols = ((N + tile_size - 1) / tile_size) * tile_size;
        
        return queue.submit([&](sycl::handler& h) {        
            sycl::local_accessor<T, 2> local_A(sycl::range<2>(tile_size, tile_size+1), h);
        
            sycl::local_accessor<T, 2> local_B(sycl::range<2>(tile_size, tile_size), h);
            h.depends_on(dependencies);
            //sycl::stream out(1024,256,h);
        
            h.parallel_for(sycl::nd_range<2>(sycl::range<2>(global_rows, global_cols),sycl::range<2>(tile_size, tile_size)), [=](sycl::nd_item<2> item)
            {
                
                //out << "subgroup size = " << item.get_sub_group().get_local_range()[0] << sycl::endl;
                std::size_t row = item.get_global_id(0);
                std::size_t col = item.get_global_id(1);
                
                std::size_t local_row = item.get_local_id(0);
                std::size_t local_col = item.get_local_id(1);
                T local_cij=static_cast<T>(0);
                for (std::size_t tile_start = 0;tile_start < K;tile_start += tile_size)
                {
                if(row<M && (tile_start+local_col)<K){
                    local_A[local_row][local_col]=a[row*K+(tile_start+local_col)];
                }
                else{
                    local_A[local_row][local_col]=static_cast<T>(0);
                }
                if(col<N && (local_row+tile_start)<K){
                    local_B[local_row][local_col]=b[(local_row+tile_start)*N+col];
                }
                else{
                    local_B[local_row][local_col]=static_cast<T>(0);
                }
                // Synchronize
                item.barrier(sycl::access::fence_space::local_space); // barrier within the group
                // Compute partial result
                for(std::size_t k=0; k<tile_size; k++){
                    local_cij+=local_A[local_row][k]*local_B[k][local_col];
                }
                // Synchronize
                item.barrier(sycl::access::fence_space::local_space); // barrier within the group
                }
                if (row < M && col < N) {
                    c[row * N + col] = local_cij;
                }
            });
        });
    }


    template <typename T>
    sycl::event matmul_tiled(sycl::queue& queue,tinyml::Tensor<T>& a, tinyml::Tensor<T>& b, tinyml::Tensor<T>& c,  std::size_t tile_size, const std::vector<sycl::event>& dependencies={}){
        
        if (a.shape().size() != 2 || b.shape().size() != 2 || c.shape().size() != 2 || a.shape()[1] != b.shape()[0] || c.shape()[0] != a.shape()[0] || c.shape()[1] != b.shape()[1]) {
            throw std::invalid_argument("matmul: tensor shapes must match");
        }
        std::size_t M = a.shape()[0];
        std::size_t K = a.shape()[1];
        std::size_t N = b.shape()[1];
        return  matmul_tiled(queue,a.data(), b.data(), c.data(),M, K, N,tile_size, dependencies);

    } 


    template <typename T>
    sycl::event matmul(sycl::queue& queue,T *a, T *b, T *c,std::size_t M,std::size_t K,std::size_t N, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        return queue.parallel_for(sycl::nd_range<1>(((M*N + group_size - 1) / group_size) * group_size,group_size), dependencies, [=](sycl::nd_item<1> item)
        {
            std::size_t index=item.get_global_id(0); 
            std::size_t c_i=index/N;
            std::size_t c_j=index%N;
            if (c_i <M && c_j<N) {
                T local_cij=static_cast<T>(0);
                for(std::size_t k=0;k<K; k++){
                    std::size_t index_a=c_i*K+k;
                    std::size_t index_b=k*N+c_j;
                    local_cij+=a[index_a]*b[index_b];
                }
                c[index]=local_cij;
            }
        });
    }


    template <typename T>
    sycl::event matmul(sycl::queue& queue,tinyml::Tensor<T>& a, tinyml::Tensor<T>& b, tinyml::Tensor<T>& c, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        
        if (a.shape().size() != 2 || b.shape().size() != 2 || c.shape().size() != 2 || a.shape()[1] != b.shape()[0] || c.shape()[0] != a.shape()[0] || c.shape()[1] != b.shape()[1]) {
            throw std::invalid_argument("matmul: tensor shapes must match");
        }
        std::size_t M = a.shape()[0];
        std::size_t K = a.shape()[1];
        std::size_t N = b.shape()[1];
        return  matmul(queue,a.data(), b.data(), c.data(),M, K, N,group_size, dependencies);

    }


    template <typename T>
    sycl::event copy_to_device(sycl::queue& queue, tinyml::Tensor<T>& tensor, const T *host_data, const std::vector<sycl::event>& dependencies={}){

        return  queue.memcpy(tensor.data(),host_data,tensor.size()*sizeof(T),dependencies);

    }

    template <typename T>
    sycl::event copy_to_host(sycl::queue& queue, T *host_data, tinyml::Tensor<T>& tensor, const std::vector<sycl::event>& dependencies={}){

        return  queue.memcpy(host_data,tensor.data(),tensor.size()*sizeof(T),dependencies);

    }
        
    template <typename T>
    sycl::event scalar_mul(sycl::queue& queue, T *a, T b, T *c, std::size_t counts, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){

        return queue.parallel_for(sycl::nd_range<1>(((counts + group_size - 1) / group_size) * group_size,group_size), dependencies, [=](sycl::nd_item<1> item)
        {
            std::size_t index=item.get_global_id(0); 
            if (index < counts) {       
                c[index]=a[index]*b; 
            }
        });

    }


    template <typename T>
    sycl::event scalar_mul(sycl::queue& queue, tinyml::Tensor<T>& a, T b, tinyml::Tensor<T>& c, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        if (a.shape() != c.shape()) {
            throw std::invalid_argument("scalar_mul: tensor shapes must match");
        }
        return scalar_mul(queue,a.data(),b,c.data(),c.size(),group_size,dependencies);

    }
        
    template <typename T>
    sycl::event div(sycl::queue& queue, T *a, T *b, T *c, std::size_t counts, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){

        return queue.parallel_for(sycl::nd_range<1>(((counts + group_size - 1) / group_size) * group_size,group_size), dependencies, [=](sycl::nd_item<1> item)
        {
            std::size_t index=item.get_global_id(0); 
            if (index < counts) {       
                c[index]=a[index]/b[index]; 
            }
        });

    }


    template <typename T>
    sycl::event div(sycl::queue& queue, tinyml::Tensor<T>& a, tinyml::Tensor<T>& b, tinyml::Tensor<T>& c, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        if (a.shape() != b.shape() || a.shape() != c.shape()) {
            throw std::invalid_argument("div: tensor shapes must match");
        }
        return div(queue,a.data(),b.data(),c.data(),c.size(),group_size,dependencies);

    }
        
    template <typename T>
    sycl::event sub(sycl::queue& queue, T *a, T *b, T *c, std::size_t counts, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){

        return queue.parallel_for(sycl::nd_range<1>(((counts + group_size - 1) / group_size) * group_size,group_size), dependencies, [=](sycl::nd_item<1> item)
        {
            std::size_t index=item.get_global_id(0); 
            if (index < counts) {       
                c[index]=a[index]-b[index]; 
            }
        });

    }


    template <typename T>
    sycl::event sub(sycl::queue& queue, tinyml::Tensor<T>& a, tinyml::Tensor<T>& b, tinyml::Tensor<T>& c, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        if (a.shape() != b.shape() || a.shape() != c.shape()) {
            throw std::invalid_argument("sub: tensor shapes must match");
        }
        return sub(queue,a.data(),b.data(),c.data(),c.size(),group_size,dependencies);

    }
        
    template <typename T>
    sycl::event mul(sycl::queue& queue, T *a, T *b, T *c, std::size_t counts, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){

        return queue.parallel_for(sycl::nd_range<1>(((counts + group_size - 1) / group_size) * group_size,group_size), dependencies, [=](sycl::nd_item<1> item)
        {
            std::size_t index=item.get_global_id(0); 
            if (index < counts) {       
                c[index]=a[index]*b[index]; 
            }
        });

    }


    template <typename T>
    sycl::event mul(sycl::queue& queue, tinyml::Tensor<T>& a, tinyml::Tensor<T>& b, tinyml::Tensor<T>& c, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        if (a.shape() != b.shape() || a.shape() != c.shape()) {
            throw std::invalid_argument("mul: tensor shapes must match");
        }
        return mul(queue,a.data(),b.data(),c.data(),c.size(),group_size,dependencies);

    }
        
    template <typename T>
    sycl::event add(sycl::queue& queue, T *a, T *b, T *c, std::size_t counts, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){

        return queue.parallel_for(sycl::nd_range<1>(((counts + group_size - 1) / group_size) * group_size,group_size), dependencies, [=](sycl::nd_item<1> item)
        {
            std::size_t index=item.get_global_id(0); 
            if (index < counts) {       
                c[index]=a[index]+b[index]; 
            }
        });

    }


    template <typename T>
    sycl::event add(sycl::queue& queue, tinyml::Tensor<T>& a, tinyml::Tensor<T>& b, tinyml::Tensor<T>& c, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){
        if (a.shape() != b.shape() || a.shape() != c.shape()) {
            throw std::invalid_argument("add: tensor shapes must match");
        }
        return add(queue,a.data(),b.data(),c.data(),c.size(),group_size,dependencies);

    }

    template <typename T>
    sycl::event fill(sycl::queue& queue, T *data, T value, std::size_t counts, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){

        return queue.parallel_for(sycl::nd_range<1>(((counts + group_size - 1) / group_size) * group_size,group_size),dependencies, [=](sycl::nd_item<1> item)
        {
            std::size_t index=item.get_global_id(0);
            if (index < counts) {
                data[index]=value; 
            }

        });

    }

    template <typename T>
    sycl::event fill(sycl::queue& queue, tinyml::Tensor<T>& tensor, T value, std::size_t group_size, const std::vector<sycl::event>& dependencies={}){

        return fill(queue,tensor.data(),value,tensor.size(),group_size,dependencies);

    }

}