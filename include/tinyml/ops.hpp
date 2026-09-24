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
    sycl::event matmul_tiled_transposed(sycl::queue& queue,T *a, T *b, T *c,std::size_t M,std::size_t K,std::size_t N, std::size_t tile_size, const std::vector<sycl::event>& dependencies={})
    {
        std::size_t global_rows = ((M + tile_size - 1) / tile_size) * tile_size;
        std::size_t global_cols = ((N + tile_size - 1) / tile_size) * tile_size;
    
        return queue.submit([&](sycl::handler& h) {
            sycl::local_accessor<T, 2> local_A(sycl::range<2>(tile_size, tile_size+1), h);
            sycl::local_accessor<T, 2> local_B(sycl::range<2>(tile_size, tile_size), h);
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
            throw std::invalid_argument("softmax: tensors must be 2D, non-empty, and have matching shapes");
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