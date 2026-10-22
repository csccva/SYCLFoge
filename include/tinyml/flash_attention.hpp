
#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <vector>
#include <stdexcept>
#include <limits>
#include <numbers>
#include "tinyml/tensor.hpp"

template <typename T>
sycl::event softmax_matmul_tiled(sycl::queue& queue,T *s, T *b, T *c,std::size_t M,std::size_t N, std::size_t Dv, std::size_t tile_size, const std::vector<sycl::event>& dependencies={}){
    std::size_t global_rows = ((M + tile_size - 1) / tile_size) * tile_size;
    std::size_t global_cols = ((Dv + tile_size - 1) / tile_size) * tile_size;

    return queue.submit([&](sycl::handler& h){
        h.depends_on(dependencies);
        sycl::local_accessor<T, 2> local_S(sycl::range<2>(tile_size, tile_size), h);
        sycl::local_accessor<T, 2> local_V(sycl::range<2>(tile_size, tile_size), h);
        sycl::local_accessor<T, 2> local_S_row(sycl::range<2>(tile_size, tile_size), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(global_rows, global_cols),sycl::range<2>(tile_size, tile_size)), [=](sycl::nd_item<2> item){
            std::size_t row = item.get_global_id(0);                
            std::size_t col = item.get_global_id(1);
            
            std::size_t local_row = item.get_local_id(0);
            std::size_t local_col = item.get_local_id(1);

            T m = std::numeric_limits<T>::lowest();// -std::numeric_limits<T>::infinity();
            T l = static_cast<T>(0);
            T local_cij=static_cast<T>(0);
            bool first_tile = true;
            for(std::size_t tile_start=0;tile_start<N; tile_start+=tile_size){                
                if(row<M && tile_start+local_col<N){
                    local_S[local_row][local_col]=s[row*N+tile_start+local_col];
                }
                else{
                    local_S[local_row][local_col]=std::numeric_limits<T>::lowest();
                }
                if(tile_start+local_row<N && col<Dv){
                    local_V[local_row][local_col]=b[(tile_start+local_row)*Dv+col];
                }
                else{
                    local_V[local_row][local_col]=static_cast<T>(0);
                }
                item.barrier(sycl::access::fence_space::local_space); // barrier  
                T row_max = local_S[local_row][local_col];
                local_S_row[local_row][local_col] = row_max;                
                item.barrier(sycl::access::fence_space::local_space); // barrier 
                for (std::size_t stride = tile_size / 2; stride > 0; stride /= 2) {
                    if(local_col<stride){
                        local_S_row[local_row][local_col]=(local_S_row[local_row][local_col]>local_S_row[local_row][local_col+stride]?local_S_row[local_row][local_col]:local_S_row[local_row][local_col+stride]);
                    }
                    item.barrier(sycl::access::fence_space::local_space); // barrier 
                }
                row_max = local_S_row[local_row][0];
                T m_prev = m;
                m = first_tile ? row_max : (m > row_max ? m : row_max);//m=(m>row_max?m:row_max);
                T alpha = first_tile ? static_cast<T>(0) : sycl::exp(m_prev - m);
                first_tile = false;
                T p=((tile_start + local_col <N && row< M) ?sycl::exp(local_S[local_row][local_col]-m): static_cast<T>(0));
                local_S_row[local_row][local_col]=p;
                item.barrier(sycl::access::fence_space::local_space); // barrier 
                for (std::size_t stride = tile_size / 2; stride > 0; stride /= 2) {
                    if(local_col<stride){
                        local_S_row[local_row][local_col]+=local_S_row[local_row][local_col+stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space); // barrier  
                }
                T tile_Sum=local_S_row[local_row][0];
                l=alpha*l+tile_Sum;
                local_cij*=alpha;
                local_S_row[local_row][local_col]=p;
                item.barrier(sycl::access::fence_space::local_space); // barrier 
                for(std::size_t k=0;k<tile_size; k++){
                    local_cij+=local_S_row[local_row][k]*local_V[k][local_col];
                }
                item.barrier(sycl::access::fence_space::local_space);
            }
            if (row < M && col < Dv){
                c[row * Dv + col] = local_cij / l;
            } 
        });
    });
}



template <typename T>
sycl::event flash_attenton_tiled(sycl::queue& queue,T *s, T *b, T *c,std::size_t M,std::size_t N, std::size_t Dv, std::size_t tile_size, const std::vector<sycl::event>& dependencies={}){
    std::size_t global_rows = ((M + tile_size - 1) / tile_size) * tile_size;
    std::size_t global_cols = ((Dv + tile_size - 1) / tile_size) * tile_size;

    return queue.submit([&](sycl::handler& h){
        h.depends_on(dependencies);
        sycl::local_accessor<T, 2> local_S(sycl::range<2>(tile_size, tile_size), h);
        sycl::local_accessor<T, 2> local_V(sycl::range<2>(tile_size, tile_size), h);
        sycl::local_accessor<T, 2> local_S_row(sycl::range<2>(tile_size, tile_size), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(global_rows, global_cols),sycl::range<2>(tile_size, tile_size)), [=](sycl::nd_item<2> item){
            std::size_t row = item.get_global_id(0);                
            std::size_t col = item.get_global_id(1);
            
            std::size_t local_row = item.get_local_id(0);
            std::size_t local_col = item.get_local_id(1);

            T m = std::numeric_limits<T>::lowest();// -std::numeric_limits<T>::infinity();
            T l = static_cast<T>(0);
            T local_cij=static_cast<T>(0);
            bool first_tile = true;
            for(std::size_t tile_start=0;tile_start<N; tile_start+=tile_size){                
                if(row<M && tile_start+local_col<N){
                    local_S[local_row][local_col]=s[row*N+tile_start+local_col];
                }
                else{
                    local_S[local_row][local_col]=std::numeric_limits<T>::lowest();
                }
                if(tile_start+local_row<N && col<Dv){
                    local_V[local_row][local_col]=b[(tile_start+local_row)*Dv+col];
                }
                else{
                    local_V[local_row][local_col]=static_cast<T>(0);
                }
                item.barrier(sycl::access::fence_space::local_space); // barrier  
                T row_max = local_S[local_row][local_col];
                local_S_row[local_row][local_col] = row_max;                
                item.barrier(sycl::access::fence_space::local_space); // barrier 
                for (std::size_t stride = tile_size / 2; stride > 0; stride /= 2) {
                    if(local_col<stride){
                        local_S_row[local_row][local_col]=(local_S_row[local_row][local_col]>local_S_row[local_row][local_col+stride]?local_S_row[local_row][local_col]:local_S_row[local_row][local_col+stride]);
                    }
                    item.barrier(sycl::access::fence_space::local_space); // barrier 
                }
                row_max = local_S_row[local_row][0];
                T m_prev = m;
                m = first_tile ? row_max : (m > row_max ? m : row_max);//m=(m>row_max?m:row_max);
                T alpha = first_tile ? static_cast<T>(0) : sycl::exp(m_prev - m);
                first_tile = false;
                T p=((tile_start + local_col <N && row< M) ?sycl::exp(local_S[local_row][local_col]-m): static_cast<T>(0));
                local_S_row[local_row][local_col]=p;
                item.barrier(sycl::access::fence_space::local_space); // barrier 
                for (std::size_t stride = tile_size / 2; stride > 0; stride /= 2) {
                    if(local_col<stride){
                        local_S_row[local_row][local_col]+=local_S_row[local_row][local_col+stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space); // barrier  
                }
                T tile_Sum=local_S_row[local_row][0];
                l=alpha*l+tile_Sum;
                local_cij*=alpha;
                local_S_row[local_row][local_col]=p;
                item.barrier(sycl::access::fence_space::local_space); // barrier 
                for(std::size_t k=0;k<tile_size; k++){
                    local_cij+=local_S_row[local_row][k]*local_V[k][local_col];
                }
                item.barrier(sycl::access::fence_space::local_space);
            }
            if (row < M && col < Dv){
                c[row * Dv + col] = local_cij / l;
            } 
        });
    });
}