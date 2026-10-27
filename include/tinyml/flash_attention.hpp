
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
sycl::event flash_attention_tiled_old(sycl::queue& queue,T *q, T *k, T *b, T *c,std::size_t M,std::size_t N, std::size_t D, std::size_t Dv, std::size_t tile_size, const std::vector<sycl::event>& dependencies={}){
    std::size_t global_rows = ((M + tile_size - 1) / tile_size) * tile_size;
    std::size_t global_cols = ((Dv + tile_size - 1) / tile_size) * tile_size;

    return queue.submit([&](sycl::handler& h){
        h.depends_on(dependencies);
        sycl::local_accessor<T, 2> local_S(sycl::range<2>(tile_size, tile_size), h);
        sycl::local_accessor<T, 2> local_V(sycl::range<2>(tile_size, tile_size), h);
        sycl::local_accessor<T, 2> local_S_row(sycl::range<2>(tile_size, tile_size), h);
        sycl::local_accessor<T, 2> local_A(sycl::range<2>(tile_size, tile_size+1), h);
        sycl::local_accessor<T, 2> local_B(sycl::range<2>(tile_size, tile_size+1), h);
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
                /*if(row<M && tile_start+local_col<N){
                    local_S[local_row][local_col]=s[row*N+tile_start+local_col];
                }
                else{
                    local_S[local_row][local_col]=std::numeric_limits<T>::lowest();
                }*/
                std::size_t b_row = tile_start + local_row;
                T local_score=static_cast<T>(0);    
                for (std::size_t s_tile_start=0; s_tile_start<D; s_tile_start+=tile_size)
                {
                    if(row<M && (s_tile_start+local_col)<D){
                        local_A[local_row][local_col]=q[row*D+(s_tile_start+local_col)];
                    }
                    else{
                        local_A[local_row][local_col]=static_cast<T>(0);
                    }
        
                    if(b_row<N && (s_tile_start+local_col)<D){
                        local_B[local_row][local_col]=k[b_row*D+(s_tile_start+local_col)];
                    }
                    else{
                        local_B[local_row][local_col]=static_cast<T>(0);
                    }
        
                    item.barrier(sycl::access::fence_space::local_space);
        
                    for(std::size_t i_k=0; i_k<tile_size; i_k++){
                        local_score+=local_A[local_row][i_k]*local_B[local_col][i_k];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                local_score/=sycl::sqrt(static_cast<T>(D));
                if(row<M && tile_start+local_col<N){
                    local_S[local_row][local_col]=local_score;
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
sycl::event flash_attention_tiled(sycl::queue& queue, T* q, T* k, T* b, T* c, std::size_t M, std::size_t N, std::size_t D, std::size_t Dv, std::size_t tile_size, const std::vector<sycl::event>& dependencies = {}) {
    if (M == 0 || N == 0 || D == 0 || Dv == 0 || tile_size == 0 || (tile_size & (tile_size - 1)) != 0 || tile_size > queue.get_device().get_info<sycl::info::device::max_work_group_size>()) throw std::invalid_argument("Invalid FlashAttention dimensions or tile_size");
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(dependencies);
        sycl::local_accessor<T, 1> scores(sycl::range<1>(tile_size), h);
        sycl::local_accessor<T, 1> probabilities(sycl::range<1>(tile_size), h);
        sycl::local_accessor<T, 1> reduction(sycl::range<1>(tile_size), h);
        sycl::local_accessor<T, 1> output_acc(sycl::range<1>(Dv), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(M * tile_size), sycl::range<1>(tile_size)), [=](sycl::nd_item<1> item) {
            const std::size_t row = item.get_group(0);
            const std::size_t lane = item.get_local_id(0);
            const T neg_inf = -std::numeric_limits<T>::infinity();
            T m = neg_inf;
            T l = static_cast<T>(0);
            if (lane == 0) for (std::size_t dv = 0; dv < Dv; ++dv) output_acc[dv] = static_cast<T>(0);
            item.barrier(sycl::access::fence_space::local_space);
            for (std::size_t j = 0; j < N; j += tile_size) {
                const std::size_t key = j + lane;
                T score = static_cast<T>(0);
                if (key < N) {
                    for (std::size_t d = 0; d < D; ++d) score += q[row * D + d] * k[key * D + d];
                    score /= sycl::sqrt(static_cast<T>(D));
                } else score = neg_inf;
                scores[lane] = score;
                item.barrier(sycl::access::fence_space::local_space);
                for (std::size_t stride = tile_size / 2; stride > 0; stride /= 2) {
                    if (lane < stride) scores[lane] = sycl::fmax(scores[lane], scores[lane + stride]);
                    item.barrier(sycl::access::fence_space::local_space);
                }
                const T tile_max = scores[0];
                const T m_new = sycl::fmax(m, tile_max);
                const T alpha = sycl::exp(m - m_new);
                const T p = key < N ? sycl::exp(score - m_new) : static_cast<T>(0);
                probabilities[lane] = p;
                reduction[lane] = p;
                item.barrier(sycl::access::fence_space::local_space);
                for (std::size_t stride = tile_size / 2; stride > 0; stride /= 2) {
                    if (lane < stride) reduction[lane] += reduction[lane + stride];
                    item.barrier(sycl::access::fence_space::local_space);
                }
                const T tile_sum = reduction[0];
                const T l_new = alpha * l + tile_sum;
                for (std::size_t dv = 0; dv < Dv; ++dv) {
                    reduction[lane] = key < N ? probabilities[lane] * b[key * Dv + dv] : static_cast<T>(0);
                    item.barrier(sycl::access::fence_space::local_space);
                    for (std::size_t stride = tile_size / 2; stride > 0; stride /= 2) {
                        if (lane < stride) reduction[lane] += reduction[lane + stride];
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    if (lane == 0) output_acc[dv] = alpha * output_acc[dv] + reduction[0];
                    item.barrier(sycl::access::fence_space::local_space);
                }
                m = m_new;
                l = l_new;
                item.barrier(sycl::access::fence_space::local_space);
            }
            if (lane == 0) for (std::size_t dv = 0; dv < Dv; ++dv) c[row * Dv + dv] = output_acc[dv] / l;
        });
    });
}


template <typename T>
sycl::event flash_attention_tiled_old_new(sycl::queue& queue, T* q, T* k, T* b, T* c, std::size_t M, std::size_t N, std::size_t D, std::size_t Dv, std::size_t tile_size, const std::vector<sycl::event>& dependencies = {}) {
    const std::size_t global_rows = ((M + tile_size - 1) / tile_size) * tile_size;
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(dependencies);
        sycl::local_accessor<T, 2> local_S(sycl::range<2>(tile_size, tile_size), h);
        sycl::local_accessor<T, 2> local_P(sycl::range<2>(tile_size, tile_size), h);
        sycl::local_accessor<T, 2> local_R(sycl::range<2>(tile_size, tile_size), h);
        sycl::local_accessor<T, 2> local_A(sycl::range<2>(tile_size, tile_size + 1), h);
        sycl::local_accessor<T, 2> local_B(sycl::range<2>(tile_size, tile_size + 1), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(global_rows, tile_size), sycl::range<2>(tile_size, tile_size)), [=](sycl::nd_item<2> item) {
            const std::size_t row = item.get_global_id(0);
            const std::size_t local_row = item.get_local_id(0);
            const std::size_t local_col = item.get_local_id(1);
            const T neg_inf = -std::numeric_limits<T>::infinity();
            T m = neg_inf;
            T l = static_cast<T>(0);
            for (std::size_t tile_start = 0; tile_start < N; tile_start += tile_size) {
                const std::size_t key = tile_start + local_col;
                T score = static_cast<T>(0);
                for (std::size_t d_start = 0; d_start < D; d_start += tile_size) {
                    const std::size_t feature = d_start + local_col;
                    const std::size_t key_for_load = tile_start + local_row;
                    local_A[local_row][local_col] = row < M && feature < D ? q[row * D + feature] : static_cast<T>(0);
                    local_B[local_row][local_col] = key_for_load < N && feature < D ? k[key_for_load * D + feature] : static_cast<T>(0);
                    item.barrier(sycl::access::fence_space::local_space);
                    for (std::size_t d = 0; d < tile_size; ++d) score += local_A[local_row][d] * local_B[local_col][d];
                    item.barrier(sycl::access::fence_space::local_space);
                }
                score /= sycl::sqrt(static_cast<T>(D));
                local_S[local_row][local_col] = row < M && key < N ? score : (row < M ? neg_inf : static_cast<T>(0));
                local_R[local_row][local_col] = local_S[local_row][local_col];
                item.barrier(sycl::access::fence_space::local_space);
                for (std::size_t stride = tile_size / 2; stride > 0; stride /= 2) {
                    if (local_col < stride) local_R[local_row][local_col] = sycl::fmax(local_R[local_row][local_col], local_R[local_row][local_col + stride]);
                    item.barrier(sycl::access::fence_space::local_space);
                }
                const T tile_max = local_R[local_row][0];
                const T m_new = sycl::fmax(m, tile_max);
                const T alpha = sycl::exp(m - m_new);
                local_P[local_row][local_col] = row < M && key < N ? sycl::exp(local_S[local_row][local_col] - m_new) : static_cast<T>(0);
                local_R[local_row][local_col] = local_P[local_row][local_col];
                item.barrier(sycl::access::fence_space::local_space);
                for (std::size_t stride = tile_size / 2; stride > 0; stride /= 2) {
                    if (local_col < stride) local_R[local_row][local_col] += local_R[local_row][local_col + stride];
                    item.barrier(sycl::access::fence_space::local_space);
                }
                const T tile_sum = local_R[local_row][0];
                const T l_new = alpha * l + tile_sum;
                for (std::size_t dv = 0; dv < Dv; ++dv) {
                    local_R[local_row][local_col] = key < N ? local_P[local_row][local_col] * b[key * Dv + dv] : static_cast<T>(0);
                    item.barrier(sycl::access::fence_space::local_space);
                    for (std::size_t stride = tile_size / 2; stride > 0; stride /= 2) {
                        if (local_col < stride) local_R[local_row][local_col] += local_R[local_row][local_col + stride];
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    if (local_col == 0 && row < M) c[row * Dv + dv] = alpha * c[row * Dv + dv] + local_R[local_row][0];
                    item.barrier(sycl::access::fence_space::local_space);
                }
                m = m_new;
                l = l_new;
                item.barrier(sycl::access::fence_space::local_space);
            }
            if (local_col == 0 && row < M) for (std::size_t dv = 0; dv < Dv; ++dv) c[row * Dv + dv] /= l;
        });
    });
}


template <typename T>
sycl::event flash_attention_tiled_vv(sycl::queue& queue, T* q, T* k, T* b, T* c, std::size_t M, std::size_t N, std::size_t D, std::size_t Dv, std::size_t tile_size, const std::vector<sycl::event>& dependencies = {}) {
    if (tile_size == 0 || D == 0 || Dv == 0) throw std::invalid_argument("Invalid dimensions");
    const T scale = T(1) / sycl::sqrt(static_cast<T>(D));
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(dependencies);
        h.parallel_for(sycl::range<1>(M), [=](sycl::id<1> item) {
            const std::size_t row = item[0];
            T m = -std::numeric_limits<T>::infinity();
            T l = T(0);
            for (std::size_t d = 0; d < Dv; ++d) c[row * Dv + d] = T(0);
            for (std::size_t tile_start = 0; tile_start < N; tile_start += tile_size) {
                const std::size_t count = sycl::min(tile_size, N - tile_start);
                T tile_max = -std::numeric_limits<T>::infinity();
                for (std::size_t r = 0; r < count; ++r) {
                    T score = T(0);
                    for (std::size_t d = 0; d < D; ++d) score += q[row * D + d] * k[(tile_start + r) * D + d];
                    tile_max = sycl::fmax(tile_max, score * scale);
                }
                const T m_new = sycl::fmax(m, tile_max);
                const T alpha = sycl::isfinite(m) ? sycl::exp(m - m_new) : T(0);
                T tile_sum = T(0);
                for (std::size_t r = 0; r < count; ++r) {
                    T score = T(0);
                    for (std::size_t d = 0; d < D; ++d) score += q[row * D + d] * k[(tile_start + r) * D + d];
                    tile_sum += sycl::exp(score * scale - m_new);
                }
                l = alpha * l + tile_sum;
                for (std::size_t d = 0; d < Dv; ++d) {
                    T acc = c[row * Dv + d] * alpha;
                    for (std::size_t r = 0; r < count; ++r) {
                        T score = T(0);
                        for (std::size_t x = 0; x < D; ++x) score += q[row * D + x] * k[(tile_start + r) * D + x];
                        acc += sycl::exp(score * scale - m_new) * b[(tile_start + r) * Dv + d];
                    }
                    c[row * Dv + d] = acc;
                }
                m = m_new;
            }
            for (std::size_t d = 0; d < Dv; ++d) c[row * Dv + d] /= l;
        });
    });
}


template <typename T>
sycl::event flash_attention_tiled_stream(sycl::queue& queue, T* q, T* k, T* b, T* c,
    std::size_t M, std::size_t N, std::size_t D, std::size_t Dv, std::size_t tile_size,
    const std::vector<sycl::event>& dependencies = {})
{
    if (M == 0 || N == 0 || D == 0 || Dv == 0 || tile_size == 0 ||
        (tile_size & (tile_size - 1)) != 0) {
        throw std::invalid_argument("Invalid FlashAttention dimensions or tile_size");
    }

    const std::size_t global_rows = ((M + tile_size - 1) / tile_size) * tile_size;
    const T scale = static_cast<T>(1) / sycl::sqrt(static_cast<T>(D));

    return queue.submit([&](sycl::handler& h) {
        h.depends_on(dependencies);
        sycl::local_accessor<T, 2> local_S(sycl::range<2>(tile_size, tile_size), h);
        sycl::local_accessor<T, 2> local_P(sycl::range<2>(tile_size, tile_size), h);
        sycl::local_accessor<T, 2> local_V(sycl::range<2>(tile_size, tile_size), h);
        sycl::local_accessor<T, 2> local_R(sycl::range<2>(tile_size, tile_size), h);
        sycl::local_accessor<T, 2> local_A(sycl::range<2>(tile_size, tile_size + 1), h);
        sycl::local_accessor<T, 2> local_B(sycl::range<2>(tile_size, tile_size + 1), h);

        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(global_rows, tile_size),
                              sycl::range<2>(tile_size, tile_size)),
            [=](sycl::nd_item<2> item) {
                const std::size_t row = item.get_global_id(0);
                const std::size_t local_row = item.get_local_id(0);
                const std::size_t local_col = item.get_local_id(1);
                const T neg_inf = -std::numeric_limits<T>::infinity();

                T m = neg_inf;
                T l = static_cast<T>(0);
                bool first_tile = true;

                for (std::size_t dv_tile = 0; dv_tile < Dv; dv_tile += tile_size) {
                    const std::size_t dv = dv_tile + local_col;
                    if (row < M && dv < Dv) {
                        c[row * Dv + dv] = static_cast<T>(0);
                    }
                }

                for (std::size_t tile_start = 0; tile_start < N; tile_start += tile_size) {
                    const std::size_t b_row = tile_start + local_row;
                    const std::size_t key = tile_start + local_col;

                    T local_score = static_cast<T>(0);
                    for (std::size_t s_tile_start = 0; s_tile_start < D; s_tile_start += tile_size) {
                        if (row < M && (s_tile_start + local_col) < D) {
                            local_A[local_row][local_col] = q[row * D + (s_tile_start + local_col)];
                        } else {
                            local_A[local_row][local_col] = static_cast<T>(0);
                        }

                        if (b_row < N && (s_tile_start + local_col) < D) {
                            local_B[local_row][local_col] = k[b_row * D + (s_tile_start + local_col)];
                        } else {
                            local_B[local_row][local_col] = static_cast<T>(0);
                        }

                        item.barrier(sycl::access::fence_space::local_space);

                        for (std::size_t i_k = 0; i_k < tile_size; ++i_k) {
                            local_score += local_A[local_row][i_k] * local_B[local_col][i_k];
                        }

                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    local_score *= scale;
                    local_S[local_row][local_col] =
                        (row < M && key < N) ? local_score : neg_inf;
                    local_R[local_row][local_col] = local_S[local_row][local_col];
                    item.barrier(sycl::access::fence_space::local_space);

                    for (std::size_t stride = tile_size / 2; stride > 0; stride /= 2) {
                        if (local_col < stride) {
                            local_R[local_row][local_col] = sycl::fmax(
                                local_R[local_row][local_col],
                                local_R[local_row][local_col + stride]);
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    const T row_max = local_R[local_row][0];
                    const T m_prev = m;
                    const T m_new = first_tile ? row_max : sycl::fmax(m, row_max);
                    const T alpha = first_tile ? static_cast<T>(0) : sycl::exp(m_prev - m_new);
                    first_tile = false;

                    const T p = (row < M && key < N)
                                    ? sycl::exp(local_S[local_row][local_col] - m_new)
                                    : static_cast<T>(0);
                    local_P[local_row][local_col] = p;
                    local_R[local_row][local_col] = p;
                    item.barrier(sycl::access::fence_space::local_space);

                    for (std::size_t stride = tile_size / 2; stride > 0; stride /= 2) {
                        if (local_col < stride) {
                            local_R[local_row][local_col] +=
                                local_R[local_row][local_col + stride];
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    l = alpha * l + local_R[local_row][0];
                    m = m_new;

                    for (std::size_t dv_tile = 0; dv_tile < Dv; dv_tile += tile_size) {
                        const std::size_t dv = dv_tile + local_col;
                        if (b_row < N && dv < Dv) {
                            local_V[local_row][local_col] = b[b_row * Dv + dv];
                        } else {
                            local_V[local_row][local_col] = static_cast<T>(0);
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        T acc = static_cast<T>(0);
                        for (std::size_t kk = 0; kk < tile_size; ++kk) {
                            acc += local_P[local_row][kk] * local_V[kk][local_col];
                        }

                        if (row < M && dv < Dv) {
                            c[row * Dv + dv] = alpha * c[row * Dv + dv] + acc;
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                }

                for (std::size_t dv_tile = 0; dv_tile < Dv; dv_tile += tile_size) {
                    const std::size_t dv = dv_tile + local_col;
                    if (row < M && dv < Dv) {
                        c[row * Dv + dv] /= l;
                    }
                }
            });
    });
}