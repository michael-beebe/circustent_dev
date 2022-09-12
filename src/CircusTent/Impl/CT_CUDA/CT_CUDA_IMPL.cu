/*
 * _CT_OMP_IMPL_C_
 *
 * Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
 * All Rights Reserved
 * contact@tactcomplabs.com
 *
 * See LICENSE in the top level directory for licensing details
 */

#include <stdlib.h>
#include <cuda.h>

__global__ void RAND_ADD( u_int64_t *restrict ARRAY, u_int64_t *restrict IDX, u_int64_t iters, u_int64_t pes ) {

    u_int64_t start = threadIdx.x + blockIdx.x * blockDim.x;
    if( start < iters)
        atomicAdd( &ARRAY[IDX[start]], (u_int64_t) 0x1 );
}

__global__ void RAND_CAS( u_int64_t *restrict ARRAY, u_int64_t *restrict IDX, u_int64_t iters, u_int64_t pes ) {

    u_int64_t start = threadIdx.x + blockIdx.x * blockDim.x;
    if( start < iters)
        atomicCAS( &ARRAY[IDX[start]], ARRAY[IDX[start]], ARRAY[IDX[start]]);
}

__global__ void STRIDE1_ADD( u_int64_t *restrict ARRAY, u_int64_t *restrict IDX, u_int64_t iters, u_int64_t pes ) {
    
    u_int64_t start = threadIdx.x + blockIdx.x * blockDim.x;
    if( start < iters)
        atomicAdd( &ARRAY[start], (u_int64_t) 0x1 );
}

__global__ void STRIDE1_CAS( u_int64_t *restrict ARRAY, u_int64_t *restrict IDX, u_int64_t iters, u_int64_t pes ) {
    
    u_int64_t start = threadIdx.x + blockIdx.x * blockDim.x;
    if( start < iters)
        atomicCAS( &ARRAY[start], ARRAY[start], ARRAY[start]);
}

__global__ void STRIDEN_ADD( u_int64_t *restrict ARRAY, u_int64_t *restrict IDX, u_int64_t iters, u_int64_t pes, uint64_t stride ) {
    
    u_int64_t start = threadIdx.x + blockIdx.x * blockDim.x;
    if( start < iters && start % stride == 0 )
        atomicAdd( &ARRAY[start], (u_int64_t) 0x1 );
}

__global__ void STRIDEN_CAS( u_int64_t *restrict ARRAY, u_int64_t *restrict IDX, u_int64_t iters, u_int64_t pes, uint64_t stride ) {
    
    u_int64_t start = threadIdx.x + blockIdx.x * blockDim.x;
    if( start < iters && start % stride == 0 )
        atomicCAS( &ARRAY[start], ARRAY[start], ARRAY[start]);
}

__global__ void CENTRAL_ADD( u_int64_t *restrict ARRAY, u_int64_t *restrict IDX, u_int64_t iters, u_int64_t pes ) {
    
    u_int64_t start = threadIdx.x + blockIdx.x * blockDim.x;
    if( start < iters)
        atomicAdd( &ARRAY[0], (u_int64_t) 0x1 );
}

__global__ void CENTRAL_CAS( u_int64_t *restrict ARRAY, u_int64_t *restrict IDX, u_int64_t iters, u_int64_t pes ) {
    
    u_int64_t start = threadIdx.x + blockIdx.x * blockDim.x;
    if( start < iters)
        atomicCAS( &ARRAY[0], ARRAY[start], ARRAY[start]);
}