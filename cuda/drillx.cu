#include <stdint.h>
#include <stdio.h>
#include "drillx.h"
#include "equix.h"
#include "hashx.h"
#include "equix/src/context.h"
#include "equix/src/solver.h"
#include "equix/src/solver_heap.h"
#include "hashx/src/context.h"

const int BATCH_SIZE = 4096;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(err);                                                         \
        }                                                                      \
    } while (0)

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint64_t *out) {
    // Allocate host memory with cudaHostAlloc for better performance
    hashx_ctx **ctxs;
    uint64_t **hash_space;

    CUDA_CHECK(cudaHostAlloc(&ctxs, BATCH_SIZE * sizeof(hashx_ctx*), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&hash_space, BATCH_SIZE * sizeof(uint64_t*), cudaHostAllocDefault));

    // Allocate device memory in a single allocation to reduce fragmentation
    uint64_t *device_memory;
    CUDA_CHECK(cudaMalloc(&device_memory, BATCH_SIZE * INDEX_SPACE * sizeof(uint64_t)));

    // Initialize contexts and hash space
    for (int i = 0; i < BATCH_SIZE; i++) {
        hash_space[i] = device_memory + i * INDEX_SPACE;

        uint8_t seed[40];
        memcpy(seed, challenge, 32);
        uint64_t nonce_offset = *((uint64_t *)nonce) + i;
        memcpy(seed + 32, &nonce_offset, 8);

        ctxs[i] = hashx_alloc(HASHX_INTERPRETED);
        if (!ctxs[i] || !hashx_make(ctxs[i], seed, 40)) {
            for (int j = 0; j <= i; j++) {
                hashx_free(ctxs[j]);
            }
            CUDA_CHECK(cudaFree(device_memory));
            CUDA_CHECK(cudaFreeHost(hash_space));
            CUDA_CHECK(cudaFreeHost(ctxs));
            return;
        }
    }

    // Optimize kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (BATCH_SIZE * INDEX_SPACE + threadsPerBlock - 1) / threadsPerBlock;

    do_hash_stage0i<<<blocksPerGrid, threadsPerBlock>>>(ctxs, hash_space);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure all kernels are finished

    // Copy the result back to host
    CUDA_CHECK(cudaMemcpy(out, device_memory, BATCH_SIZE * INDEX_SPACE * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Clean up
    for (int i = 0; i < BATCH_SIZE; i++) {
        hashx_free(ctxs[i]);
    }
    CUDA_CHECK(cudaFree(device_memory));
    CUDA_CHECK(cudaFreeHost(hash_space));
    CUDA_CHECK(cudaFreeHost(ctxs));
}

__global__ void do_hash_stage0i(hashx_ctx **ctxs, uint64_t **hash_space) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < BATCH_SIZE * INDEX_SPACE) {
        int ctx_idx = idx / INDEX_SPACE;
        int space_idx = idx % INDEX_SPACE;

        // Kernel logic here (as per your specific algorithm)
    }
}
