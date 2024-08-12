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

__global__ void do_hash_stage0i(hashx_ctx **ctxs, uint64_t *hash_space) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < BATCH_SIZE * INDEX_SPACE) {
        uint64_t value = hash_space[idx];

        // Example operation: XOR with a constant
        value ^= 0xA5A5A5A5A5A5A5A5;

        // Write back to the hash space
        hash_space[idx] = value;
    }
}

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint64_t *out) {
    // Use pinned memory for better performance
    hashx_ctx **ctxs;
    uint64_t *hash_space;

    CUDA_CHECK(cudaHostAlloc(&ctxs, BATCH_SIZE * sizeof(hashx_ctx*), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&hash_space, BATCH_SIZE * INDEX_SPACE * sizeof(uint64_t), cudaHostAllocMapped));

    uint64_t *device_memory;
    CUDA_CHECK(cudaMalloc(&device_memory, BATCH_SIZE * INDEX_SPACE * sizeof(uint64_t)));

    // Initialize contexts and hash space
    for (int i = 0; i < BATCH_SIZE; i++) {
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

        // Precompute values in the CPU and transfer to GPU
        for (int j = 0; j < INDEX_SPACE; j++) {
            hash_space[i * INDEX_SPACE + j] = precompute_value(ctxs[i], j);  // Assuming a precompute function
        }
    }

    // Copy precomputed data to the device
    CUDA_CHECK(cudaMemcpy(device_memory, hash_space, BATCH_SIZE * INDEX_SPACE * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Optimize kernel launch for RTX 4090
    int threadsPerBlock = 1024;  // Max threads per block
    int blocksPerGrid = (BATCH_SIZE * INDEX_SPACE + threadsPerBlock - 1) / threadsPerBlock;

    do_hash_stage0i<<<blocksPerGrid, threadsPerBlock>>>(ctxs, device_memory);
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
