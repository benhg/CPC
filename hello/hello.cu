/**
 * hello.cu
 * A basic helloworld from CUDA
 */

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * Global means can be called from CPU or GPU
 * Will be run on GPU
 */
__global__ void say_hi(){
	// Seems small, but not. printf "just works" in CUDA. Not so in ROCm or heaven forbid Metal
	printf("Hello from thread (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
} 


int main(){
	say_hi<<<2,2,2>>>();
	cudaDeviceSynchronize();
	return 0;
}
