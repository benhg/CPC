/**
 * dot_product_add_in_kernel.cu
 * dot product in CUDA using atomic store in the kernel to do the add.
 */

#include <stdio.h>

/**
 * Multiply across the vector. We will do an accumulate on CPU
 * 
 * In another version of this, we will use atomic stores to add from the GPU too.
 */
__global__ void vec_mul(int *a, int *b, int *c){
	int idx = threadIdx.x + (blockDim.x * blockIdx.x);
	int c_d = a[idx] * b[idx];
	atomicAdd(c, c_d);
	//printf("Hello from thread (%d, %d, %d) block %d idx %d. We MULed %d and %d to get %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, idx, a[threadIdx.x], b[threadIdx.x], c_d);
}


int main(){
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	// This should be 4 bytes.
	int size = sizeof(int);
	int n_blocks = 100;
	int threads_per_block = 256;
	int vec_elems = n_blocks * threads_per_block;

	a = (int *) malloc(sizeof(int) * vec_elems);
	b = (int *) malloc(sizeof(int) * vec_elems);
	// This is just 1 int.
	c = (int *) malloc(sizeof(int));

	// Alloc device ptrs
	cudaMalloc(&d_a, size*vec_elems);
	cudaMalloc(&d_b, size*vec_elems);
	cudaMalloc(&d_c, size);
	
	int cpu_dot = 0;
	for (int i =0; i<vec_elems;i++){
		a[i] = 1+i;
		b[i] = 2 + i;
		cpu_dot += a[i] * b[i];
	}


	// H2D copies
	cudaMemcpy(d_a, a, size*vec_elems, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size*vec_elems, cudaMemcpyHostToDevice);
	// N Blocks, N threads per block, bytes of shared mem
	vec_mul<<<n_blocks,threads_per_block,0>>>(d_a, d_b, d_c);

	// D2H Copies
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	printf("Dot product: %d, CPU Dot: %d\n", *c, cpu_dot);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	
	return 0;
}
