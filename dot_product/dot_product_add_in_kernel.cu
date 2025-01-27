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
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int c_d = a[idx] * b[idx];
	atomicAdd(c, c_d);
	printf("Hello from thread (%d, %d, %d) idx %d. We MULed %d and %d to get %d\n", threadIdx.x, threadIdx.y, threadIdx.z, idx, a[threadIdx.x], b[threadIdx.x], c_d);
}


int main(){
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	// This should be 4 bytes.
	int size = sizeof(int);
	int vec_elems = 16;

	a = (int *) malloc(sizeof(int) * vec_elems);
	b = (int *) malloc(sizeof(int) * vec_elems);
	// This is just 1 int.
	c = (int *) malloc(sizeof(int));

	// Alloc device ptrs
	cudaMalloc(&d_a, size*vec_elems);
	cudaMalloc(&d_b, size*vec_elems);
	cudaMalloc(&d_c, size);
	
	for (int i =0; i<vec_elems;i++){
		a[i] = 1+i;
		b[i] = 2 + i; 
	}


	// H2D copies
	cudaMemcpy(d_a, a, size*vec_elems, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size*vec_elems, cudaMemcpyHostToDevice);

	vec_mul<<<1,16,1>>>(d_a, d_b, d_c);

	// D2H Copies
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	printf("Dot product: %d\n", *c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	
	return 0;
}
