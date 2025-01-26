/**
 * hello.cu
 * A basic helloworld from CUDA
 */

#include <stdio.h>

/**
 * Global means can be called from CPU or GPU
 * Will be run on GPU
 */
__global__ void vec_add(int *a, int *b, int *c){
	*c = *a + *b;
	printf("Hello from thread (%d, %d, %d). We added %d and %d to get %d\n", threadIdx.x, threadIdx.y, threadIdx.z, *a, *b, *c);
} 


int main(){
	int a, b, c;
	int *d_a, *d_b, *d_c;
	// This should be 4 bytes.
	int size = sizeof(int);

	// Alloc device ptrs
	cudaMalloc((int**) &d_a, size);
	cudaMalloc((void**) &d_b, size);
	cudaMalloc((void**) &d_c, size);
	a = 1;
	b = 2;
	c = 0;
	// H2D copies
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	vec_add<<<16,1,1>>>(d_a, d_b, d_c);

	// D2H Copies
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	printf("We added %d and %d to get %d\n", a, b, c);

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	
	return 0;
}
