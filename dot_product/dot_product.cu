/**
 * vec_add.cu
 * Adding vectors in CUDA
 */

#include <stdio.h>

/**
 * Multiply across the vector. We will do an accumulate on CPU
 * 
 * In another version of this, we will use atomic stores to add from the GPU too.
 */
__global__ void vec_mul(int *a, int *b, int *c){
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	c[idx] = a[idx] * b[idx];
	printf("Hello from thread (%d, %d, %d) idx %d. We MULed %d and %d to get %d\n", threadIdx.x, threadIdx.y, threadIdx.z, idx, a[threadIdx.x], b[threadIdx.x], c[threadIdx.x]);
}


int main(){
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	// This should be 4 bytes.
	int size = sizeof(int);
	int vec_elems = 16;

	a = (int *) malloc(sizeof(int) * vec_elems);
	b = (int *) malloc(sizeof(int) * vec_elems);
	c = (int *) malloc(sizeof(int) * vec_elems);

	// Alloc device ptrs
	cudaMalloc(&d_a, size*vec_elems);
	cudaMalloc(&d_b, size*vec_elems);
	cudaMalloc(&d_c, size*vec_elems);
	
	for (int i =0; i<vec_elems;i++){
		a[i] = 1+i;
		b[i] = 2 + i; 
	}


	// H2D copies
	cudaMemcpy(d_a, a, size*vec_elems, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size*vec_elems, cudaMemcpyHostToDevice);

	vec_mul<<<1,16,1>>>(d_a, d_b, d_c);

	// D2H Copies
	cudaMemcpy(c, d_c, size*vec_elems, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	int dot = 0;
	for (int i=0; i<vec_elems; i++){
		dot += c[i];
	}

	printf("Dot product: %d\n", dot);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	
	return 0;
}
