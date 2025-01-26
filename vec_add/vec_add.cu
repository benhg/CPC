/**
 * vec_add.cu
 * Adding vectors in CUDA
 */

#include <stdio.h>

/**
 * Global means can be called from CPU or GPU
 * Will be run on GPU
 */
__global__ void vec_add(int *a, int *b, int *c){
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
	printf("Hello from thread (%d, %d, %d). We added %d and %d to get %d\n", threadIdx.x, threadIdx.y, threadIdx.z, a[threadIdx.x], b[threadIdx.x], c[threadIdx.x]);
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

	vec_add<<<1,16,1>>>(d_a, d_b, d_c);

	// D2H Copies
	cudaMemcpy(c, d_c, size*vec_elems, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	for (int i=0; i<vec_elems; i++){
		printf("Host side. c[%d]=%d\n", i, c[i]);
	}

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	
	return 0;
}
