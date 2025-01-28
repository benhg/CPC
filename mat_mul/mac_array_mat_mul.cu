/**
 * dot_product_mat_mul.cu
 * Implementing a MatMul in CUDA via dot products
 * This is a horribly sub-optimal implementation, not just because we ignore tensor cores, but also for 2 other reasons:
 * 1. The kernel is too small. Launch latency will dominate over it (spend thousands of cycles launching kernels for probably 30 cycles per kernel)
 * 2. We are artifically turning matmul into something memory-bandwidth bound. That's dumb.
 */

#include <stdio.h>

/**
 * Perform a single MAC (which is a single element of a dot product) We will do an accumulate on CPU
 */
__global__ void gpu_mac(int *a, int *b, int *c, int dim) {
    int x = threadIdx.x + blockIdx.x * blockDim.x; // Row index for result matrix
    int y = threadIdx.y + blockIdx.y * blockDim.y; // Column index for result matrix
    int k = threadIdx.z + blockIdx.z * blockDim.z; // Index along the shared dimension

    // Ensure the thread is within bounds
    if (x < dim && y < dim && k < dim) {
        // Compute one term of the dot product
        int partial_mac = a[x * dim + k] * b[k * dim + y];

        // Accumulate the result into the correct output element
        atomicAdd(&c[x * dim + y], partial_mac);
    }
}




/**
 * Do the matmul on CPU as well, to compute correctness
 * a, b inputs
 * c output
 * Caller responsible for allocation
 */
void cpu_mul(int * a, int *b, int *c, int n){
	int i, j, k;
	for(i=0;i<n;i++){
    	for(j=0;j<n;j++){
        	* (c+ ((i*n) +j))=0;
        	for(k=0;k<n;k++){
        		// This is an abomination. I see why God invented array syntax.
            	* (c + ((i*n) +j)) += (*(a+(i*n)+k)) * (*(b+(k*n)+j));
       	 	}
    }
}
	return;
}

/**
 * Helper function to launch a bunch of GPU kernels
 * Each kernel computes one element of the value of one of the entries in the output matrix by taking the dot product of a row from A with a colunm from B
 *  That is to say, each kernel does a single Multiply-Accumulate. For a matrix of size NxN, each element is computed by a dot product of an n-dimensional row vector with an n-dimensional column vector. therefore, there are N multiply-accumulates that need to be performed (for each element in the matrix)
 * 	This means there are N(each dot product)*N*N(each place in the array) MACs that happen per matmul. This kernel needs to be launched N**3 times. 
 * They all run in parallel (no data dependencies)
 * 
 * There are 10 SMs on the A2 I am testing this with. Each of them has 128 threads.
 */
void gpu_mul(int *a, int *b, int *c, int n) {
    int *d_a, *d_b, *d_c;
    int vec_elems = n * n;
    int size = sizeof(int);

    // Allocate device memory
    cudaMalloc(&d_a, size * vec_elems);
    cudaMalloc(&d_b, size * vec_elems);
    cudaMalloc(&d_c, size * vec_elems);

    // Copy data to device
    cudaMemcpy(d_a, a, size * vec_elems, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * vec_elems, cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, size * vec_elems); // Initialize output matrix to 0

    // Configure grid and block size (3D configuration)
    dim3 block_dim(8, 8, 8);  // Each block is 8x8x8 threads
    dim3 grid_dim((n + 7) / 8, (n + 7) / 8, (n + 7) / 8);  // 3D grid to cover all dimensions

    // Launch the kernel
    gpu_mac<<<grid_dim, block_dim>>>(d_a, d_b, d_c, n);

    // Synchronize and copy results back
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, size * vec_elems, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}


int main(){
	int *a, *b, *c;
	int n_blocks = 1;
	int threads_per_block = 256;
	int vec_elems = n_blocks * threads_per_block;
	// Create a square matrix: sqrt(vex_elems) x sqrt(vec_elems)
	int dim = sqrt(vec_elems);

	// This means the matrices are sqrt(vec_elems x vec_elems = vec_elems large
	a = (int *) malloc(sizeof(int) * vec_elems);
	b = (int *) malloc(sizeof(int) * vec_elems);
	// The output matrix must be the same size
	c = (int *) malloc(sizeof(int) * vec_elems);

	// Populate input vectors
	for (int i=0; i<dim; i++){
		for(int j=0;j<dim;j++){
			*(a+ (i*dim) +j) = i;
			*(b+ (i*dim) +j) = j;
		}
		printf("\n");
	}


	// Compute CPU matmul
	int * cpu_arr = (int *) malloc(sizeof(int) * vec_elems);
	cpu_mul(a, b, cpu_arr, dim);


	// Run GPU version
	gpu_mul(a, b, c, dim);
	
	
	for (int i=0; i<dim; i++){
		for(int j=0;j<dim;j++){
			printf("GPU %d CPU %d ", *(c+ (i*dim) +j), *(cpu_arr+ (i*dim) +j) );
		}
		printf("\n");
	}
	
	
	return 0;
}
