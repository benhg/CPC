from numba import cuda
import cupy

@cuda.jit
def add_scalar(array_a, array_b, array_c):
	tx = cuda.threadIdx.x
	ty = cuda.blockIdx.x
	bw = cuda.blockDim.x
	pos = tx + ty * bw
	array_c[pos] = array_a[pos] + array_b[pos]

threadsperblock = 256
array_a = cupy.ones(threadsperblock, dtype="float32")
array_b = cupy.ones(threadsperblock, dtype="float32")
array_c = cupy.zeros(threadsperblock,dtype="float32")
blockspergrid = (array_a.size +(threadsperblock - 1 )) // threadsperblock
add_scalar[threadsperblock, blockspergrid](array_a, array_b, array_c)
print(array_c)
