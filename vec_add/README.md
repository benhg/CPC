# Vector Addition

Also does what it says on the tin.

There's a scalar add and a vector add. Scalar add is only legal to run on 1 thread

## Build and run

Scalar:

`nvcc scalar_add.cu --gpu-code=compute_86 -arch=compute_86 -o scalar_add`
