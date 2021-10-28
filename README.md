## Note:

This tutorial was prepared on an NVIDIA Jetson, so no CUDA setup was required. If you are on another platform with an NVIDIA GPU, additional setup may be required to install the CUDA runtime and toolkit. Please consult [NVIDIA developer guides](https://developer.nvidia.com/cuda-toolkit) on instructions.

# Introduction

Early computers could only perform one computation at a time. To multiply 100 numbers, the computer would have to multiply the first 2, then the result of that with the third number, then the result of that with the fourth, yadayadayada ...

Computers can multiply numbers pretty fast, but consider that a single 4k video frame contains 24,883,200 bytes of color information. Multiply that by 30 Hz, 60 Hz, or even 120 Hz, and even the fastest single processor can struggle to keep up.

Modern electronic engineering has enabled us to embed thousands of processing units on to a single thumbnail-sized wafer of silicon. The RTX3070 has 5,888 processing cores. 24,883,200 pixels distributed across those cores means each core only has to process about 4,226 bytes. And when you consider that each core is capable of vector operations (it can do operations on multiple bytes simultaneously) that potentially breaks down the number of operations to only 1408.

From 24,883,200 to 1408. That's a pretty significant reduction ... 4 orders of magnitude better!

Of course, that's a best-case scenario where your problem is very parallelizable, or what CS professors like to call "embarrassingly parallelizable." One such problem is multiplying an array of numbers by a single value, like fading an image to black or white. It's embarrassingly parallelizable because the problem can be readily divided into sub-processes which are completely independent, i.e. processing one element in the array does not rely on the result of processing any other element.

# CUDA

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel processing ecosystem based on their proprietary GPU architecture. It is a bit of a walled garden, but in return you get very tight integration of the API with the hardware. And NVIDIA has done quite an amazing job of making the CUDA API accessible. It's very well documented, and there are loads of examples.

If you know C/C++, you already know the syntax of CUDA, as it is a subset of C/C++ with a few extensions to allow the pre-processor to distinguish it from regular C/C++. If you like, you can write CUDA code in separate files to your code, but you can also embed CUDA code in your programs and NVIDIA's API tools will automagically parse them out for separate compilation.

# Basic Steps

1. Allocate memory buffers on the GPU for the source and result data.
2. Copy the data from host memory to GPU memory.
3. Decide whether your problem is best tackled in 1, 2 or 3 dimensions. For example, if you are processing a simple array of numbers, choose 1 dimension. If you are working on matrices, you may want to opt for 2 dimensions. And if you are working on volumetric gas simulation, you may want 3.
4. Write a CUDA "kernel", which is basically a fancy way of saying a function. When you launch your code, the CUDA run-time will start as many threads as you specify, one thread per processing core, and run an instance of this kernel in each of those threads, in parallel. This is called SIMT (Single Instruction Multiple Thread). The only difference between instances is what memory locations they each reference.
5. Call your kernel.
6. Retrieve the result data from GPU memory.

# Vector Scaling

As mentioned, scaling a vector is an embarrassingly parallelizable problem. Let's say you have 8,096 numbers and want to multiply them all by one constant. You could hire one mathematician to do them all in serial, or hire 8,096 mathematicians and assign each just one of the numbers. No mathematician would have to communicate with, or wait for, any other mathematician, and assuming that the mathematician is competent at multiplication, you would have your result  instantly.

So let's write a simple program.

Open your favorite code editor and enter a main function.

```cpp
int main(int argc, char **argv)
{
	return 0;
}
```

Let's create an array of random numbers to work with.

```cpp
//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#include <stdlib.h>
#define ARRAYSIZE 8096
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

int main(int argc, char **argv)
{
	//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	// Allocate a new buffer in host memory for the source data
	float *hostSrcBuf = new float[ARRAYSIZE];
	// Populate the buffer with random floats between 0.0 and 1.0
	size_t n;
	n = ARRAYSIZE;
	while(n--) hostSrcBuf = ((float) rand()) / (float) RAND_MAX;
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	return 0;
}
```

To make it more interesting, let's seed the random number generator so you get a different sequence every time.

```cpp
#include <stdlib.h>
//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#include <chrono>

using namespace std;
using namespace std::chrono;
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#define ARRAYSIZE 8096

int main(int argc, char **argv)
{
	//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	// Seed the random number generator
	srand(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	// Allocate a new buffer in host memory for the source data
	float *hostSrcBuf = new float[ARRAYSIZE];
	// Populate the buffer with random floats between 0.0 and 1.0
	size_t n;
	n = ARRAYSIZE;
	while(n--) hostSrcBuf = ((float) rand()) / (float) RAND_MAX;
	return 0;
}
```

Next, allocate a buffer of equal size to contain the results.

```cpp
#include <stdlib.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define ARRAYSIZE 8096

int main(int argc, char **argv)
{
	// Seed the random number generator
	srand(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());

	// Allocate a new buffer in host memory for the source data
	float *hostSrcBuf = new float[ARRAYSIZE];
	// Populate the buffer with random floats between 0.0 and 1.0
	size_t n;
	n = ARRAYSIZE;
	while(n--) hostSrcBuf = ((float) rand()) / (float) RAND_MAX;

	//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	// Allocate a new buffer in host memory for the result data
	float *hostResBuf = new float[ARRAYSIZE];
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	return 0;
}
```

Because CUDA executes on the discrete GPU, it cannot access host memory. So the data must be transferred from host memory to device memory. So we need to allocate two buffers on the GPU. For clarity, let's ignore error handling for now.

```cpp
#include <stdlib.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define ARRAYSIZE 8096

int main(int argc, char **argv)
{
	// Seed the random number generator
	srand(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());

	// Allocate a new buffer in host memory for the source data
	float *hostSrcBuf = new float[ARRAYSIZE];
	// Populate the buffer with random floats between 0.0 and 1.0
	size_t n;
	n = ARRAYSIZE;
	while(n--) hostSrcBuf = ((float) rand()) / (float) RAND_MAX;

	// Allocate a new buffer in host memory for the result data
	float *hostResBuf = new float[ARRAYSIZE];

	//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	// Allocate equivalent buffers on the GPU
	size_t byteCount = ARRAYSIZE * sizeof(float);
	float *devSrcBuf = nullptr;
	cudaMalloc((void **) &devSrcBuf, byteCount);
	float *devResBuf = nullptr;
	cudaMalloc((void **) &devResBuf, byteCount);
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	return 0;
}
```

Then copy the array from host to device memory. Note that the `cudaMemcpy()` function takes the size of the data in bytes, not in array elements.

```cpp
#include <stdlib.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define ARRAYSIZE 8096

int main(int argc, char **argv)
{
	// Seed the random number generator
	srand(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());

	// Allocate a new buffer in host memory for the source data
	float *hostSrcBuf = new float[ARRAYSIZE];
	// Populate the buffer with random floats between 0.0 and 1.0
	size_t n;
	n = ARRAYSIZE;
	while(n--) hostSrcBuf = ((float) rand()) / (float) RAND_MAX;

	// Allocate a new buffer in host memory for the result data
	float *hostResBuf = new float[ARRAYSIZE];

	// Allocate equivalent buffers on the GPU
	size_t byteCount = ARRAYSIZE * sizeof(float);
	float *devSrcBuf = nullptr;
	cudaMalloc((void **) &devSrcBuf, byteCount);
	float *devResBuf = nullptr;
	cudaMalloc((void **) &devResBuf, byteCount);

	//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	// Transfer the data
	cudaMemcpy(devSrcBuf, hostSrcBuf, byteCount, cudaMemcpyHostToDevice);
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	return 0;
}
```

Now we decide how to parallelize the problem. The CUDA run-time divides the data into blocks, and each block is capable of running a maximum of 1,024 threads. But you don't HAVE to assign the maximum number of threads to a block. So let's settle on 256 threads per block. The number is arbitrary, but I like powers of 2.

```cpp
#include <stdlib.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define ARRAYSIZE 8096

int main(int argc, char **argv)
{
	// Seed the random number generator
	srand(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());

	// Allocate a new buffer in host memory for the source data
	float *hostSrcBuf = new float[ARRAYSIZE];
	// Populate the buffer with random floats between 0.0 and 1.0
	size_t n;
	n = ARRAYSIZE;
	while(n--) hostSrcBuf = ((float) rand()) / (float) RAND_MAX;

	// Allocate a new buffer in host memory for the result data
	float *hostResBuf = new float[ARRAYSIZE];

	// Allocate equivalent buffers on the GPU
	size_t byteCount = ARRAYSIZE * sizeof(float);
	float *devSrcBuf = nullptr;
	cudaMalloc((void **) &devSrcBuf, byteCount);
	float *devResBuf = nullptr;
	cudaMalloc((void **) &devResBuf, byteCount);

	// Transfer the data
	cudaMemcpy(devSrcBuf, hostSrcBuf, byteCount, cudaMemcpyHostToDevice);

	//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	// Set the threads per bloc
	int threadsPerBlock = 256;
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	return 0;
}
```

Then, theoretically, we would require a block size = ARRAYSIZE / threadsPerBlock. However, because integer math would truncate any fractional part, we need to compensate for that.

```cpp
#include <stdlib.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define ARRAYSIZE 8096

int main(int argc, char **argv)
{
	// Seed the random number generator
	srand(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());

	// Allocate a new buffer in host memory for the source data
	float *hostSrcBuf = new float[ARRAYSIZE];
	// Populate the buffer with random floats between 0.0 and 1.0
	size_t n;
	n = ARRAYSIZE;
	while(n--) hostSrcBuf = ((float) rand()) / (float) RAND_MAX;

	// Allocate a new buffer in host memory for the result data
	float *hostResBuf = new float[ARRAYSIZE];

	// Allocate equivalent buffers on the GPU
	size_t byteCount = ARRAYSIZE * sizeof(float);
	float *devSrcBuf = nullptr;
	cudaMalloc((void **) &devSrcBuf, byteCount);
	float *devResBuf = nullptr;
	cudaMalloc((void **) &devResBuf, byteCount);

	// Transfer the data
	cudaMemcpy(devSrcBuf, hostSrcBuf, byteCount, cudaMemcpyHostToDevice);

	//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	// Set the threads per block, and calculate how many blocks we need
	int threadsPerBlock = 256;
	int numberOfBlocks = (ARRAYSIZE + threadsPerBlock - 1) / threadsPerBlock;
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	return 0;
}
```

Now, we're ready to write the kernel.

```cpp
#include <stdlib.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define ARRAYSIZE 8096

//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
// CUDA Kernel
__global__
void cuda_vec_scale(const float* a, float s, float* c, int count)
{
	// Calculate a linear index and store it in variable i
	// This is also known as the "thread id"
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// If i is beyond the end of the array, do nothing
	if(i >= count) return;

	// Otherwise, multiply the ith element in a by s and 
	// store the result in the ith element of c
	c[i] = a[i] * s;
}
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

int main(int argc, char **argv)
{
	// Seed the random number generator
	srand(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());

	// Allocate a new buffer in host memory for the source data
	float *hostSrcBuf = new float[ARRAYSIZE];
	// Populate the buffer with random floats between 0.0 and 1.0
	size_t n;
	n = ARRAYSIZE;
	while(n--) hostSrcBuf = ((float) rand()) / (float) RAND_MAX;

	// Allocate a new buffer in host memory for the result data
	float *hostResBuf = new float[ARRAYSIZE];

	// Allocate equivalent buffers on the GPU
	size_t byteCount = ARRAYSIZE * sizeof(float);
	float *devSrcBuf = nullptr;
	cudaMalloc((void **) &devSrcBuf, byteCount);
	float *devResBuf = nullptr;
	cudaMalloc((void **) &devResBuf, byteCount);

	// Transfer the data
	cudaMemcpy(devSrcBuf, hostSrcBuf, byteCount, cudaMemcpyHostToDevice);

	// Set the threads per block, and calculate how many blocks we need
	int threadsPerBlock = 256;
	int numberOfBlocks = (ARRAYSIZE + threadsPerBlock - 1) / threadsPerBlock;

	return 0;
}
```

Yes, I did a double take as well when I saw this. The only thing distinguishing this code is the `__global__` keyword, and the `blockIdx`, `blockDim` and `threadIdx` identifiers. At the risk of being obvious, `blockIdx` holds the index of the block containing the thread running the kernel, `blockDim` the size of each block, and `threadIdx` the index of the thread within the block. These variables are assigned by the CUDA runtime before calling the kernel code and can be used to derive a unique index per thread. Because I'm only dealing with a 1 dimensional array, I'm only using the x-components of these variables. They each also have y-, and z-components for use in 2- and 3-dimensional algorithms.

Now we can call the kernel from our code.

```cpp
#include <stdlib.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define ARRAYSIZE 8096

// CUDA Kernel
__global__
void cuda_vec_scale(const float* a, float s, float* c, int count)
{
	// Calculate a linear index and store it in variable i
	// This is also known as the "thread id"
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// If i is beyond the end of the array, do nothing
	if(i >= count) return;

	// Otherwise, multiply the ith element in a by s and 
	// store the result in the ith element of c
	c[i] = a[i] * s;
}

int main(int argc, char **argv)
{
	// Seed the random number generator
	srand(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());

	// Allocate a new buffer in host memory for the source data
	float *hostSrcBuf = new float[ARRAYSIZE];
	// Populate the buffer with random floats between 0.0 and 1.0
	size_t n;
	n = ARRAYSIZE;
	while(n--) hostSrcBuf = ((float) rand()) / (float) RAND_MAX;

	// Allocate a new buffer in host memory for the result data
	float *hostResBuf = new float[ARRAYSIZE];

	// Allocate equivalent buffers on the GPU
	size_t byteCount = ARRAYSIZE * sizeof(float);
	float *devSrcBuf = nullptr;
	cudaMalloc((void **) &devSrcBuf, byteCount);
	float *devResBuf = nullptr;
	cudaMalloc((void **) &devResBuf, byteCount);

	// Transfer the data
	size_t byteCount = ARRAYSIZE * sizeof(float);
	float *devSrcBuf = nullptr;
	cudaMalloc((void **) &devSrcBuf, byteCount);
	float *devResBuf = nullptr;
	cudaMalloc((void **) &devResBuf, byteCount);

	// Transfer the data
	cudaMemcpy(devSrcBuf, hostSrcBuf, byteCount, cudaMemcpyHostToDevice);

	// Set the threads per block, and calculate how many blocks we need
	int threadsPerBlock = 256;
	int numberOfBlocks = (ARRAYSIZE + threadsPerBlock - 1) / threadsPerBlock;

	//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	// Execute the kernel
	cuda_vec_scale<<<numberOfBlocks, threadsPerBlock>>>(devSrcBuf, 1324.0f, devResBuf, ARRAYSIZE);
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	return 0;
}
```

Again, note how the only things distinguishing this line of code are the parameters passed via the triple angle-brackets, and the fact that we are passing in the buffers that have been specially allocated. Regular built in types like ints, floats and doubles can be passed as if passing them to a regular C++ function destined for host execution. I've just plucked a random floating point value out of my @$$ to use as the scalar.

The result now sits in `devResBuf`, so we need to transfer it out to host memory.

```cpp
#include <stdlib.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define ARRAYSIZE 8096

// CUDA Kernel
__global__
void cuda_vec_scale(const float* a, float s, float* c, int count)
{
	// Calculate a linear index and store it in variable i
	// This is also known as the "thread id"
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// If i is beyond the end of the array, do nothing
	if(i >= count) return;

	// Otherwise, multiply the ith element in a by s and 
	// store the result in the ith element of c
	c[i] = a[i] * s;
}

int main(int argc, char **argv)
{
	// Seed the random number generator
	srand(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());

	// Allocate a new buffer in host memory for the source data
	float *hostSrcBuf = new float[ARRAYSIZE];
	// Populate the buffer with random floats between 0.0 and 1.0
	size_t n;
	n = ARRAYSIZE;
	while(n--) hostSrcBuf = ((float) rand()) / (float) RAND_MAX;

	// Allocate a new buffer in host memory for the result data
	float *hostResBuf = new float[ARRAYSIZE];

	// Allocate equivalent buffers on the GPU
	size_t byteCount = ARRAYSIZE * sizeof(float);
	float *devSrcBuf = nullptr;
	cudaMalloc((void **) &devSrcBuf, byteCount);
	float *devResBuf = nullptr;
	cudaMalloc((void **) &devResBuf, byteCount);

	// Transfer the data
	cudaMemcpy(devSrcBuf, hostSrcBuf, byteCount, cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int numberOfBlocks = (ARRAYSIZE + threadsPerBlock - 1) / threadsPerBlock;

	// Execute the kernel
	cuda_vec_scale<<<numberOfBlocks, threadsPerBlock>>>(devSrcBuf, 1324.0f, devResBuf, ARRAYSIZE);
	
	//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	// Retrieve the results
	cudaMemcpy(hostResBuf, devResBuf, byteCount, cudaMemcpyDeviceToHost);
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	
	return 0;
}
```

Next, let's validate the results, to make sure our kernel worked. To avoid floating point precision errors from invalidating the code, instead of testing for equality, let's compare against a minimum error.

```cpp
#include <stdlib.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define ARRAYSIZE 8096

// CUDA Kernel
__global__
void cuda_vec_scale(const float* a, float s, float* c, int count)
{
	// Calculate a linear index and store it in variable i
	// This is also known as the "thread id"
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// If i is beyond the end of the array, do nothing
	if(i >= count) return;

	// Otherwise, multiply the ith element in a by s and 
	// store the result in the ith element of c
	c[i] = a[i] * s;
}

int main(int argc, char **argv)
{
	// Seed the random number generator
	srand(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());

	// Allocate a new buffer in host memory for the source data
	float *hostSrcBuf = new float[ARRAYSIZE];
	// Populate the buffer with random floats between 0.0 and 1.0
	size_t n;
	n = ARRAYSIZE;
	while(n--) hostSrcBuf = ((float) rand()) / (float) RAND_MAX;

	// Allocate a new buffer in host memory for the result data
	float *hostResBuf = new float[ARRAYSIZE];

	// Allocate equivalent buffers on the GPU
	size_t byteCount = ARRAYSIZE * sizeof(float);
	float *devSrcBuf = nullptr;
	cudaMalloc((void **) &devSrcBuf, byteCount);
	float *devResBuf = nullptr;
	cudaMalloc((void **) &devResBuf, byteCount);

	// Transfer the data
	cudaMemcpy(devSrcBuf, hostSrcBuf, byteCount, cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int numberOfBlocks = (ARRAYSIZE + threadsPerBlock - 1) / threadsPerBlock;

	// Execute the kernel
	cuda_vec_scale<<<numberOfBlocks, threadsPerBlock>>>(devSrcBuf, 1324.0f, devResBuf, ARRAYSIZE);
	
	// Retrieve the results
	cudaMemcpy(hostResBuf, devResBuf, byteCount, cudaMemcpyDeviceToHost);

	//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	// Validate the results
	bool valid = true;
	n = ARRAYSIZE;
	while(n--)
		if(fabs((hostResBuf[n] / 1324.0f) - hostSrcBuf[n]) > 0.000001f) {
			valid = false;
			break;
		}
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	
	return 0;
}
```

And then we can print the result of the validation.

```cpp
#include <stdlib.h>
#include <chrono>
//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#include <iostream>
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

using namespace std;
using namespace std::chrono;

#define ARRAYSIZE 8096

// CUDA Kernel
__global__
void cuda_vec_scale(const float* a, float s, float* c, int count)
{
	// Calculate a linear index and store it in variable i
	// This is also known as the "thread id"
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// If i is beyond the end of the array, do nothing
	if(i >= count) return;

	// Otherwise, multiply the ith element in a by s and 
	// store the result in the ith element of c
	c[i] = a[i] * s;
}

int main(int argc, char **argv)
{
	// Seed the random number generator
	srand(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());

	// Allocate a new buffer in host memory for the source data
	float *hostSrcBuf = new float[ARRAYSIZE];
	// Populate the buffer with random floats between 0.0 and 1.0
	size_t n;
	n = ARRAYSIZE;
	while(n--) hostSrcBuf = ((float) rand()) / (float) RAND_MAX;

	// Allocate a new buffer in host memory for the result data
	float *hostResBuf = new float[ARRAYSIZE];

	// Allocate equivalent buffers on the GPU
	size_t byteCount = ARRAYSIZE * sizeof(float);
	float *devSrcBuf = nullptr;
	cudaMalloc((void **) &devSrcBuf, byteCount);
	float *devResBuf = nullptr;
	cudaMalloc((void **) &devResBuf, byteCount);

	// Transfer the data
	cudaMemcpy(devSrcBuf, hostSrcBuf, byteCount, cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int numberOfBlocks = (ARRAYSIZE + threadsPerBlock - 1) / threadsPerBlock;

	// Execute the kernel
	cuda_vec_scale<<<numberOfBlocks, threadsPerBlock>>>(devSrcBuf, 1324.0f, devResBuf, ARRAYSIZE);
	
	// Retrieve the results
	cudaMemcpy(hostResBuf, devResBuf, byteCount, cudaMemcpyDeviceToHost);

	// Validate the results
	bool valid = true;
	n = ARRAYSIZE;
	while(n--)
		if(fabs((hostResBuf[n] / 1324.0f) - hostSrcBuf[n]) > 1E6) {
			valid = false;
			break;
		}
	
	//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	// Report
	if(valid)
		cout << "CUDA vector scale results valid.\n";
	else
		cout << "CUDA vector scale results invalid.\n";
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	return 0;
}
```

And finally, be kind, rewind.

```cpp
#include <stdlib.h>
#include <chrono>
//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#include <iostream>
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

using namespace std;
using namespace std::chrono;

#define ARRAYSIZE 8096

// CUDA Kernel
__global__
void cuda_vec_scale(const float* a, float s, float* c, int count)
{
	// Calculate a linear index and store it in variable i
	// This is also known as the "thread id"
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// If i is beyond the end of the array, do nothing
	if(i >= count) return;

	// Otherwise, multiply the ith element in a by s and 
	// store the result in the ith element of c
	c[i] = a[i] * s;
}

int main(int argc, char **argv)
{
	// Seed the random number generator
	srand(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());

	// Allocate a new buffer in host memory for the source data
	float *hostSrcBuf = new float[ARRAYSIZE];
	// Populate the buffer with random floats between 0.0 and 1.0
	size_t n;
	n = ARRAYSIZE;
	while(n--) hostSrcBuf = ((float) rand()) / (float) RAND_MAX;

	// Allocate a new buffer in host memory for the result data
	float *hostResBuf = new float[ARRAYSIZE];

	// Allocate equivalent buffers on the GPU
	size_t byteCount = ARRAYSIZE * sizeof(float);
	float *devSrcBuf = nullptr;
	cudaMalloc((void **) &devSrcBuf, byteCount);
	float *devResBuf = nullptr;
	cudaMalloc((void **) &devResBuf, byteCount);

	// Transfer the data
	cudaMemcpy(devSrcBuf, hostSrcBuf, byteCount, cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int numberOfBlocks = (ARRAYSIZE + threadsPerBlock - 1) / threadsPerBlock;

	// Execute the kernel
	cuda_vec_scale<<<numberOfBlocks, threadsPerBlock>>>(devSrcBuf, 1324.0f, devResBuf, ARRAYSIZE);
	
	// Retrieve the results
	cudaMemcpy(hostResBuf, devResBuf, byteCount, cudaMemcpyDeviceToHost);

	// Validate the results
	bool valid = true;
	n = ARRAYSIZE;
	while(n--)
		if(fabs((hostResBuf[n] / 1324.0f) - hostSrcBuf[n]) > 1E6) {
			valid = false;
			break;
		}
	
	// Report
	if(valid)
		cout << "CUDA vector scale results valid.\n";
	else
		cout << "CUDA vector scale results invalid.\n";
	
	//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	// Good C++ citizens houseclean
	cudaFree(devResBuf);
	cudaFree(devSrcBuf);
	delete [] hostResBuf;
	delete [] hostSrcBuf;
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	
	return 0;
}
```

Save your code as `main.cu` and drop out to the command line. To compile and link this code, type the following at the shell prompt.

```bash
nvcc -ccbin c++ -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o main.cpp.o -c main.cu
nvcc -ccbin c++ -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o vec_scale main.cpp.o
```

Here are a few items of note:

- nvcc calls CUDA tools to compile the CUDA specific code and then passes the rest of the code off to the regular C++ compiler and linker. Hence the `-ccbin c++` option, which basically informs the nvcc how to invoke the host tools.
- The `-gencode` options specify what compute level to target. Each generation of NVIDIA GPU has offered successively more features, and the API has evolved accordingly. The compiler accomplishes compatibility with GPUs with differing feature sets by bundling multiple versions of the code.
- The first line compiles the source into a binary object file, which theoretically can be linked using host link tools. I only invoke nvcc in the second line to avoid having to specify the CUDA libraries to link against.

To run the code, execute the following on the command line.

```bash
./vec_scale
```

Here's the code with some proper error checking.

```cpp
#include <stdlib.h>
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

#define ARRAYSIZE 8096

// CUDA Kernel
__global__
void cuda_vec_scale(const float* a, float s, float* c, int count)
{
	// Calculate a linear index and store it in variable i
	// This is also known as the "thread id"
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// If i is beyond the end of the array, do nothing
	if(i >= count) return;

	// Otherwise, multiply the ith element in a by s and 
	// store the result in the ith element of c
	c[i] = a[i] * s;
}

bool handleError(cudaError_t iErr)
{
    if(iErr != cudaSuccess) {
        cerr << "CUDA error: " << iErr << " " << cudaGetErrorString(iErr) << endl;
        return false;
    }
    return true;
}

int main(int argc, char **argv)
{
	// Seed the random number generator
	srand(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());

	// Allocate a new buffer in host memory for the source data
	float *hostSrcBuf = new float[ARRAYSIZE];
	if(!hostSrcBuf) {
	  cerr << "Failed to allocate buffer for source data\n";
	  exit(-1);
	}

	// Populate the buffer with random floats between 0.0 and 1.0
	size_t n;
	n = ARRAYSIZE;
	while(n--) hostSrcBuf[n] = ((float) rand()) / (float) RAND_MAX;

	// Allocate a new buffer in host memory for the result data
	float *hostResBuf = new float[ARRAYSIZE];
	if(!hostResBuf) {
	  delete [] hostSrcBuf;
	  cerr << "Failed to allocate buffer for result data\n";
	  exit(-1);
	}
	
	// Allocate equivalent buffers on the GPU
	size_t byteCount = ARRAYSIZE * sizeof(float);
	float *devSrcBuf = nullptr;
	if(!handleError(cudaMalloc((void **) &devSrcBuf, byteCount))) {
	  delete [] hostResBuf;
	  delete [] hostSrcBuf;
	  exit(-1);
	}

	float *devResBuf = nullptr;
	if(!handleError(cudaMalloc((void **) &devResBuf, byteCount))) {
	  cudaFree(devSrcBuf);
	  delete [] hostResBuf;
	  delete [] hostSrcBuf;
	  exit(-1);
	}
	
	// Transfer the data
	if(!handleError(cudaMemcpy(devSrcBuf, hostSrcBuf, byteCount, cudaMemcpyHostToDevice))) {
	  cudaFree(devResBuf);
	  cudaFree(devSrcBuf);
	  delete [] hostResBuf;
	  delete [] hostSrcBuf;
	  exit(-1);
	}
	
	int err = -1;
	int threadsPerBlock = 256;
	int numberOfBlocks = (ARRAYSIZE + threadsPerBlock - 1) / threadsPerBlock;

	// Execute the kernel
	cuda_vec_scale<<<numberOfBlocks, threadsPerBlock>>>(devSrcBuf, 1324.0f, devResBuf, ARRAYSIZE);
	if(handleError(cudaGetLastError())) {
	  
	  // Retrieve the results
	  if(handleError(cudaMemcpy(hostResBuf, devResBuf, byteCount, cudaMemcpyDeviceToHost))) {
			
			// Validate the results
			bool valid = true;
			n = ARRAYSIZE;
			while(n--)
			  if(fabs((hostResBuf[n] / 1324.0f) - hostSrcBuf[n]) > 1E6) {
		      valid = false;
		      break;
			  }
			
			// Report
			if(valid) {
				err = 0;
			  cout << "CUDA vector scale results valid.\n";
			} else
			  cout << "CUDA vector scale results invalid.\n";
	  }
	}

	// Good C++ citizens houseclean
	cudaFree(devResBuf);
	cudaFree(devSrcBuf);
	delete [] hostResBuf;
	delete [] hostSrcBuf;
	
	return err;
}
```
