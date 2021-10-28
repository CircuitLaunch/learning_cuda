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
