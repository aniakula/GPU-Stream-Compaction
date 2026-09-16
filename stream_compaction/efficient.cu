#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int stride = 1 << (d + 1); // 2^(d+1)

            //find cell to do work on:
            int k = index * stride;

            if (k >= n) return;

            //in place addition
            data[k + stride - 1] += data[k + (stride >> 1) - 1];
        }

        __global__ void kernDownSweep(int n, int d, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            int stride = 1 << (d + 1);
            int k = index * stride;

            if (k >= n) return;

            int leftChild = k + (stride >> 1) - 1;
            int rightChild = k + stride - 1;

            int t = data[leftChild];
            data[leftChild] = data[rightChild];
            data[rightChild] += t;
        }

        void scanOnDevice(int n, int* dev_data) {
            int d_max = ilog2ceil(n);

            for (int d = 0; d < d_max; d++) {
                // calculate num active threads (prevent modulo use)
                int numThreads = n >> (d + 1);
                int gridSize = (numThreads + blockSize - 1) / blockSize;
                kernUpSweep << <gridSize, blockSize >> > (n, d, dev_data);
                checkCUDAError("kernUpSweep failed");
            }

            // zero the root before down-sweep.
            cudaMemset(dev_data + n - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset root failed");

            for (int d = d_max - 1; d >= 0; d--) {
                int numThreads = n >> (d + 1);
                int gridSize = (numThreads + blockSize - 1) / blockSize;
                kernDownSweep << <gridSize, blockSize >> > (n, d, dev_data);
                checkCUDAError("kernDownSweep failed");
            }
        }

        void scan(int n, int* odata, const int* idata) {
            int paddedN = 1 << ilog2ceil(n);
            int* dev_data;
            cudaMalloc((void**)&dev_data, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed");

            cudaMemset(dev_data, 0, paddedN * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata -> dev_data failed");

            timer().startGpuTimer();
            scanOnDevice(paddedN, dev_data);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_data -> odata failed");

            cudaFree(dev_data);
        }

        int compact(int n, int* odata, const int* idata) {
            int paddedN = 1 << ilog2ceil(n);

            int* dev_idata;
            int* dev_odata;
            int* mask;
            int* indices;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&mask, paddedN * sizeof(int));
            cudaMalloc((void**)&indices, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc failed in compact");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata to dev_idata failed");

            dim3 gridN((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            StreamCompaction::Common::kernMapToBoolean << <gridN, blockSize >> > (n, mask, dev_idata);
            checkCUDAError("kernMapToBoolean failed");

            cudaMemset(indices, 0, paddedN * sizeof(int));
            cudaMemcpy(indices, mask, n * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy mask to indices failed");

            scanOnDevice(paddedN, indices);

            StreamCompaction::Common::kernScatter << <gridN, blockSize >> > (n, dev_odata, dev_idata, mask, indices);
            checkCUDAError("kernScatter failed");

            timer().endGpuTimer();

            //save last mask and index elem to compute count
            int lastMask, lastIndex;
            cudaMemcpy(&lastMask, mask + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy lastMAsk and lastIndex failed");
            int count = lastMask == 0 ? lastIndex : lastIndex + 1;

            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata to odata failed");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(mask);
            cudaFree(indices);

            return count;
        }
    }
}