#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        //each level of naive sacn (runs log(n) times)
        __global__ void kernNaiveScan(int n, int stride, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;

            if (index >= stride) {
                odata[index] = idata[index] + idata[index - stride];
            }
            else {
                odata[index] = idata[index];
            }
        }

        //shift an inclusive scan right by one to make it exclusive
        __global__ void kernShiftRight(int n, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;

            odata[index] = (index == 0) ? 0 : idata[index - 1];
        }

        void scan(int n, int* odata, const int* idata) {
            int* ping_buffer;
            int* pong_buffer;
            cudaMalloc((void**)&ping_buffer, n * sizeof(int));
            cudaMalloc((void**)&pong_buffer, n * sizeof(int));

            cudaMemcpy(ping_buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            int steps = ilog2ceil(n);
            for (int d = 1; d <= steps; d++) {
				int stride = 1 << (d - 1); // 2^(d-1)
                kernNaiveScan << <((n + blockSize - 1) / blockSize), blockSize >> > (n, stride, pong_buffer, ping_buffer);

                std::swap(ping_buffer, pong_buffer);
            }

            //inclusive to exclusive
            kernShiftRight << <((n + blockSize - 1) / blockSize), blockSize >> > (n, pong_buffer, ping_buffer);

            timer().endGpuTimer();

            cudaMemcpy(odata, pong_buffer, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(ping_buffer);
            cudaFree(pong_buffer);
        }
    }
}