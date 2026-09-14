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

        __global__ void kernSinglePassNaiveScan(int n, int* odata, const int* idata, const int stride) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= stride) {
                odata[index] = idata[index] + idata[index - stride];
            } else {
                odata[index] = idata[index];
            }
        }

        /*
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            for (int i = 1; i < n; i *= 2) {
				kernSinglePassNaiveScan << <(n + blockSize - 1) / blockSize, blockSize>> > (n, odata, idata, i);
				int* temp = odata;
                odata = const_cast<int*>(odata);
				odata = temp;
            }
            timer().endGpuTimer();
        }
    }
}
