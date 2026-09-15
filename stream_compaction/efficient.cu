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

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int padded_size = 1 << ilog2ceil(n);
            int* kern_input;
            cudaMalloc((void**)&kern_input, padded_size * sizeof(int));

            cudaMemset(kern_input, 0, padded_size * sizeof(int));
            cudaMemcpy(kern_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            //scanEfficient(paddedN, dev_data);
            timer().endGpuTimer();

            cudaMemcpy(odata, kern_input, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            
        }
    }
}
