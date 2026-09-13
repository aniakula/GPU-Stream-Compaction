#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
			odata[0] = 0;
            for (int i = 0; i < n-1; i++) {
				odata[i + 1] = odata[i] + idata[i];
            }
            timer().endCpuTimer();
        }

		//remove timer so compactWithScan can use it without double timing
        void scan_no_timer(int n, int* odata, const int* idata) {
            odata[0] = 0;
            for (int i = 0; i < n - 1; i++) {
                odata[i + 1] = odata[i] + idata[i];
            }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
			int outputIndex = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[outputIndex] = idata[i];
					outputIndex++;
                }
            }
            timer().endCpuTimer();
            return outputIndex;
        }

        void scatter(int n, int *odata, const int *idata, const int *mask, const int *indices) {
            for (int i = 0; i < n; i++) {
                if (mask[i] == 1) {
                    odata[indices[i]] = idata[i];
                }
            }
		}

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int* mask = new int[n];
            int* outputIndices = new int[n];

            timer().startCpuTimer();
			//map to boolean array
            for (int i = 0; i < n; i++) {
                mask[i] = (idata[i] != 0) ? 1 : 0;
			}
			
			scan_no_timer(n, outputIndices, mask);
            scatter(n, odata, idata, mask, outputIndices);

            //if last elem is 0, last ind == num valid elems (index of the next valid elem if it existed)
            //if last elem is i, last ind == the index of the last valid elem so we add 1
			int count = outputIndices[n - 1] + mask[n - 1];
            timer().endCpuTimer();

            delete[] mask;
            delete[] outputIndices;
            return count;
        }
    }
}
