CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Anirudh Akula
  * [linkedIn](https://www.linkedin.com/in/anirudh-akula/)
* Tested on: Windows 11, NVIDIA T1000 4096MB (CETS Virtual PC)

## Stream Compaction

### What is Scanning?
Exclusive scan (prefix sum) takes an array and produces a new array where each element is the sum of all elements before it in the original array not including itself. For example, given [1, 2, 3, 4], an exclusive scan produces [0, 1, 3, 6].

Two scan algorithms are compared in this project:

Naive scan performs log(n) passes over the array, where each pass adds each element to another element a fixed stride away, and doubles the stride each iteration. It performs O(n log n) additions.

<img width="1062" height="530" alt="image" src="https://github.com/user-attachments/assets/504cee7d-8e8e-47ef-9cf6-93695e80ee08" />

* image source [GPU Gems](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

Work-efficient scan does two passes: an up-sweep phase computes partial sums followed by a down-sweep phase that traverses back down the tree distributing those partial sums to produce the final scan. This reduces total work to O(n) at the cost of needing twice as many total kernel launches compared to naive scan.

<img width="1024" height="488" alt="image" src="https://github.com/user-attachments/assets/4cd564f9-4bb7-40e5-b1dd-72b59bf02447" />

<img width="1022" height="620" alt="image" src="https://github.com/user-attachments/assets/6c3452df-76c2-4fe2-a29a-7c2eaa417014" />

* images source [GPU Gems](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)


## Features

* Implemented a CPU version of scan and compact
* Implemented a GPU naive version of scan and compact
* Implemented a GPU work efficient version of scan and compact
* Wrote a wrapper around the thrust library implementation of scan
* Benchmarked all the implementations and analysed results (below)

## Analysis

### Choice of Block Size
Below is a graph comparing block size and scan implementation run times. Given these results I chose to use a block size of 256 since it seemed to on average have the least latency between the two implementations. The other candidate was a block size of 512 but it had much higher latency in the naive case and is within 0.02 ms of the 256 size run for the work efficient scan.

<img width="1510" height="622" alt="image" src="https://github.com/user-attachments/assets/0d7cb13d-b00a-4c73-a5ac-edee0ca39cb1" />

### Comparison of Scan Implementations:
<img width="1514" height="1004" alt="image" src="https://github.com/user-attachments/assets/798918d5-c8ce-4deb-a597-b82204151f4e" />

#### Performance Bottleneck Analysis

CPU scan wins at the array size tested (n = ~6,553) simply because it avoids overhead of kernel launch, memory transfer setup, and data fits in cache. GPU scan methods have a lot more of these launch overheads as we explore below.

Naive GPU scan is likely bottlenecked by kernel launch overhead. It needs log(n) = ~13 launches, each with fixed overhead regardless of work done. At this array size, memory traffic is trivial so the fixed launch cost dominates.

Work-efficient GPU scan is bottlenecked by launch overhead even more, which is maybe why it is slower than naive here despite doing less total arithmetic (O(n) vs O(n log n)). It needs ~2× the launches (~26 total), and later stage launches have fewer active threads paying full launch overhead little work.

Thrust beats both versions likely by leveraging shared memory instead of global memory reads and writes as done in the hand written methods. (could not perform NSight analysis due to unavailabillity of a physical Nvidia computer).

### Program Test Output
```
****************
** SCAN TESTS **
****************
    [  38  20   6   4  22  44  41   6  26  12  25  17  31 ...  22   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0004ms    (std::chrono Measured)
    [   0  38  58  64  68  90 134 175 181 207 219 244 261 ... 6244 6266 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0004ms    (std::chrono Measured)
    [   0  38  58  64  68  90 134 175 181 207 219 244 261 ... 6181 6185 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.114688ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.049152ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.225952ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.092224ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.07584ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.033088ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   0   2   0   0   0   1   2   0   2   1   3   3 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0007ms    (std::chrono Measured)
    [   2   1   2   2   1   3   3   3   1   3   3   1   2 ...   2   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0007ms    (std::chrono Measured)
    [   2   1   2   2   1   3   3   3   1   3   3   1   2 ...   1   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0012ms    (std::chrono Measured)
    [   2   1   2   2   1   3   3   3   1   3   3   1   2 ...   2   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.182272ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.151584ms    (CUDA Measured)
    passed
```
