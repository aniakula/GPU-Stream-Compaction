CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Anirudh Akula
  * [linkedIn](https://www.linkedin.com/in/anirudh-akula/)
* Tested on: Windows 11, NVIDIA T1000 4096MB (CETS Virtual PC)

## Stream Compaction Analysis

### Choice of Block Size:

<img width="1510" height="622" alt="image" src="https://github.com/user-attachments/assets/0d7cb13d-b00a-4c73-a5ac-edee0ca39cb1" />

### Comparison of Scan Implementations:
<img width="1514" height="1004" alt="image" src="https://github.com/user-attachments/assets/798918d5-c8ce-4deb-a597-b82204151f4e" />

#### Performance Bottleneck Analysis

CPU scan wins at the array size tested (n = ~6,553) simply because it avoids overhead of kernel launch, memory transfer setup, and data fits in cache. GPU scan methods have a lot more of these launch overheads as we explore below.

Naive GPU scan is likely bottlenecked by kernel launch overhead. It needs log(n) = ~13 launches, each with fixed overhead regardless of work done. At this array size, memory traffic is trivial so the fixed launch cost dominates.

Work-efficient GPU scan is bottlenecked by launch overhead even more, which is maybe why it is slower than naive here despite doing less total arithmetic (O(n) vs O(n log n)). It needs ~2× the launches (~26 total), and later stage launches have fewer active threads paying full launch overhead little work.

Thrust beats both versions likely by leveraging shared memory instead of global memory reads and writes as done in the hand written methods. (could not perform NSight analysis due to unavailabillity of a physical Nvidia computer).
