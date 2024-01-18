

> 



## CUDA programming Masterclass with C++

> https://www.udemy.com/course/cuda-programming-masterclass/

## 01 Introduction to CUDA programming and CUDA programming model

- single core通过context（上下文）switching机制“同时”运行多个process，是concurrency（并发）。只是在软件层面上有并行执行的错觉

- multi core不需要context，其使用task level和data level parallelism（并行）

- CPU和GPU除了核心数量差距大，其他差距很小

    | GPU                                            | CPU                                                          |
    | ---------------------------------------------- | ------------------------------------------------------------ |
    | 硬件的context switching                        | 软件的context switching                                      |
    | Can switch between thread if one thread stalls | For memory instruction latencies with L1 and L2 cache misses,thread are going to stall |
    | 硬件的线程调度器和调度单元                     | 软件的线程调度器和调度单元                                   |

- CUDA是Compute unified Device Architecture的缩写

- GPGPU: General Purpose Computing in Graphic Processing Unit

- [Wiki](https://zh.wikipedia.org/wiki/CUDA#%E6%98%BE%E5%8D%A1%E7%9A%84%E5%8F%97%E6%94%AF%E6%8C%81%E6%83%85%E5%86%B5)上可查看CUDA、GPU、架构对应信息
