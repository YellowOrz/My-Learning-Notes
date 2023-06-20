# [NVIDIA CUDA初级教程视频](https://www.bilibili.com/video/BV1kx411m7Fk)

## [1.CPU体系架构概述](https://www.easyhpc.net/course/26/lesson/281/material/356)

- 编译好的程序，最优化目标 = CPI(每条指令的时钟数) * 时钟周期
- Pipelining：利用指令集的并行instruction-level parallelism(ILP)
    - \+ 极大的减小时钟周期
    - \- 增加一些延迟和芯片面积
    - 流水线长度：Alleged Pipeline Length
    - Bypassing旁路：将之前的数据临时开一个小路先送到后面去

5.GPU编程模型
6.CUDA编程（1）
7.CUDA编程（2）
8.CUDA编程（3）
9.CUDA程序分析和调试工具
10.CUDA程序基本优化
11.CUDA程序深入优化
12.CUDA Fortran 介绍 1
13.CUDA Fortran 介绍 2
14.cuDNN
15.SimpleNNwithCUDA
