# CUDA编程：基础与实践

> 豆瓣：https://book.douban.com/subject/35252459/

注：之前简单学过一遍CUDA，我认为很简单的内容就没再记录了

## 总结

### 特性

- **网格与线程块大小的限制**：对任何从开普勒到图灵架构的GPU
    - grid，在x、y和z的最大允许值分别为 $2^{31}-1$、65535和65535
    - block，在x、y和z的最大允许值分别为 1024、1024和64，<u>且x、y、z的乘积≤1024</u>
    
- NVCC的**化简编译选项**：

    ```bash
    -arch=sm_XY
    # 等价于
    -gencode arcn＝compute_XY,code＝sm_XY
    -gencode arcn＝compute_XY,code＝sm_XY
    ```

- 所有CUDA runtime API都以cuda开头

### 编程习惯

- device的变量以`d_`开头，host的以`h_`开头
- 推荐，写了cudaMalloc后马上写cudaFree
- `cudaMemcpy()`可以起到隐式同步host和device的作用
- CHECK宏函数不能用于cudaEventQuery函数
- 调试时，可以设置环境变量`CUDA_LAUNCH_BLOCKING=1`将核函数的调用设置成同步（即host调用核函数，必须等它都执行完了才走下一步）
- 内存错误检查：可执行文件cuda-memcheck

### 优化技巧

- 提升利用率：总的线程数>计算核心数
- 

## 第1章 GPU硬件与CUDA程序开发工具

- CPU中有更多的晶体管用于数据缓存和流程控制，GPU中有更多的晶体管用于算术逻辑单元

- **==GPU计算==**：CPU+GPU的异构（heterogeneous）计算

- **==计算能力（compute capability）==**：形式为`X.Y`，前为主版本，后为次版本

    - 计算能力和性能没有简单的正比关系
    - 版本号越大，GPU架构（architecture）越新，∵主版本号与GPU的核心架构相关

    ![image-20230620171424897](images/image-20230620171424897.png)

    > 查看各GPU的计算能力：https://developer.nvidia.cn/zh-cn/cuda-gpus#compute

- 表征计算能力的参数：

    - 浮点数运算峰值（floating-point operations per second，FLOPS）
        - Tesla系列，双精度FLOPs是单精度的1/2左右，GeForce系列为1/32
    - GPU中的内存带宽 （memory bandwidth）

- CUDA提供2层API：性能上几乎没有差别

    - CUDA驱动（driver）API：更底层，提供了更为灵活的编程接口。
        - 在其他编程语言中使用CUDA，必须用这个
    - ==CUDA运行时（run time）API==：在CUDA驱动API的基础上构建，更高级、容易使用，都以cuda开头

- CUDA和计算能力的关系：CUDA版本是GPU软件开发平台的版本，而计算能力对应着 GPU硬件架构的版本

    ![image-20230620172148690](images/image-20230620172148690.png)

- 从CUDA 10.2开始，`CUDA C`改名为`CUDA C++`

- nvidia-smi信息：

    - GPU模式：

        - WDDM（Windows display driver model）模式
        - TCC（Tesla compute cluster）：仅在Tesla、Quadro和Titan系列的可选

        ```bash
        sudo nvidia-smi -g [GPU_ID] -dm 0 ＃设置为WDDM模式
        sudo nvidia-smi -g [GPU_ID] -dm 1 ＃设置为TCC模式
        ```

    - 计算模式（compute mode）

        ```bash
        sudo nvidia-smi -g [GPU_ID] -c 0 ＃默认模式
        sudo nvidia-smi -g [GPU_ID] -c 1 ＃独占进程（exclusive process mode，E.Process）
        ```

## 第2章 CUDA中的线程组织

- 核函数定义，`__global__`和void顺序随意

- 核函数不支持C++的`iostream`

- 核函数中有`printf`，调用后需要使用代码`cudaDeviceSynchr○nize();`

    - ∵核函数调用`printf`，输出流是先存放在缓冲区的，而缓冲区不会自动刷新。只有程序遇到某种同步操作时缓冲区才会刷新

- 提升利用率：总的线程数>计算核心数（一般为几倍）

    - ∵计算和内存访问之间及不同的计算之间合理地重叠，从而减小计算核心空闲的时间
    - 在执行时能够同时活跃（不活跃的线程处于等待状态）的线程数，由硬件（主要是CUDA核心数）和软件 （即核函数中的代码）决定

- **网格与线程块大小的限制**：对任何从开普勒到图灵架构的GPU

    - grid，在x、y和z的最大允许值分别为 $2^{31}-1$、65535和65535
    - block，在x、y和z的最大允许值分别为 1024、1024和64，<u>且x、y、z的乘积≤1024</u>

- CUDA的头文件

    - 使用nvcc编译.cu，自动包含必要的CUDA头文件，如`<cuda.h>`和` <cuda_runt1me.h>`
    - `<cuda.h>`包含`<stdlib.h>`

- **NVCC编译.cu**，包含3个步骤：

    - 步骤零：将全部源代码分离为主机代码（支持C++，交给gcc等编译器）和设备代码（部分地支持C++，继续由nvcc）
    - 步骤一：编译为PTX（parallel thread execution）伪汇编代码，使用选项`-arch＝compute_XY`指定虚拟架构的计算能力为X.Y，用以确定代码中能够使用的CUDA功能
    - 步骤二：再将PTX代 码编译为二进制的cubin目标代码，使用选项`-code＝sm_ZW`指定真实架构的计算能力为Z.W，用以确定可执行文件能够使用的GPU

    > 注意：
    >
    > - ZW必须大于XY
    > - 编译出的程序只能运行 在计算能力的主版本号=Z、次版本号≥W 的GPU上

- ==胖二进制文件（fatbinary）==：同时指定多组计算能力，可以用于更多的GPU（例如下），但是会增加编译时间&&可执行文件大小

    ```bash
    -gencode arcn＝compute_35,code＝sm_35
    -gencode arcn＝compute_50,code＝sm_50
    -gencode arcn＝compute_60,code＝sm_60
    -gencode arcn＝compute_70,code＝sm_70
    ```

- NVCC的==即时编译（just-in-time compilation）机制==：在运行可执行文件时从其中保留的PTX代码临时编译出一个cubin目标代码，需要使用如下方式保留PTX代码的虚拟架构

    ```bash
    -gencode arcn＝compute_XY,code＝compute_XY	# 注意：前后都是compute
    ```

- NVCC的**化简编译选项**：

    ```bash
    -arch=sm_XY
    # 等价于
    -gencode arcn＝compute_XY,code＝sm_XY
    -gencode arcn＝compute_XY,code＝sm_XY
    ```

- NVCC编译不指定计算能力，用默认的

    - CUDA 6.0及更早，默认为1.0。无法在核函数中使用printf
    - CUDA 6.5~8.0，默认为2.0
    - CUDA 9.0~10.2，默认为3.0

## 第3章 简单CUDA程序的基本框架

- 判断两个float是否相等时,不能用==。而要将这两个数的差的绝对值与—个很小的数进行比较

- 隐形的设备初始化：CUDA runtime API中，在第一次调用一个和设备管理及版本查询功能无关的run time API函数时，设备将自动初始化。

- 所有CUDA runtime API都以cuda开头

- device上**内存分配和释放**：在host中调用如下函数

    ```c++
    /* 内存分配 */
    // 定义
    __host__ cudaError_t cudaMalloc (
    	void **devPtr, 	// 传入双重指针，∵为了改变指针的值（指向其他地方），而不是改变指针所指内存的值
    	size_t  size );
    // 示例
    double *d_x;
    cudaMalloc((void **)&d_x, 1000);	// (void **)为强制类型转换（可以不明确写出来）
    /* 内存释放 */
    // 定义
    __host__ __device__ cudaError_t cudaFree ( void* devPtr );
    // 示例
    cudaFree(d_x);
    ```

    > - cudaMalloc和cudaFree一定要成对使用。**推荐，写了cudaMalloc后马上写cudaFree**
    > - [不建议] CUDA 2.0以及之后，可以在核函数内部使用`malloc()`和`free()`来分配 动态全局内存，但是性能较差

- 编程习惯：device的变量以`d_`开头，host的以`h_`开头

- 数据拷贝：

    ```c++
    __host__ cudaError_t cudaMemcpy ( 
        void* dst, 
        const void* src, 
        size_t count, 
        cudaMemcpyKind kind 	// 数据传递方向
    );
    ```

    > `cudaMemcpyKind`包含5种：`cudaMemcpyHostToDevice`、`cudaMemcpyDeviceToHost`、`cudaMemcpyHostToHost`、`cudaMemcpyDeviceToDevice`、`cudaMemcpyDefault`（根据指针自动判断传输方向，要求64位系统&&==统一虚拟寻址(unified virtual addressing)==）

- 核函数要求

    - 必须使用限定符`__global__`，跟其他C++中的限定符（如static）次序任意
    - 支持C++中的重载（overload）
    - 不支持可变数量的参数列表
    - 可传递非指针变量（如int n），其内容对每个线程可见
    - 除非使用统一内存（见第12章），否则传给核函数的数组 （指针）必须指向device内存
    - 核函数不可成为—个类的成员。通常的做法，用—个包装函数调用核函数，而将包装函数定义为类的成员
    - 从计算能力3.5开始, 引入==动态并行（dynamic parallelism）==机制，使得核函数可以调用其他核函数（包括自己）

- CUDA标识符

    - `__global__`：表示核函数。一般由host调用，在device上执行；通过动态并行，也可以在其他核函数中调用
    - `__device__`：表示==设备函数(device function)==。只能由核函数or其他设备函数调用，在device上执行。可以有return
    - `__host__`：表示host上的普通c++函数。可以省略
    - `__noinline__`：建议一个设备函数为非内联（编译器不一定接受）
    - `__forceinline__`：建议一个设备函数为内联

    > - 可以同时使用`__device__`和`__host__`，表示在host和device都可以运行，减少冗余代码
    > - 不能同时使用`__device__`和`__global__`
    > - 不能同时使用`__host__`和`__global__`

## 第4章 CUDA程序的错误检测

- 方法一：使用宏定义

    ```c
    #pragma once
    #include <stdio.h>
    
    #define CHECK(call)                                   \
    do {                                                  \
        const cudaError_t error_code = call;              \
        if (error_code != cudaSuccess) {                  \
            printf("CUDA Error:\n");                      \
            printf("    File:       %s\n", __FILE__);     \
            printf("    Line:       %d\n", __LINE__);     \
            printf("    Error code: %d\n", error_code);   \
            printf("    Error text: %s\n",                \
                cudaGetErrorString(error_code));          \
            exit(1);                                      \
        }                                                 \
    } while (0)
    ```

    > - 使用do-while是为了安全（没说为啥）
    > - **不能用于cudaEventQuery函数**，∵可能返回cudaErrorNotReady，但不代表程序错了

- 以上方法不能用于核函数，因为没有return，需要在调用核函数后加两句

    ```c++
    CHECK(cudaGetLastError());			// 捕捉cudaDeviceSynchronize之前的最后一个错误
    CHECK(cudaDeviceSynchronize());		// 同步host和device
    ```

    > 注意：`cudaDeviceSynchronize`比较耗时，不在内层循环使用

- `cudaMemcpy()`可以起到隐式同步host和device的作用

- 调试时，可以设置环境变量`CUDA_LAUNCH_BLOCKING=1`将核函数的调用设置成同步（即host调用核函数，必须等它都执行完了才走下一步）

- 内存错误检查：使用**CUDA-MEMCHECK**工具集，包含memcheck、racecheck、 jnitcheck、synccheck四个工具，由可执行文件cuda-memcheck调用

    ```c++
    cuda-memcheck --tool memcheck [options] app_name [app_options]
    // 上面可化简为 cuda-memcheck [options] app_name [app_options]
    cuda-memcheck --tool racecheck [options] app_name [app_options]
    cuda-memcheck --tool jnitcheck [options] app_name [app_options]
    cuda-memcheck --tool synccheck [options] app_name [app_options]
    ```

## 第5章 获得GPU加速的关键

- 基于CUDA event的计时方式

    ```c++
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));		// 初始化
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));		// 记录开始
    cudaEventQuery(start);		// 不能用CHECK宏定义。对于TCC驱动模式的GPU可省略，WDDM的必须保留，原因见下
    // TODO
    CHECK(cudaEventRecord(stop));		// 记录结束
    CHECK(cudaEventSynchronize(stop));	// 等待记录完成
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Time = %g ms.\n", elapsed_time);
    ```

    > - 处于WDDM驱动模式的GPU中，一个==CUDA流（CUDA stream）==中的操作（如cudaEventRecord函数）并不是直接提交给GPU执行，而是先提交到一个软件队列，需要添加一条对该流的 cudaEventQuery操作（或者cudaEventSynchromze）刷新队列，才能促使前面的操作在GPU执行



