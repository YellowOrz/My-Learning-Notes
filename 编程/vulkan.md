> 《Vulkan学习指南》 <=> 《Learning Vulkan》

# 第1章 开始学习新一代3D图形API

## 1.3 重要术语
- physical device (物理设备): 各品牌型号的独显、集显等支持vulkan的硬件设备
- device (设备): 物理设备在应用程序中的逻辑表示。一个物理设备对应一个设备？
- queue (队列): 执行引擎与应用程序之间的接口。一个物理设备包含一个或多个队列。队列负责收集准备执行的工作（指令缓存）并分发到物理设备执行
- queue Family (队列族): 一组具有相同属性和能力的队列。一个队列族负责图形渲染、计算、数据传输、内存管理等操作中的一种。
- memory type (内存类型): 分为宿主内存和设备内存
- command (指令): 执行用户定义的行为。或者叫做命令
    - action command (动作指令): 包含绘制图源、清除表面、复制缓存、查询时间戳操作、以及子通道的开始和结束操作。用于修改帧缓存附件、读取或写入内存（缓存或者图像）以及写入查询池？
    - set state command (状态设计指令): 用来绑定流水线、描述符集合、缓存，或者设置 动态状态、渲染通道、子通道的状态
    - synchronization command (同步指令): 通过设置同步事件、等待事件、流水线屏障对象、渲染通道、子通道的依赖，来保证多个动作指令的同步
- command buffer (指令缓存): 一组指令的集合，记录多个指令并统一发送到队列中

## 1.4 vulkan的原理
- （简化的）执行模型：vulkan程序控制一组vulkan设备，将多个command记录到 多个command buffer 中，并发送到多个queue。设备的驱动会读取queue并按照记录的顺序依次执行各个command
    
    ![image-20240504170458134](images/image-20240504170458134.png)
    
    - 指令队列的构件需要代价 而一旦构建完成就可以 被缓存 和 发送到队列，根据自己的需要多次执行
    - 有些指令缓存支持以多线程的方式并行构建
    - vulkan程序还负责：各种准备（资源、着色器、流水线）、内存管理、同步、风险管理
    
- **queue（队列）**: 一种中间层机制，负责接收指令缓存并传递给设备。指令缓存的发送分成两类
    
    - 单一对列：按照指令缓存发送的顺序进行维护、执行或者回放
    - 多重队列：指令缓存 以并行的方式在多个队列中执行。除非有同步操作，否则无法保证发送和执行的顺序不变
    
- 同步
    
    - **semaphore (信号量)**: 跨队列，或在单一队列中以粗粒度执行？
    - **event (事件)**: 单个队列中，以细粒度执行，确保单一指令缓存中或多个指令缓存之间的同步要求
    - **fence (栅栏)**: 允许host和device之间同步
    - **pipeline barrier (流水线屏障)**: 插入command buffer中的指令，保证它之前的指令先执行，它之后的后执行
    
- vulkan对象：vulkan程序中的device、queue、command buffer、pipeline等对象，分为两类
    - 可分发的句柄：~~这类指针指向不透明的内部图形实体。不能直接访问成员，要通过API函数访问。~~
        - 包含vk::instance、vk::CommandBuffer、vk::PhysicalDevice、vk::Device、vk::Queue
    - 非可分发的句柄：这些64位整型类型的句柄不指向结构体，而是直接包含对象自身的信息
        - 包含vk::Pipeline、vk::PipelineCache、vk::PipelineLayout、vk::Buffer、vk::DeviceMemory、vk::QueryPool、vk::ShaderModule、vk::DescriptorPool、vk::DescriptorSet、vk::DescriptorSetLayout、vk::CommandPool、vk::Semaphore、vk::Fence、vk::Event等
    
- **指令语法**
    - 创建&&销毁：需要创建的对象使用`vk::Device::createXXX()`实现，需要一个结构体`vk::XXXCreateInfo`作为输入；销毁该对象则使用对应的`vk::Dvice::destoryXXX()`
        - 比如创建buffer，使用`vk::Device::createBuffer()`，输入为`vk::BufferCreateInfo`，销毁则使用`vk::Dvice::destoryBuffer()`
        - 包含：Instance、DeviceQueue、Device、CommandPool、DescriptorPool、DescriptorSetLayout、PipelineLayout、ShaderModule、PipelineShaderStage、ComputePipeline、Buffer 等
    - 分配&&释放：从已有对象池or堆中创建（分配）使用`vk::Device::allocateXXX()`实现，需要一个结构体`vk::XXXAllocateInfo`作为输入；释放该对象则使用对应的`vk::Dvice:freeXXX()`
        - 比如分配Memory，使用`vk::Device::allocateMemory()`，输入为`vk::MemoryAllocateInfo`，释放则使用`vk::Dvice::freeMemory()`
        - 包含：Memory、CommandBuffer、DescriptorSet 等
    - 上述所有的实现方法都可以通过`vk::getXXX()`获取  ？？？
    - 将指令记录到指令缓存中，使用`vk::cmdXXX()`  ？？？   

## 1.5 理解vulkan应用程序

![image-20240504171129813](images/image-20240504171129813.png)

- 驱动：支持vulkan的系统至少包含一个CPU和一个GPU。GPU的生产者会为某个vulkan标准提供完整的驱动实现。驱动为vulkan程序提供了高级的功能接口，使其可以与设备通信。例如，驱动可以找到系统中所有可用的设备、可用的队列类型等
- 应用程序（vulkan程序）：用户编写的、可以调用vulkan API执行图形or计算工作的程序。首先，初始化硬件和软件，可以检测驱动并找到所有可用的vulkan API。然后，创建资源并绑定到着色器阶段，会用到descriptor。descriptor辅助将创建后的资源绑定到底层（基于某种图形or计算类型）的pipeline。最后，记录command buffer并发送到queue执行
- WSI (Windows System Integration)： 将不同操作系统的展示层（presentation layer）统一起来
- SPIV-R：将不同着色器代码语言（HLSL、GLSL）转成相同的、预编译的二进制数据格式
- LunarG SDK：包含加载器、验证层、跟踪回放工具、SPIR-V工具、运行库、文档、demo等工具资源的vulkan skd

## 1.6 开始学习Vulkan编程模型



- 应用程序编程模型：采用自顶向下的实现过程

    ![image-20240504171817644](images/image-20240504171817644.png)

- 硬件初始化：应用程序需要与loader（加载器）进行通信来激活vulkan的驱动。

    ![image-20240505092654806](images/image-20240505092654806.png)

    - loader（加载器）：一段应用程序启动时执行的代码，它使用平台 无关的方式来定位系统中的Vulkan驱动
        - 负责：①定位并加载驱动；②保证API与系统无关；③ 支持层次化的结构，并且可以在运行过程中随时注入不同类型的层（例如 开发阶段打开所有需要注入的层，发布的时候关闭它们）
    - 注入层的功能：① 跟踪vulkan API的指令执行② 捕获渲染的场景 稍后再继续执行 ③ 满足调试需要，进行错误处理和验证
    - 加载器完成后，就可以①创建实例②查询物理设备上所有可用队列 ③支持注入层

- 资源设置：
    - 内存分类
    ① device local：在device上只对device可见的内存
    ② device local, host visible：在device上，但对device和host都可见
    ③ host local, host visible：在host上，对device和host都可见，比device local慢
    - 【推荐】**子分配**：应用程序提前申请一大块物理内存，然后将物理内存的很大一部分立即分配完成并存入不同的资源对象
    - sparse memory（稀疏内存）：将图像分割为多个小块 根据需求加载必需的，从而使得存储资源比实际的内存容量更大
    - staging buffer（阶段缓存）：应用程序先将资源设置到阶段缓存中，它对host是可见的，然后再传递到理想的存储区域（对host不可见）

- 流水线设置

     ![image-20240505092738504](images/image-20240505092738504.png)

     - **pipeline**：根据应用程序逻辑定义的一系列事件，照固定的顺序执行。包含设置着色器、资源的绑定以及状态管理

     - **descriptor set（描述符集）**：资源和着色器之间的接口。可以将着色器绑定到资源（例如image or buffer），也可以将资源内存关联或者绑定到准备使用的着色器实例上。它变化频繁，支持多线程同步更新，从**descriptor pool（描述符缓冲池）**分配而来

         - 更新或者改变描述符集是vulkan中关键的性能瓶颈之一。需要保证高频率更新的描述符 不会影响到低频率的描述符。

     - 基于SPIR-V的**shader（着色器）**：
         - 支持GLSL和HLSL等源语言转换成SPIR-V格式
         - shader的编译是离线的，不过预先就进行了注入
         - shader提供了多种不同的程序入口

     - **pipeline state（流水线状态）**：物理设备包含的一系列硬件设置，用来定义准备发送的几何输入数 据是如何解释和绘制的。

         - 包含光栅化状态、融混状态，以及深度/模板状态、输入 几何数据的图元拓扑类型（点/线/三角形）以及渲染所用的着色器
         - 流水线状态分为：动态状态和静态状态。<u>后者对于性能的优化来说至关重要。</u>
         - Vulkan允许用户使用**Pipeline（流水线）**与**Pipeline Cache（流水线缓存）**和**pipeline layout（流水线布局）**一起，来进行状态的控制
         - Pipeline Cache的实现由驱动完成

         > cache（缓存）和buffer（缓冲区）不同。cache是为了加快访问速度，buffer是为了减少响应次数

     - **pipeline layout（流水线布局）**：提供了pipeline中所用 的descriptor set，其中设置了各种不同的资源关联到着色器的不同方法。 不同的pipeline可以使用相同的pipeline layout。

     - 

- 指令的记录：

    ![image-20240505095627617](images/image-20240505095627617.png)

    - command（指令）的记录是逐渐构成指令缓存的过程
    - command buffer（指令缓存）是从command pool（指令池）当中分配而来的。command pool 可以用来同时分配多个command buffer
    - command buffer的创建是对性 能影响最大的一项操作
    - 也可以通过多线程的方式同步生 成多个command buffer。command pool 的设计确保了多线程环境下不会出现资源互锁的问题

- 队列的提交

    - Vulkan向应用程序暴露了不同类型的队列接口，例如图形、DMA/传 输，或者计算队列
    - 提交的工作通 过异步的方式执行
    - 多个command buffer可以被压送到独立、兼容的队列 里，从而实现并行的执行



