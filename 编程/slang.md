# [Slang Documentation](https://docs.shader-slang.org/en/latest/index.html)

## OVERVIEW

### 

## TUTORIALS

### [Write Your first Slang shader](https://docs.shader-slang.org/en/latest/first-slang-shader.html)

- 使用如下命令将.slang转成SPIR-V或者GLSL

    ```sh
    .\slangc.exe hello-world.slang -profile glsl_450 -target spirv -o hello-world.spv -entry computeMain
    .\slangc.exe hello-world.slang -profile glsl_450 -target glsl -o hello-world.glsl -entry computeMain
    ```

- Slang保证所有参数都有固定的binding，而不受着色器优化的影响

    - 有助于shader specializations等情况中确定已编译着色器内核的binding位置

- 强烈建议用户不要在Slang代码中使用明确的绑定限定符，而是让Slang编译器来设置layout参数。这是保持代码模块化并避免不同着色器模块之间潜在绑定位置冲突的最佳实践。

## [User Guide](https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/index.html)

### Conventional Language Features

#### Types

- 所有目标平台均支持32位的int和uint类型、32位float
- 在结构体中存储布尔类型时，务必对主机端数据结构进行相应填充，或将布尔值存储为 uint8_t 等类型，以确保与主机语言的布尔类型保持一致
- Slang 不支持长度超过 4 个元素的vector。如有需要，使用类似 float myArray[8] 的数组
- Slang 的 float3x4 表示一个三行四列的矩阵，而 GLSL 的 mat3x4 表示一个三列、四行的矩阵。在大多数情况下，这种差异无关紧要，因为在两种语言中，下标表达式 m[i] 都会返回一个 float4（vec4）。
  - 目前只需了解 Slang/HLSL/D3D 与 GLSL/OpenGL 在约定上存在差异即可。
- 在调用函数时，允许将定长数组作为参数传递给不定长数组参数
  - Array 类型拥有成员函数getCount()，返回数组的长度
  - 某个函数f对未指定大小的array参数调用了getCount()方，那么所有对函数f的调用都必须提供指定大小的数组参数，否则编译器将无法确定大小并报错。
  - 在 Slang 中，数组属于值类型，这意味着赋值、参数传递等操作在语义上会**复制**数组类型的值。
- struct的构造函数名字必须为`__init`
- enum类似于 C++ 中的`enum class`，默认始终具有作用域。
  - 希望某个enum类型不具有作用域，可以使用`[UnscopedEnum]`属性
  - 基础类型为 int，可以指定显式的底层整数类型，比如`uint16_t`
  - 使用`[Flags]`属性使默认值从1开始并以2的幂递增，适用于表示位标志的枚举
- opaque类型（不透明类型）用于访问通过 GPU API 分配的对象。
  - 包含Texture Types、Sampler、Buffers
  - **可能因为平台有如下限制**
    - 函数中不允许返回opaque类型
    - 全局变量和static变量不允许使用opaque类型
    - 不得出现在缓冲区的元素类型中，除非明确注明允许出现
- Formatted Buffers(也称为typed buffers或者buffer textures)，与一维纹理类似。语法为
  ```
  <<access>>Buffer<<arrayness>><<element type>>
  ```
  - access, array-ness和element type和texture类型一致，但是element type不会默认为float4、需要手动设置
  - 支持在加载时进行格式转换，比如`Buffer<float4>`可作为float4获取，但是内部可能以其他格式存储，例如 RGBA8
- Flat Buffers不支持格式转换，其要么是structured buffers要么是byte-addressed buffers
  - structured buffers（如`StructuredBuffer<T>`）包含一种显式元素类型 T，该类型将从缓冲区中加载和存储
  - byte-addressed buffers（如`ByteAddressBuffer`）不指定任何特定的元素类型，而是允许从缓冲区中的任意（适当对齐的）字节偏移量加载或存储值。
  - 二者均可以使用`<<access>>`区分只读（无前缀）和读写（RW）
- Constant Buffers（也称为uniform buffers）用于将不可变的参数数据从主机应用程序传递到 GPU 代码，语法为
  ```
  ConstantBuffer<T>
  ```
  - 仅包含其元素类型的 单个 值，而非一个或多个值
- TODO: Texture、Sampler、

#### Expressions

> **运算符的默认行为在未来的 Slang 版本中可能会发生变化**

- 与 HLSL/GLSL 类似，Slang 中无后缀的浮点字面量为float类型，而非double类型
- 与 HLSL 类似，&& 和 || 运算符目前不执行“短路”操作（即无条件计算所有操作数）
  - 如果?: 运算符的条件是标量，会执行短路操作；如果是vector，不执行短路操作，建议使用select
- 普通的一元和二元运算符用于向量和矩阵，按元素进行运算
  - 传统线性代数乘积使用`mul()`（在GLSL中使用\*运算符）。将mat3x4 与 vec3 相乘，操作数的顺序需要反转（以适应行/列约定），即`mul(v,m)`（GLSL中为`m * v`）
- Swizzles（分量重排操作）：
  - 例如`vector<float4> v`，则v.xy 是一个 float2 类型向量，还可以v.yx或者v.yyy
  - 与 GLSL 不同，Slang 仅支持 xyzw 和 rgba 作为重排元素
  - 与 HLSL 不同，Slang 目前不支持矩阵重排语法。
- Slang 中的 defer 语句与作用域绑定。该延迟语句会在作用域结束时执行，这一点与 Swift 类似，而非像 Go 那样仅在函数结束时执行。defer 支持但不要求使用块语句：`defer f(); `和 `defer { f(); g(); } `均为合法写法。
- Slang 不支持 C/C++ 中的 goto、throw关键字
- discard语句可在片段着色器的上下文中使用，用于终止当前片段的着色器执行，并使图形系统丢弃对应的片段。

#### Functions 

- 除了传统的 C 语法外，你还可以使用现代语法，通过func关键字来定义函数：
  ```
  float addSomeThings(int x, float y) { return x + y; }
  func addSomeThings(x : int, y : float) -> float { return x + y; }
  ```
- Slang 支持基于参数类型的函数重载。
- 用方向限定符进行标记输入/输出参数：in（默认，按值传递）、out、inout 或 in out

#### Preprocessor 

- Slang 支持 C 风格的预处理器
  ```
  #include
  #define, #undef
  #if, #ifdef, #ifndef, #else, #elif, #endif
  #error, #warning
  #line
  #pragma, #pragma once
  ```
  - 不建议在新代码中使用#include，因为此功能已被模块系统取代

#### Attributes 

- 属性是一种通用语法，用于为声明和语句添加额外的语义信息或元数据。属性由方括号（[]或者[[]]）包围，并置于其所应用的声明或语句之前。
  ```
  [unroll]  // 尽可能展开循环
  for(int i = 0; i < n; i++)
  { /* ... */ }
  ```

#### Global Variables and Shader Parameters

- ==Global Constants (全局常量)==：全局作用域的 **static const 变量**定义了一个编译时常量，供着色器代码使用
- ==Global-Scope Static Variables (全局作用域静态变量)==: 一个非const的全局作用域**static变量**，为每个线程分配独立的存储空间，而非真正意义上的全局变量。
  - 部分目标平台并非在所有用例下都支持static全局变量。对static全局变量的支持应被视为遗留功能，不建议进一步使用。
- ==Global Shader Parameters (全局着色器参数)==: 可以使用任意类型，包括opaque类型和non-opaque类型
  - Slang 编译器会对non-opaque类型的全局着色器参数发出警告，因为用户很可能以为自己在声明全局常量或传统的全局变量。可以标记为 uniform 来禁止此警告
    ```
    // WARNING: this declares a global shader parameter, not a global variable
    int gCounter = 0;
    // OK:
    uniform float scaleFactor;
    ```
- 全局作用域的 ==cbuffer==声明，在语义上等同于使用 ConstantBuffer 类型声明的着色器参数
  ```
  cbuffer PerFrameCB {
      float4x4 mvp;
      float4 skyColor;
      // ...
  }
  // 上下等价
  struct PerFrameData {
      float4x4 mvp;
      float4 skyColor;
      // ...
  }
  ConstantBuffer<PerFrameData> PerFrameCB;
  ```
- Explicit Binding Markup (显式绑定标记): 用于将不透明类型的着色器参数与特定 API 的绑定信息相关联。
  ```
  // Direct3D: 使用 register 语法
  Texture2D a : register(t0);
  Texture2D b : register(t1, space0);
  
  // Vulkan (and OpenGL)
  [[vk::binding(0)]]
  Texture2D a;
  [[vk::binding(1, 0)]]
  Texture2D b;
  ```
  - 单个参数可同时使用 D3D 风格和 Vulkan 风格的标记
  - 在 Slang 代码库中**几乎不需要使用这种标记**

#### Shader Entry Points

- 实例
  ```
  [shader("vertex")]
  float4 vertexMain(
      float3 modelPosition : POSITION,
      uint vertexID : SV_VertexID,
      uniform float4x4 mvp)
      : SV_Position
  { /* ... */ }
  ```
- Slang 允许在一个文件中存在多个入口点，可对应任意组合的着色器阶段，且入口点名称可以是任意合法标识符。
- [shader(...)] 特性用于将 Slang 中的某个函数标记为着色器入口点，同时还能指定其对应的管线阶段。
  - 支持省略 `[shader(...)]` 属性，但是必须使用 `IModule::findAndCheckEntryPoint()`，且必须指定着色阶段
  - 建议终使用 `[shader(...)]` 属性，以简化工作流程
- 输入参数分成varying or uniform
  - 【默认】varying inputs指在作为同一批次（一次绘制调用、一次计算调度等）调用的线程中可能发生变化的输入
  - uniform inputs指保证在批次中所有线程都相同的输入
- 输入参数必须声明binding semantic (绑定语义)
  - 通过在变量名后添加冒号（:）以及所选绑定语义的标识符来指定
- 输出参数也可以指定binding semantic，但是要放在输入参数的括号后面
- system-defined binding semantic以SV_开头
  - 给输入参数用的，表示从 GPU **接收**由所使用的pipeline和stage定义的特定数据
  - 给输出参数用的，表示GPU 应按照所使用的pipeline和stage定义的特定方式来**存储**在该输出中的值
    > GLSL 并未使用带有系统定义绑定语义的普通函数参数，而是采用了名称前缀为 gl_ 的特殊系统定义全局变量
- user-defined binding semantic
  - 给输入参数用的，表示从前一阶段接收具有匹配绑定语义的数据
  - 给输出参数用的，表示向后一阶段中具有匹配绑定语义的参数提供数据
- 将入口点的输入/输出与user-defined binding semantic进行匹配 有两种方式：根据不同的 API 以及同一 API 内的不同stage
  - 按索引匹配：将一个stage的user-defined输出与下一阶段的输入按声明顺序进行匹配。匹配的输出/输入参数类型必须完全相同或兼容（依据特定于 API 的规则）。
    - 部分 API 还要求匹配的输出/输入参数的binding semantic必须完全相同。
  - 按名称匹配：一个stage的user-defined输出与下一阶段的输入将根据其binding semantic进行匹配。匹配的输出/输入参数类型必须完全相同或兼容（依据特定于 API 的规则）。参数的声明顺序无需保持一致。
- entry-point uniform parameter在语义上与全局作用域的shader parameters相似，但不会污染全局作用域
- 语法就是加上uniform，例如uniform float4x4 mvp
  > GLSL 不支持入口点uniform参数；所有着色器参数都必须在全局作用域中声明；
  > 
  > HLSL 曾支持入口点uniform参数，但这一特性已被最新的编译器移除。

#### Mixed Shader Entry Points

- mixed entry points就是将多个入口点自由合并到同一个文件中，可以提升共享结构定义的类型安全性

>  GLSL 不支持多个入口点；但 SPIR-V 支持。
> 
> 希望在vulkan中利用 Slang 混合mixed entry points在编译器参数中加上`-fvk-use-entrypoint-name` 和 `-emit-spirv-directly`

- 在vulkan中，slang会将大部分的entry ponit中的uniform parameters映射到普通的pipeline layout，但是在ray tracing的entry point中会有不一样的映射逻辑

#### Auto-Generated Constructors

NOTE：以下自动生成的前提是与用户定义的构造函数不冲突

- 所有成员的可见性均相同（public、internal、private），则自动生成一个“member-wise constructor”（例如`__init(int in_a, int in_b, int in_c)`
- 如果可见性不同，会按顺序生成不同可见性的member-wise constructor，具体见[这里](https://shader-slang.org/slang/user-guide/conventional-features.html#auto-generated-constructors)
  ```
  struct GenerateCtorInner1 {
      internal int a = 0;
      // Slang will automatically generate an implicit
      // internal __init(int in_a) {
      //     a = 0;
      //     a = in_a;
      // }
  };
  struct GenerateCtor1 : GenerateCtorInner1 {
      internal int b = 0;
      public int c;
  
      // Slang will automatically generate an implicit
      // internal __init(int in_a, int in_b, int in_c) {
      //     b = 0;
      //     this = GenerateCtorInner1(in_a);
      //     b = in_b;
      //     c = in_c;
      // }
      // public __init(int in_c) {
      //     b = 0;
      //     this = GenerateCtorInner1();
      //     c = in_c;
      // }
  };
  ```

#### Initializer Lists 

- Flattened Array Initializer
  ```
  // Equivalent to `float3 a[2] = { {1,2,3}, {4,5,6} };`
  float3 a[3] = {1,2,3, 4,5,6}; 
  ```
- 在大多数情况下，使用初始化列表创建结构体类型的值，等同于调用该结构体的构造函数，并将初始化列表中的元素作为构造函数的参数
- 支持struct的** C 风格初始化列表**。
  - 结构体会被视为 C 风格结构体的条件如下
    - 用户从未定义过带有超过0个参数的自定义构造函数
    - struct中的所有成员变量具有相同的可见性
  - 部分初始化列表
    ```
    struct Foo {
        int a;
        int b;
        int c;
    };
    // Equivalent to `Foo val; val.a = 2; val.b = 3; val.c = 0;`
    Foo val = {2, 3};
    ```
  - 扁平化数组初始化
    ```
    struct Foo {
        int a;
        int b;
        int c;
    };
    // Equivalent to `Foo val[2] = { {0,1,2}, {3,4,5} };`
    Foo val[2] = {0,1,2, 3,4,5};
    ```
- 允许在默认构造函数中调用默认初始化器
  ```
  __init() {
      this = {}; //zero-initialize `this`
  }
  ```

### Basic Convenience Features

#### Type Inference in Variable Definitions

- 用于变量类型自动推导的`var`相当于C++中的`auto`，可以使用现代语法
  ```
  var a : int = 1; // OK.
  var b : int; // OK.
  ```

#### Immutable Values

- 定义不可变或常量值，可使用`let`关键字

#### Namespaces 

- 嵌套命名空间的简写语法
  ```
  namespace ns1.ns2 {
      int f();
  }
  // equivalent to:
  namespace ns1::ns2 {
      int f();
  }
  // equivalent to:
  namespace ns1 {
      namespace ns2 {
          int f();
      }
  }
  ```
- 引入命名空间使用关键字`using`，可以省略namespace

#### Member functions

- 支持静态成员函数
- 出于 GPU 性能考量，**成员函数中的 this 参数默认是不可变的**。尝试修改 this 会导致编译错误。
  - 想要成员函数能修改成员变量，请在成员函数上使用 [mutating] 特性，如下例所示
    ```
    struct Foo {
        int count;
    
        [mutating]
        void setCount(int x) { count = x; }
    
        // This would fail to compile.
        // void setCount2(int x) { count = x; }
    }
    ```

#### Properties 

- property可以快捷访问struct中public的变量，在getter和setter函数中定义访问方式。
  ```
  struct MyType {
      uint flag;
      property uint highBits {
      // property highBits : uint // 更加现代的语法
          get { return flag >> 16; }
          set { flag = (flag & 0xFF) + (newValue << 16); }
          set(uint x) { flag = (flag & 0xFF) + (x << 16);  }  // 也可以使用显式参数
      }
  };
  ```
- Slang 的property特性与 C# 和 Swift 类似

#### Initializers 

- 定义构造函数的语法未来可能会发生变化

#### Operator Overloading

- 支持重载的运算符：+、-、*、/、%、&、|、<、>、<=、>=、==、!=，以及一元运算符 -、~ 和 !
  - 不支持 && 和 || 运算符
- 可以将运算符 () 作为成员方法进行重载

#### Subscript Operator

- Slang 允许使用`__subscript`语法重写operator[]
  ```
  struct MyType {
      int val[12];
      __subscript(int x, int y) -> int {
          get { return val[x*3 + y]; }
          set { val[x*3+y] = newValue; }
      }
  }
  int test() {
      MyType rs;
      return rs[1, 0];
  }
  ```

#### Tuple Types

- 在 Slang 中，元组类型通过 `Tuple<...>` 语法定义，可通过构造函数或 `makeTuple` 函数来创建：
  ```
  Tuple<int, float, bool> t0 = Tuple<int, float, bool>(5, 2.0f, false);
  Tuple<int, float, bool> t1 = makeTuple(3, 1.0f, true);
  ```
- 元组元素通过`_0`、`_1`成员名访问
- 使用类似于向量和矩阵的**重排语法**来生成新元组
  ```
  t0._0_0_1 // evaluates to (5, 5, 2.0f)
  ```
- 拼接两个元组
  ```
  concat(t0, t1) // evaluates to (5, 2.0f, false, 3, 1.0f, true)
  ```
- 对元组进行比较：要求元组的所有元素类型都实现了 IComparable 接口，然后该元组本身也会实现 IComparable 接口
  ```
  let cmp = t0 < t1; // false
  ```
- 获取元组中的元素数量：对元组类型或元组值使用countof()。视为编译时常量
  ```
  int n = countof(Tuple<int, float>); // 2
  int n1 = countof(makeTuple(1,2,3)); // 3
  ```
- 所有元组类型都将转换为struct类型，并采用与struct类型相同的布局。

#### Optional<T> type

- `Optional<T> `类型来表示可能不存在的值，使用`none`表示任何`Optional<T>`的无值，通过`Optional<T>::value`获取值

#### Conditional<T, bool condition> Type

- `Conditional`类型可用于定义可被特化移除的结构体字段。若condition为false，编译器会从目标代码中移除该字段。
  ```
  interface IVertex {
      property float3 position{get;}
      property Optional<float3> normal{get;}
      property Optional<float3> color{get;}
  }
  
  struct Vertex<bool hasNormal, bool hasColor> : IVertex {
      private float3 m_position;
      private Conditional<float3, hasNormal> m_normal;
      private Conditional<float3, hasColor> m_color;
  
      __init(float3 position, float3 normal, float3 color) {
          m_position = position;
          m_normal = normal;
          m_color = color;
      }
  
      property float3 position {
          get { return m_position; }
      }
      property Optional<float3> normal {
          get { return m_normal; }
      }
      property Optional<float3> color {
          get { return m_color; }
      }
  }
  ```
- 使用场景：确保某一字段在着色器未使用时，不被定义在特化着色器变体中。

#### if_let syntax

- `if (let name = expr) `语法，用于在处理 `Optional<T>` 或 `Conditional<T, hasValue>` 值时简化代码
  ```
  Optional<int> getOptInt() { ... }
  
  void test() {
      if (let x = getOptInt()) {
          // if we are here, `getOptInt` returns a value `int`.
          // and `x` represents the `int` value.
      }
  }
  ```

#### reinterpret<T> operation

- `reinterpret` 可以将任何类型打包成任何其他类型，只要目标类型不小于源类型。

#### Pointers (limited)

- Slang 在为 SPIRV、C++ 和 CUDA 目标生成代码时支持指针
- 指针的语法与 C 语言类似，此外可以用`.`获取成员
- 指针类型也可通过泛型语法`Ptr<MyType, AccessMode=Access.ReadWrite, AddressSpace=AddressSpace.Device>`指定
  - 支持声明指向只读且不可变值的指针，以及指向Device以外地址空间的指针
  - **不可变值的指针**还可通过类型别名`ImmutablePtr<T, AddressSpace>`声明
- `Ptr<MyType>`与`MyType*`等效
- 指针的限制
  - Slang 支持指向全局内存和共享内存的指针，但**不支持local内存的指针**。定义指向本地变量的指针是无效的。
  - Slang 支持被定义为shader parameters的指针（例如作为constant buffer区字段）。
  - Slang **不支持opaque类型**（例如Texture2D）的指针，使用 `DescriptorHandle<T>`。
  - Slang **不支持自定义对齐规范**。对于使用已知对齐指针的加载和存储操作，可使用函数`loadAligned()`和`storeAligned()`。
  - Slang 目前**不支持 const 指针**

#### DescriptorHandle for Bindless Descriptor Access

- `DescriptorHandle<T> `类型用于表示资源的无绑定句柄。此特性为实现无绑定资源范式提供了一种可移植的方式
- 在 HLSL、GLSL 和 SPIRV 上，描述符类型（如textures、samplers、buffers）为opaque句柄，`DescriptorHandle<T> `会转换为 uint2，因此可在任意内存位置定义
  - 而在其他平台上，不是opaque句柄的`DescriptorHandle<T>`会映射为 T
- 声明方式
  ```
  struct DescriptorHandle<T> where T:IOpaqueDescriptor {}
  ```
  - `IOpaqueDescriptor` 是一个由所有资源类型实现的接口，包括textures、ConstantBuffer、RaytracingAccelerationStructure、SamplerState、SamplerComparisonState以及所有类型的StructuredBuffer
  - `DescriptorHandle<Texture2D>` 的简写形式为`Texture2D.Handle`
  - `DescriptorHandle<T>` 可以隐式转换为 T
- 转成 HLSL 时，`DescriptorHandle<T>` 会转换为对 `ResourceDescriptorHeap[index]` 和 `SamplerDescriptorHeap[index]` 的使用
- 转成 SPIRV 时，根据在编译请求中是否声明或请求了 spvDescriptorHeapEXT 功能，Slang 可为描述符句柄生成两种不同风格的代码
  - 【默认】不请求spvDescriptorHeapEXT，Slang 会引入一个全局描述符数组并从该全局数组中获取数据
    - 可通过 -bindless-space-index（代码中则为 `CompilerOptionName::BindlessSpaceIndex`）选项配置全局描述符数组的描述符集编号
  - 请求spvDescriptorHeapEXT，Slang 会将描述符句柄映射到 SPV_EXT_descriptor_heap 扩展，且不声明任何显式描述符集
    - 通过 -capability 命令行选项或编译 API
  - 若想避免上述行为，在代码中提供一个 `getDescriptorFromHandle` 函数，来自定义从描述符句柄到资源对象的转换逻辑
    - `getDescriptorFromHandle`不应直接从用户代码中调用，由编译器自动调用
- TODO

#### Extensions 

- Slang 允许在类型的初始定义之外为其定义额外的成员函数（不能扩展成员变量）
  ```
  struct MyType {
      int field;
      int get() { return field; }
  }
  extension MyType {
      float getNewField() { return newField; }
  }
  ```
- 更多细节见[这里](#extending-a-type-with-additional-interface-conformances)

#### Multi-level break

- Slang 允许**带标签的 break 语句**跳转到任何上层控制流的断点，而不仅仅是直接的父级
  ```
  outer:
  for (int i = 0; i < 5; i++) {
      inner:
      for (int j = 0; j < 10; j++)
          if (someCondition)
              break outer;
  }
  ```

#### Force inlining

- 大多数下游着色器编译器会内联所有函数调用。
- 使用 [ForceInline] 来指示 Slang 编译器执行内联操作：
  ```
  [ForceInline]
  int f(int x) { return x + 1; }
  ```

#### Error handling

- 函数必须使用 `throws` 声明该错误的类型
  ```
  enum MyError {
      Failure,
      CatastrophicFailure
  }
  
  int f() throws MyError {
      if (computerIsBroken())
          throw MyError.CatastrophicFailure;
      return 42;
  }
  ```
- 调用一个可能抛出异常的函数在前面加上 `try`
  - 如果只try没有catch，会向上层继续抛错误，所以调用`f()`的函数也必须声明它throws该错误类型
    ```
    void g() throws MyError {
        // This would not compile if `g()` wasn't declared to throw MyError as well.
        let result = try f();
        printf("Success: %d\n", result);
    }
    ```
- 捕获错误使用`do-catch`语句
  ```
  void g() {
      do {
          let result = try f();
          printf("Success: %d\n", result);
      } catch(err: MyError) {
          printf("Not good!\n");
      }
  }
  ```

#### Special Scoping Syntax

- `__ignored_block`里随便写什么，编译器都当看不见，而且支持嵌套 {}
  ```
  __ignored_block {
      arbitrary content in the source file,
      will be ignored by the compiler as if it is a comment.
      Can have nested {} here.
  }
  ```
- `__transparent_block` 里的代码，会被编译器直接当成 “写在它外层父作用域里” 一样处理，即有没有这个块完全一样，只是语法上多了一层包裹。

```
struct MyType{
    __transparent_block {
        int myFunc() { return 0; }
    }
}
// 等价于
struct MyType {
    int myFunc() { return 0; }
}
```

- `__file_decl`里的代码将被视为在全局作用域中定义，不同的`__file_decl `彼此不可见
  ```
  __file_decl {
      void f1() {}
  }
  __file_decl {
      void f2() {
          f1(); // error: f1 is not visible from here.
      }
  }
  ```

#### User Defined Attributes (Experimental)

- 用户可以定义自己的自定义属性类型，语法是`[UserDefinedAttribute(args...)]` ，具体见[这里](https://shader-slang.org/slang/user-guide/convenience-features.html#user-defined-attributes-experimental)
  ```
  [__AttributeUsage(_AttributeTargets.Var)]
  struct MaxValueAttribute {
      int value;
      string description;
  };
  
  [MaxValue(12, "the scale factor")]
  uniform int scaleFactor;
  ```

### Modules and Access Control

#### Defining a Module

- 一个module包含一个或多个文件
  - 必须有且仅有一个主文件（以 `module` 声明开头）
  - 额外的文件通过 `__include` 语法引入到模块中，被包含的文件须以`implementing <module-name>`声明开头
    - `__include` 前后的宏环境是完全隔离的，互不影响
    - 重复写同一文件的 `__include`只会包含一次，所以支持循环`__include`
    - `__include`顺序无关，可以访问所有的变量、函数等内容
  ```
  // 文件a.slang
  implementing m;
  void f_a() {}
  
  // 文件b.slang
  implementing "m"; // alternate syntax.
  __include a; // pulls in `a` to module `m`.
  void f_b() { f_a(); }
  
  // 文件c.slang
  implementing "m.slang"; // alternate syntax.
  
  void f_c() {
      // OK, `c.slang` is part of module `m` because it is `__include`'d by `m.slang`.
      f_a(); f_b();
  }
  
  // 文件m.slang
  module m;
  __include m; // OK, a file including itself is allowed and has no effect.
  __include "b"; // Pulls in file b (alternate syntax), and transitively pulls in file a.
  __include "c.slang"; // Pulls in file c, specifying the full file name.
  void test() { f_a(); f_b(); f_c(); }
  ```
- module、implementing和__include均支持两种语法形式来引用模块或文件：普通标识符标记、字符串字面量
  ```
  __include dir.file_name; // `file_name` is translated to "file-name".
  __include "dir/file-name.slang";
  __include "dir/file-name";
  ```
- 如果存在带有 implementing 声明的悬空文件，未被模块中的任何其他文件通过 __include 指令包含。此类悬空文件不会被视为模块的组成部分，也不会被编译
- Slang 会将任何下划线（_）转换为连字符（“-”）以获取文件名

#### Importing a Module

- 使用 import 关键字导入模块，也支持标识符标记和文件名字符串两种语法
- 可以多次导入，只会加载一次，无需使用`#pragma once`

#### Access Control

- Slang 支持访问控制修饰符：public、internal 和 private
- internal符号在整个同一模块中均可见，无论它是从同一类型还是同一文件中引用。但其他模块无法访问
  ```
  // 文件a.slang
  module a;
  __include b;
  public struct PS {
      internal int internalMember;
      public int publicMember;
  }
  internal void f() { f_b(); } // OK, f_b defined in the same module.
  
  // 文件b.slang
  implementing a;
  internal void f_b(); // Defines f_b in module `a`.
  public void publicFunc();
  
  // 文件m.slang
  module m;
  import a;
  void main() {
      f(); // Error, f is not visible here.
      publicFunc(); // OK.
      PS p; // OK.
      p.internalMember = 1; // Error, internalMember is not visible.
      p.publicMember = 1; // OK.
  }
  ```
- 默认可见性为`internal`，除了interface 的成员的默认可见性=接口的可见性
- 类型的定义不能是private，即`private struct S {}`错误
- interface不能是private

#### Organizing File Structure of Modules

- 没有强制的文件组织方式，但是建议如下
  - 顶层目录包含用户代码会import的模块。
  - 模块的实现细节放在目录树较低层级的文件中。

#### Legacy Modules

- 旧版本的slang不支持访问控制，如果满足以下所有条件时，会被现有的编译器认定为旧版本，把所有的内容都视为public
  - 开头缺少module声明
  - 没有使用__include
  - 没有使用任何public, private or internal

### Capabilities 

- Slang 的类型系统可以推断并强制要求capabilities constraints（功能约束），以确保着色器代码在针对特定平台编译前，能与该平台集合兼容。

#### Capability Atoms and Capability Requirements

- 需查看 Slang 编译器支持的所有能力列表，请查阅 [capability definition file](https://github.com/shader-slang/slang/blob/master/source/slang/slang-capabilities.capdef)。
- 声明单个或者多个原子能力的语法如下。多个原子能力通过disjunction（析取，即“逻辑或”）合并
  ```
  [require(spvShaderClockKHR)]
  [require(glsl, GL_EXT_shader_realtime_clock)]
  [require(hlsl_nvapi)]
  uint2 getClock() {...}
  ```
- 一个功能可以隐含其他功能
  - 比如spvShaderClockKHR隐含SPV_KHR_shader_clock（代表 SPIRV的SPV_KHR_shader_clock扩展），而SPV_KHR_shader_clock又隐含spirv_1_0（代表SPIRV 代码生成目标）
  - 上述[require]构成的getClock 的最终capability requirements（能力要求）为
    ```
    spirv_1_0 + SPV_KHR_shader_clock + spvShaderClockKHR | glsl + _GL_EXT_shader_realtime_clock | hlsl + hlsl_nvapi
    ```

#### Conflicting Capabilities

- 如果两个capability requirements包含相互冲突的不同原子，则这两个需求被视为不兼容

#### Capabilities Between Parent and Members

- 成员的capability requirement始终与其父级中声明的capability requirements合并
  ```
  [require(glsl)]
  [require(hlsl)]
  struct MyType
  {
      [require(hlsl, hlsl_nvapi)]
      [require(spirv)]
      static void method() { ... }
  }
  ```
  - MyType.method 要求满足 glsl | hlsl + hlsl_nvapi | spirv
- [require] 属性也可用于模块声明，这样该要求将适用于模块内的所有成员

#### Capabilities Between Subtype and Supertype

- [TODO](https://shader-slang.org/slang/user-guide/capabilities.html#capabilities-between-subtype-and-supertype)

#### Capabilities Between Requirement and Implementation

- [TODO](https://shader-slang.org/slang/user-guide/capabilities.html#capabilities-between-requirement-and-implementation)

#### Capabilities of Functions

- 只要函数是 internal 或 private，Slang 会根据其定义推断函数的能力要求
  - 比如函数中用了discard语句，就需要fragment能力
- __target_switch 语句会在其推断的能力要求中引入disjunction
  ```
  void myFunc(){
      __target_switch {
      case spirv: ...;
      case hlsl: ...;
      }
  }
  ```
  - myFunc 的能力要求为(spirv | hlsl)，这意味着该函数可从具备spirv或hlsl能力的上下文中调用
- 函数声明必须是函数体所使用功能的超集，且需涵盖函数声明隐式/显式要求的任何着色器阶段/目标。
  ```
  [require(sm_5_0)]
  public void requires_sm_5_0() {} 
  
  [require(sm_4_0)]
  public void logic_sm_5_0_error() { // Error, missing `sm_5_0` support
      requires_sm_5_0();
  }
  
  public void logic_sm_5_0__pass() { // Pass, no requirements
      requires_sm_5_0();
  }
  
  [require(hlsl, vertex)]
  public void logic_vertex() {}
  
  [require(hlsl, fragment)]
  public void logic_fragment() {}
  
  [require(hlsl, vertex, fragment)]
  public void logic_stage_pass_1() { // Pass, `vertex` and `fragment` supported
      __stage_switch {
          case vertex: logic_vertex();
          case fragment: logic_fragment();
      }
  }
  
  [require(hlsl, vertex, fragment, mesh, hull, domain)]
  public void logic_many_stages() {}
  
  [require(hlsl, vertex, fragment)]
  public void logic_stage_pass_2() { // Pass, function only requires that the body implements the stages `vertex` & `fragment`, the rest are irelevant
      logic_many_stages();
  }
  
  [require(hlsl, any_hit)]
  public void logic_stage_fail_1() { // Error, function requires `any_hit`, body does not support `any_hit`
      logic_many_stages();
  }
  ```

#### Capability Aliases

- 为了方便，Slang 定义了许多可在 [require] 属性中使用的别名
  ```
  // 在slang-capabilities.capdef中有如下别名
  alias sm_6_6 = _sm_6_6
               | glsl_spirv_1_5 + sm_6_5
                  + GL_EXT_shader_atomic_int64 + atomicfloat2
               | spirv_1_5 + sm_6_5
                  + GL_EXT_shader_atomic_int64 + atomicfloat2
                  + SPV_EXT_descriptor_indexing
               | cuda
               | cpp;
  // 注意，GL_EXT_shader_atomic_int64 也是一个别名
  alias GL_EXT_shader_atomic_int64 = _GL_EXT_shader_atomic_int64 | spvInt64Atomics;
  
  // 那么用户就可以这么用
  [require(sm_6_6)]
  void MyFunc() {}
  ```
- 当在 [require] 属性中使用别名时，编译器会展开该别名以计算功能集，并移除所有不兼容的合取项

#### Validation of Capability Requirements

- Slang 要求所有public的函数与接口均需显式声明所需的功能权限，但不能使用超出声明范围的能力。
  - 没有声明，表示无需任何特定权限
- entry point推荐显式声明，不写就自动推断，不兼容时报错

### Interfaces and Generics

#### Interfaces 

- interfaces用于定义一个类型应提供的methods和services，可以有默认实现
- struct类型可以声明遵循多个interface，对于有默认实现的进行重写实现必须显式标记为 `override`
  ```
  interface IFoo { int myMethod(float arg); }
  interface IBar { uint myMethod2(uint2 x) { return x + 1; }; } // 默认实现
  struct MyType : IFoo, IBar {
      int myMethod(float arg) {...}
      override uint myMethod2(uint2 x) {...}  // 必须显式
  }
  ```

#### Generics 

- 定义泛型：通常是指代 generic type parameters（泛型类型参数）
  - `where T ：IFoo`是type conformance constraints（类型一致性约束），编译期提前检查。而c++的模板不提前检查，而是等实例化的时候才检查
    ```
    // 方法一
    int myGenericMethod<T>(T arg) where T : IFoo {
        return arg.myMethod(1.0);
    }
    // 方法二
    __generic<typename T> // `typename` is optional.
    int myGenericMethod(T arg) where T : IFoo {
        return arg.myMethod(1.0);
    }
    // 方法三：更加化简，无需包含where
    int myGenericMethod<T:IFoo>(T arg) { ... }
    ```
  - 类型一致性约束**可选**，使用`where option`
    ```
    int myGenericMethod<T>(T arg) where optional T: IFoo {
        if (T is IFoo) arg.myMethod(1.0); // OK in a block that checks for T: IFoo conformance.
    }
    // 等价于下面两个
    int myGenericMethod<T>(T arg) {}
    int myGenericMethod<T>(T arg) where T: IFoo { arg.myMethod(1.0); }
    ```
  - 支持多个 `where` 子句，且单个 `where` 子句中可包含多种接口类型
    ```
    struct MyType<T, U>
        where T: IFoo, IBar
        where U: IBaz<T> {}
    // equivalent to:
    struct MyType<T, U>
        where T: IFoo
        where T: IBar
        where U: IBaz<T> {}
    ```
- 调用泛型
  ```
  MyType obj;
  int a = myGenericMethod<MyType>(obj); // OK, explicit type argument
  int b = myGenericMethod(obj); // OK, automatic type deduction
  ```
- 支持generic value parameters（泛型值参数），通过`let`关键字声明。目前，泛型值参数的类型允许为int、bool和enume
  ```
  // 方法一
  void g1<let n : int>() { ... }
  
  enum MyEnum { A, B, C }
  void g2<let e : MyEnum>() { ... }
  
  //  方法二：C 语言风格
  void g1<int n>() { ... }
  ```
  - ✰Slang 支持在方括号属性中引用 泛型值参数，可以在编译时控制工作组大小等属性
  ```
  [numthreads(blockSize, blockSize, 1)]
  void computeMain<int blockSize>() { ... }
  ```

#### Supported Constructs in Interface Definitions

- property关键词支持在interface中添加变量，而且需要具备指定方法（不需要加圆括号）
  ```
  interface IFoo {
      property int count {get; set;}
  }
  
  struct MyObject : IFoo {
      int myCount = 0;
      property int count {
          get { return myCount; }
          set { myCount = newValue; } 
      }
  }
  ```
- 可以在Interfaces中添加Generic Methods、Static Methods、Static Constants
- 在Interfaces中使用特殊关键字 This 来表示 符合该interface的类型本身
  ```
  interface IComparable { int comparesTo(This other); }
  struct MyObject : IComparable {
      int val;
      int comparesTo(MyObject other) {  // 在 MyObject 的作用域内，This 类型等同于 MyObject
          return val < other.val ? -1 : 1;
      }
  }
  ```
- 想要在generic method创建generic type的实例，有两种方法
  - 方法一：在interface中引入static方法，返回为This
    ```
    interface IFoo { static This create(int a, int b); }
    void f<T:IFoo>() {
        T obj = T.create(1, 2);
    }
    ```
  - 方法二：在interface定义构造函数
    ```
    interface IFoo { __init(int a, int b); }
    void g<T:IFoo>() {
        T obj = {1, 2}; // OK, invoking the initializer on T.
    }
    ```

#### Associated Types

- 在interface中可以定义关联类型，关键字为`associatedtype`，从而使用依赖实现接口时才定义的具体类型
  ```
  // The interface for an iterator type.
  interface IIterator {
      // An iterator needs to know how to move to the next element.
      This next();
  }
  
  interface IFloatContainer {
      // Requires an implementation to define a typed named `Iterator` that conforms to the `IIterator` interface.
      associatedtype Iterator : IIterator;
  
      // Returns the number of elements in this container.
      uint getCount();
      // Returns an iterator representing the start of the container.
      Iterator begin(); // 这里的返回根据不同的实现会有变化，所以要用associatedtype
      // Returns an iterator representing the end of the container.
      Iterator end();
      // Return the element at the location represented by `iter`.
      float getElementAt(Iterator iter);
  };
  
  ```
  - 等到具体实现 IFloatContainer 接口的时候，必须在其作用域内定义一个名为 Iterator 的类型，使用关键字`typedef`
    ```
    struct ArrayIterator : IIterator {  // 可以写在ArrayFloatContainer的内部
        uint index;
        __init(int x) { index = x; }
        ArrayIterator next() { return ArrayIterator(index + 1); }
    }
    struct ArrayFloatContainer : IFloatContainer {
        float content[10];
    
        // Specify that the associated `Iterator` type is `ArrayIterator`.
        typedef ArrayIterator Iterator;
    
        Iterator getCount() { return 10; }
        Iterator begin() { return ArrayIterator(0); }
        Iterator end() { return ArrayIterator(10); }
        float getElementAt(Iterator iter) { return content[iter.index]; }
    }
    ```
  - 然后就可以直接使用 IFloatContainer 中值的泛型函数无需关心 Iterator 具体类型的实现细节
    ```
    float sum<T:IFloatContainer>(T container) {
        float result = 0.0f;
        for (T.Iterator iter = container.begin(); iter != container.end(); iter=iter.next()) {
            float val = container.getElementAt(iter);
            result += val;
        }
        return result;
    }
    ```

#### Generic Value Parameters

- 唯一的泛型值参数类型为 int、uint 和 bool。float 及其他类型不能用于泛型值参数。
- 只要类型表达式能在编译时求值，就支持其中的计算。例如，`vector<float, 1+1>`是允许的，与 `vector<float, 2>` 等效。

#### Type Equality Constraints

- type equality constraints（类型相等性约束）用于为关联类型指定额外约束
  ```
  interface IFoo { associatedtype A; }
  
  // Access all T that conforms to IFoo, and T.A is `int`.
  void foo<T>(T v)
      where T : IFoo
      where T.A == int  {}  // 类型相等性约束
  
  struct X : IFoo { typealias A = int; }
  
  struct Y : IFoo { typealias A = float; }
  
  void test() {
      foo<X>(X()); // OK
      foo<Y>(Y()); // Error, `Y` cannot be used for `T`.
  }
  ```

#### Interface-typed Values

- 为了避免大量使用泛型导致代码冗余，slang支持直接**将接口类型用作形参类型、变量or返回值**
  ```
  interface ITransform {
      int compute(MyObject obj);
  }
  // 将接口类型用作参数类型
  ITransform test(ITransform arg) {
      ITransform v = arg;
      return v;
  }
  // 等价于使用泛型
  TTransform test<TTransform : ITransform>(TTransform arg, arg) {
      TTransform v = arg;
      return v;
  }
  ```
- 注意：如果在变量or返回值中使用接口类型，编译期不确定接口值的具体类型时，Slang 会走dynamic dispatch code（动态调度代码，或者叫运行时动态分发），影响性能，out 返回也一样。
  ```
  ITransform getTransform(int x) {
      if (x == 0) {
          Type1Transform rs = {};
          return rs;
      } else {
          Type2Transform rs = {};
          return rs;
      }
  }
  ```
  - 要求接口的具体类型不能包含含任何不透明类型的成员（比如buffer），否则编译报错
  - 如果某个变量的类型为接口ITransform，当它指定为某个具体的类型Type1Transform后，就不能再指定为另一个具体的l类型Type2Transform
    ```
    ITransform t = Type1Transform();
    // Assign a different type of transform to `t`: (Not supported by Slang today)
    t = Type2Transform();
    ```

#### Extending a Type with Additional Interface Conformances

- extensions可用于让现有类型遵循额外的interface
  ```
  interface IFoo { int foo(); };
  struct MyObject : IFoo { int foo() { return 0; } }
  // 引入更多接口
  interface IBar { float bar(); }
  interface IBar2 { float bar2(); }
  extension MyObject : IBar, IBar2 { 
    float bar() { return 1.0f }
    float bar2() { return 2.0f }
  }
  ```

#### is and as Operator

- 使用`is`运算符来测试 接口类型/泛型类型 的值是否为特定的具体类型
- 使用`as`运算符将 接口类型/泛型类型 的值向下转换为特定类型（实际为`Optional<T>`类型）
  ```
  interface IFoo {}
  struct MyImpl : IFoo {}
  void test(IFoo foo) {
      bool t = foo is MyImpl; // true
      // 放在if外面的话则是这样
      // Optional<MyImpl> optV = foo as MyImpl;
      // if (t == (optV != none))
      if (let t = foo as MyImpl)  
          printf("success");  // 输出这个
      else
          printf("fail");
  }
  void main() {
      MyImpl v;
      test(v);
  }
  ```

#### Generic Interfaces

- 接口可以使用泛型类型 && 泛型类型一致性约束
  ```
  void traverse<TElement, TCollection>(TCollection c) where TCollection : IEnumerable<TElement> { ... }
  ```

#### Generic Extensions

- 可以使用 ==泛型扩展== 来扩展 泛型类型
  ```
  interface IFoo { void foo(); }
  struct MyType<T : IFoo> {   // 泛型类型
      void foo() { ... }
  }
  
  interface IBar { void bar(); }
  // Extend `MyType<T>` so it conforms to `IBar`.
  extension<T:IFoo> MyType<T> : IBar { // 使用 泛型扩展
      void bar() { ... }
  }
  // 等价于
  __generic<T:IFoo>
  extension MyType<T> : IBar {
      void bar() { ... }
  }
  ```

#### Extensions to Interfaces 

- 除了扩展普通类型之外，你还可以对符合特定接口的所有类型定义扩展：
  ```
  // An example interface.
  interface IFoo { int foo(); }
  
  // Extend any type `T` that conforms to `IFoo` with a `bar` method.
  extension<T:IFoo> T { int bar() { return 0; } } // 以符合特定interface的所有类型定义扩展
  
  int use(IFoo foo) {
      // With the extension, all uses of `IFoo` typed values
      // can assume there is a `bar` method.
      return foo.bar();
  }
  ```
  - interface类型本身无法被扩展，因为给interface加了新要求 会使所有符合该接口的现有类型都无效
- 在存在扩展的情况下，某个类型可能有多种遵循接口的方式。在这种情况下，Slang 始终会优先选择更具体的遵循方式，而非泛化的遵循方式
  ```
  interface IBase{}
  interface IFoo { int foo(); }
  
  // MyObject directly implements IBase:
  struct MyObject : IBase, IFoo { int foo() { return 0; } } // 更具体，选他
  
  // Generic extension that applies to all types that conforms to `IBase`:
  extension<T:IBase> T : IFoo { int foo() { return 1; } }   // 太泛化了
  
  int helper<T:IFoo>(T obj) { return obj.foo(); }
  
  int test() {
      MyObject obj;
      return helper(obj); // 返回0
  }
  ```

#### Variadic Generics

- Slang 支持variadic generic type parameters（可变泛型类型参数）、 variadic value parameters（可变值参数
- 使用`each T`定义generic type pack parameter（泛型类型包参数），该参数可以是零个或多个类型的列表
  ```
  struct MyType<each T> {}
  MyType // OK
  MyType<int> // OK
  MyType<int, float, void> // OK
  ```
  - 常见用途是定义printf
    ```
    void printf<each T>(String message, expand each T args) { ... }
    ```
  - `expand each T` 用于把类型包 T（一堆类型的集合）逐个展开，相当于 “遍历 + 展开”。比如 `T = int, float, bool`，`expand each T `就变成：`int, float, bool`
    - **`expand` 表达式可被视为类型包的映射操作**
  - `expand S<each T> `是 “批量套泛型”，对包里每个类型都套一遍泛型 S<...>，即`expand S<each T>` → `S<int>, S<float>, S<bool>`
  - `expand `不仅能展开类型包，还能对元组 / 参数包的值做批量操作：
    ```
    void printNumbers<each T>(expand each T args) where T == int {
        // An single expression statement whose type will be `(void, void, ...)`.
        // where each `void` is the result of evaluating expression `printf(...)` with each corresponding element in `args` passed as print operand.
        expand printf("%d\n", each args);
        // The above statement is equivalent to:
        // (printf("%d\n", args[0]), printf("%d\n", args[1]), ..., printf("%d\n", args[n-1]));
    }
    void compute<each T>(expand each T args) where T == int {
        // Maps every element in `args` to `elementValue + 1`, and forwards the new values as arguments to `printNumbers`.
        printNumbers(expand (each args) + 1);
        // The above statement is equivalent to:
        // printNumbers(args[0] + 1, args[1] + 1, ..., args[n-1] + 1);
    }
    void test() {
        compute(1,2,3);
        // Prints:
        // 2
        // 3
        // 4
    }
    ```
- 可变值参数声明了一组编译时常量整数值
  - expand 和 each 关键字在 值包 中的使用方式与在 类型包 中相同
  ```
  struct Dims<let each D : int> {}  // 等价于
  Dims<>          // empty pack
  Dims<4>         // single element
  Dims<2, 3, 4>   // three elements
  ```
- 可变 类型包参数 && 值包参数 必须出现在参数列表的末尾
  - 包含多个类型包参数，那么每个类型包在实例化位置必须包含相同数量的参数。
- 使用`countof()`获取类型包或值包中的元素数量
- 内置可变参数包运算符
  - __first(P) 返回类型包、值包或类元组包源中的第一个元素。
  - __last(P) 返回最后一个元素
  - __trimHead(P) 返回 移除了第一个元素的包。
  - __trimTail(P) 返回 移除了最后一个元素的包。
    > __first(...) 和 __last(...) 仅对已知非空的包有效。对于泛型包，请使用 where nonempty(P) 约束来明确保证这一点。

#### Builtin Interfaces

- slang支持以下内置接口：
  - `IComparable`：比较遵循该协议类型的两个值的方法。所有基本数据类型、向量类型和矩阵类型均支持此接口。
  - `IRangedValue`：检索该类型范围所表示的最小值和最大值的方法。所有整数和浮点标量类型均支持此接口。
  - `IArithmetic`：+、-、*、/、%以及取反运算的方法。同时还提供了从int进行显式转换的方法。所有内置整数、浮点标量、向量和矩阵类型均实现了该接口。
  - `ILogical `：提供了所有位运算以及逻辑and、or、not运算的方法。同时还提供了从int进行显式转换的方法。所有内置整数标量、向量和矩阵类型均实现了该接口。
  - `IInteger`：表示一种同时支持 IArithmetic 和 ILogical 运算的逻辑整数。所有内置整数标量类型均实现了此接口。
  - `IDifferentiable`：表示可微的值。
  - `IFloat`：表示一种支持 IArithmetic、ILogical 和 IDifferentiable 运算的逻辑浮点类型。同时提供了与 float 相互转换的方法。所有内置浮点标量、向量和矩阵类型均实现了此接口。
  - `IArray<T>` ：表示一种支持从索引中检索 T 类型元素的逻辑数组。由数组类型、向量、矩阵和 StructuredBuffer 实现。
  - `IRWArray<T>`： 表示元素可变的逻辑数组。由数组类型、向量、矩阵、RWStructuredBuffer 和 RasterizerOrderedStructuredBuffer 实现。
  - `IFunc<TResult, TParams...> `：表示一个可调用对象（带有 operator()），它返回 TResult 并以 TParams... 作为参数。
  - `IMutatingFunc<TResult, TParams...>`： 与 IFunc 类似，但 operator() 方法为 [mutating]。
  - `IDifferentiableFunc<TResult, TParams...> `：与 IFunc 类似，但 operator() 方法带有 [Differentiable] 特性。
  - `IDifferentiableMutatingFunc<TResult, TParams...>`： 与 IFunc, 类似，但 operator() 方法带有 [Differentiable] 和 [mutating] 特性。
  - `__EnumType`：由所有枚举类型实现。
  - `__BuiltinIntegerType`：由所有整数标量类型实现。
  - `__BuiltinFloatingPointType`：由所有浮点标量类型实现。
  - `__BuiltinArithmeticType`：由所有整数标量类型和浮点标量类型实现。
  - `__BuiltinLogicalType`：由所有整数类型和 bool 类型实现。

### Automatic Differentiation

[TODO](https://shader-slang.org/slang/user-guide/autodiff.html)

### Compiling Code with Slang

#### Concepts 

- 在最细的粒度下，代码（硬盘上的文件or内存中的字符串）会以==source units（源单元）==的形式被传递给编译器
  - 在同一次编译中指定了多个源单元，它们将被独立预处理和解析
- source units被分组成==translation units（翻译单元）==，每个translation unit在编译时都会生成一个单独的module
  - 一个translation unit可以只包含一个single source unit
- 编译后的module，属于Slang内部的==intermediate representation (中间表示，IR)==，可序列化为.slang-module二进制文件
  - 该二进制文件可通过ISession::loadModuleFromIRBlob函数加载
  - 在 Slang 源码里，导入该二进制文件 和 导入.slang源码文件模块 的语法一样
- 一个translation unit / module可以包含零个或多个entry point（入口点）。编译时识别入口点的方式有两种
  - 【推荐】 带有[shader(...)] 属性的函数声明
  - 使用着色器源代码外部的配置选项显式指定入口点函数，编译器会忽略所有[shader(...)]属性
- 一个translation unit / module可以包含零个或多个全局着色器参数
- 每个入口点可定义零个或多个入口点uniform着色器参数
- 在 Slang 系统中，target代表可为之生成输出代码的特定平台和功能集。一个target包含如下信息：
  - 代码应生成的格式：SPIR-V、DXIL 等
  - 指定通用特性/功能级别的配置文件：例如 D3D 着色器模型 5.1、GLSL 4.60 版本等。
  - 应假定target具备以下可选capabilities：例如特定的 Vulkan GLSL 扩展
  - 影响代码生成的选项：浮点严格性、要生成的调试信息级别等。
- Slang 支持在同一次编译会话中为多个target进行编译
  - **编译器前端**包含预处理、语法分析和语义分析，对每个translation uint运行一次，其结果会在所有target之间共享。
    - 因此Slang不自带target相关宏（比如`#define SPIRV 1`），需要的话 一次只编译一个目标 && 手动为每个目标设置专属的预处理器宏
  - **编译器后端**生成输出代码，因此每个目标运行一次。
- 为shader parameter计算的layout取决于以下因素
  - 哪些模块和入口点被一起使用；这些定义了哪些parameter是相关的
  - 对parameter进行明确的排序
  - target对layout施加的一些规则和约束
    > Slang 中的一个重要设计选择是让编译器的用户能够控制这些选择。
- ==composition（组合）==用来指定一起使用的module和entry point，及其相对顺序
  - component type是着色器代码组合的一个单元，包含modules和entry points
  - composite component type则是各种component type的组合，比如一个module和两个entry points
    - 一旦composite component type创建成功后，可以查询其中着色器参数的layout、也可以调用链接步骤来解析所有跨模块引用
- ==link==用于解析IR（中间表示）中的所有交叉引用，并生成一个全新的独立中间表示模块，该模块包含目标代码生成所需的全部内容
- ==kernel==是在link之后为某个entry point生成的。
  - 根据不同的target、以及不同的shader code的组合产生的不同layout，都会生成不同的kernel

#### Command-Line Compilation with slangc

- slangc是一款命令行编译器，命令行参数看[这里](https://github.com/shader-slang/slang/blob/master/docs/command-line-slangc-reference.md)
  ```sh
  # entry point为computeMain()
  slangc hello-world.slang -target spirv -o hello-world.spv
  slangc hello-world.slang -target hlsl -entry computeMain -stage compute -o hello-world.hlsl
  ```
  - 没有`[shader(...)]`属性的代码中，`-entry` 选项后应紧跟 `-stage` 选项，以指定入口点的阶段
    - 对于 HLSL 等目标，即使`有 [shader(...)]` 属性，也要指定 `-entry` 和 `-stage` 选项
  - `-profile`选项指定要使用的配置文件。比如GLSL中可以使用`glsl_430`和`glsl_460`
  - slangc指定多个文件、entry point或者target的时候，命令行参数的顺序很重要
    - 当一个选项修改或关联另一个命令行参数时，它会隐式应用于最近的相关参数
      - 有多个文件 / entry point / target，则`-entry` / `-stage` / `-profile` 适用于其前面的文件 / entry point / target
    - kernel的参数`-o`使用于其前面的entry point，同时根据文件拓展名自动选择合适的target
  - `-D<name>` 或 `-D<name>=<value>`：定义预处理器宏
  - `-I<path>` 或 `-I <path>`：引入解析 #include 指令和 import 声明时所使用的 搜索路径。
  - `-g`可用于在输出文件中启用调试信息的包含（在可行且已实现的情况下）
  - `-O<level>` 可用于在 Slang 编译器调用下游代码生成器时控制优化级别。
    > 省略-o参数，内核代码将被写入标准输出
- slang可以借助“dxc”、“fxc”、“glslang”或“gcc”等“下游”工具编译target，使用参数`-X`
  - Slang 可用的“下游”阶段在 API 中被称为`SlangPassThrough`类型，包含各种编译器（fxc/dxc/glslang/visualstudio/clang/gcc/genericcpp/nvrtc）、链接器（linker）
    ```sh
    # 将多个选项传递给 DXC，使用参数
    -Xdxc -Gfa -Xdxc -
    # 如果参数较多，可以使用省略号 以及终止符-X.
    -Xdxc... -Gfa -Vd -X.
    # -X... 选项可以嵌套
    -Xgcc -Xlinker --split -X.  # GCC看到的是-Xlinker --split，而链接器看到的是--split
    ```
  - `-X`最适合用于那些无法通过 Slang 常规机制获取的选项。
- 可以将 .slang 文件编译为二进制中间表示（IR）模块，后续使用import调用
  ```sh
  slangc my_library.slang -o my_library.slang-module
  ```

#### Using the Compilation API

- Slang C++ API大部分遵循==COM（组件对象模型，the Component Object Model）==的模式
  - 不依赖COM的任何runtime特性
  - `ISlangUnknown` 接口与标准 COM`IUnknown` 等效（且二进制兼容）
  - `Slang::ComPtr<T>`“智能指针”类型
  - `SlangResult` 类型与标准 COM `HRESULT` 类型等效（且二进制兼容）。
  - Slang API 调用在成功时返回零值（`SLANG_OK`），在出错时返回负值
- 以“_Experimental”后缀的Slang API 接口尚未完善，可能存在已知漏洞，且可能变更或被移除
- Slang的==global session（全局会话）==为`slang::IGlobalSession`，表示应用程序与 Slang API 特定实现之间的连接，通过函数`slang::createGlobalSession()`创建
  ```c_cpp
  using namespace slang;
  
  Slang::ComPtr<IGlobalSession> globalSession;
  SlangGlobalSessionDesc desc = {};
  createGlobalSession(&desc, globalSession.writeRef());
  ```
  - 创建global seesion的时候，Slang 系统会加载编译器提供给用户代码的core module的内部表示形式，时间可能会很久
  - 如果想要开启GLSL兼容，需要在调用`createGlobalSession()`的时候，设置`SlangGlobalSessionDesc::enableGLSL`为true
  - global session暂时不具备 线程安全。多线程编译需要确保每个并发线程使用不同的global session
- ==session （会话）==使用`slang::ISeesion`接口，表示一组具有一致编译器选项的编译作用域（共享编译目标列表其选项、用于 #include 和 import的搜索路径、预定义宏），使用函数`IGlobalSession::createSession()`创建
  ```c_cpp
  SessionDesc sessionDesc;
  /* ... fill in `sessionDesc` ... */
  Slang::ComPtr<ISession> session;
  globalSession->createSession(sessionDesc, session.writeRef());
  ```
  - 在`SessionDesc`中指定常用编译器选项，比如`searchPath` 和 `preprocessMacros`（类型为`PreprocessorMacroDesc`）。其他编译器选项可通过 `compilerOptionEntries`指定（`CompilerOptionEntry`类型的数组）。详细编译器选项见[这里](https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/08-compiling.html#compiler-options)
    - session中预定义的宏在每个被编译的源单元开头均可见，包括通过import加载的源单元
    ```
    const char* searchPaths[] = { "myapp/shaders/" };
    sessionDesc.searchPaths = searchPaths;
    sessionDesc.searchPathCount = 1;
    ```
  - `SessionDesc::targets` 数组 用于描述应用程序希望在会话中支持的target列表。类型为`TargetDesc`，其中最重要的是`format`（为`SlangCompileTarget `枚举）和`profile`（为 Slang 编译器支持的某一配置文件的 ID），其他可以默认
    ```c_cpp
    TargetDesc targetDesc;
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = globalSession->findProfile("glsl_450");  // 如果没有想用的proflile，可以使用SlangProfileID(0)
    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;
    ```
- 将代码加载到session中的最简单方法是使用`ISession::loadModule()`
  ```
  IModule* module = session->loadModule("MyShaders");
  ```
  - 在宿主 C++ 代码中执行 `loadModule("MyShaders")` 类似于在 Slang 代码中使用 `import MyShaders`
  - 会自动检测&&验证带有`[shader(...)]`属性的entry point，然后使用`IModule::findEntryPointByName()`在该模块中查找entry point
    ```c_cpp
    Slang::ComPtr<IEntryPoint> computeEntryPoint;
    module->findEntryPointByName("myComputeMain", computeEntryPoint.writeRef());
    ```
- Slang中很多操作可以选择性地生成诊断输出，类型为`IBlob`，然后通过 `getBufferPointer()`访问 Blob 的内容，通过 `getBufferSize()` 访问内容的大小
  ```c_cpp
  Slang::ComPtr<IBlob> diagnostics;
  Slang::ComPtr<IModule> module(session->loadModule("MyShaders", diagnostics.writeRef()));
  if(diagnostics)
      fprintf(stderr, "%s\n", (const char*) diagnostics->getBufferPointer());
  ```
  - `slang::IBlob` 与一些 Direct3D 编译 API 所使用的 `ID3D10Blob` 和 `ID3DBlob` 接口保持二进制兼容。
- loadModule可以加载任意数量的module，其中会包含多个entry points，所以需要使用==composition==（组合体）确定哪些东西要一起使用
  ```c_cpp
  IComponentType* components[] = { module, entryPoint };
  Slang::ComPtr<IComponentType> program;
  session->createCompositeComponentType(components, 2, program.writeRef());
  ```
  - `slang::IModule` 和 `slang::IEntryPoint` 均继承自 `slang::IComponentType`
  - 通过 `ISession::createCompositeComponentType()` 创建组合体
  - 组合体的作用：①确定哪些代码属于已编译的着色器程序，②为程序中的代码确立了一种可用于layout的顺序
- Slang API 允许在任何 `IComponentType` 上使用 `getLayout()` 查询layout，因为有时需要对着色器参数及其layout进行reflection
  ```c_cpp
  slang::ProgramLayout* layout = program->getLayout();
  ```
  - reflection就是在主机端 查询着色器内部的一些信息
  - `ProgramLayout` 的生命周期与返回它的 `IComponentType` 相关联
  - 由于为着色器参数计算的layout可能取决于target，因此`getLayout()`实际上会接受一个`targetIndex`参数，表示要查询lauout的target的索引（默认为0）
- 生成代码前，需要==link==所有的cross-module references。想要为程序指定额外的编译器选项，可通过调用`IComponentType::link` 或 `IComponentType::linkWithOptions` 来实现
  ```c_cpp
  Slang::ComPtr<IComponentType> linkedProgram;
  Slang::ComPtr<ISlangBlob> diagnosticBlob; // link相关的诊断消息
  program->link(linkedProgram.writeRef(), diagnosticBlob.writeRef());
  ```
  - link可用于执行链接时特化，这是着色器特化的推荐方法。详情见[这里](https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/10-link-time-specialization.html)
- link过的`IComponentType`，可以使用`IComponentType::getEntryPointCode()`获取其中指定entry point的==kernel code==
  ```c_cpp
  int entryPointIndex = 0; // only one entry point
  int targetIndex = 0; // only one target
  Slang::ComPtr<IBlob> diagnostics; // 诊断信息
  Slang::ComPtr<IBlob> kernelBlob;  // 可用于访问生成的代码（无论是二进制还是文本形式）
  linkedProgram->getEntryPointCode(
      entryPointIndex,
      targetIndex,
      kernelBlob.writeRef(),
      diagnostics.writeRef());
  ```
  - 在许多情况下，可以直接将`kernelBlob->getBufferPointer()`传递给相应的图形API，以将内核代码加载到GPU上

#### Multithreading

- slang中函数和方法在任意同一时刻只能在单个线程上被调用
- 除了通过[host-callable](https://docs.shader-slang.org/en/latest/external/slang/docs/cpu-target.html#host-callable)生成的`ISlangSharedLibrary`接口外，绝大多数 Slang API 的 COM 接口未采用原子引用计数

#### Compiler Options

- `SessionDesc`和其中的`TargetDesc`均包含用于编码的`CompilerOptionEntry`数组，用于指定要应用于session或target的额外编译器选项
  ```c_cpp
  struct CompilerOptionEntry {
      CompilerOptionName name;
      CompilerOptionValue value;
  };
  ```
  - `CompilerOptionName`用于指定要设置的==编译器选项==的 enum
  - `CompilerOptionValue`用于为编译器选项的编码，最多两个整数或字符串值
    ```c_cpp
    enum class CompilerOptionValueKind {
        Int,
        String
    };
    
    struct CompilerOptionValue {
        CompilerOptionValueKind kind = CompilerOptionValueKind::Int;
        int32_t intValue0 = 0;
        int32_t intValue1 = 0;
        const char* stringValue0 = nullptr;
        const char* stringValue1 = nullptr;
    };
    ```
  - 所有可设置的编译器选项，以及其 CompilerOptionValue 编码的含义，部分如下，完整见[这里](https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/08-compiling.html#compiler-options)
    |IncludeCompilerOptionName|Description|
    |--|--|
    |MacroDefine|指定一个预处理器宏定义项。stringValue0 编码宏名称，stringValue1 编码宏的值。|
    |Include|指定一个额外的搜索路径。stringValue0 对该附加路径进行编码。|
    |Language|指定输入语言。intValue0 对在 SlangSourceLanguage 中定义的值进行编码。|
    |MatrixLayoutColumn</br>MatrixLayoutRow|默认使用列/行主矩阵布局。intValue0 为该设置编码一个布尔值。|
    |Profile|指定目标配置文件。intValue0 对由 IGlobalSession::findProfile() 返回的原始配置文件表示形式进行编码。|
    |Stage|指定目标入口点阶段。intValue0 对在 SlangStage 枚举中定义的阶段进行编码。|
    |Target|指定目标格式。其效果与设置 TargetDesc::format 相同。|
    |WarningsAsErrors|指定一个将被视为错误的警告列表。stringValue0 编码为以逗号分隔的警告代码或名称列表，也可以为“all”以表示所有警告。|
    |DisableWarnings|指定要禁用的警告列表。stringValue0 编码为以逗号分隔的警告代码或名称列表。|
    |EnableWarning|指定要启用的警告列表。stringValue0 编码为以逗号分隔的警告代码或名称列表。|
    |DisableWarning|指定要禁用的警告。stringValue0 编码为警告代码或名称。|
    |ReportDownstreamTime|开启/关闭下游编译时间报告。intValue0 为该设置编码一个布尔值。|
    |ReportPerfBenchmark|开启/关闭编译器不同部分耗时的报告功能。intValue0 为该设置编码一个布尔值。|
    |SkipSPIRVValidation|指定是否在输出 SPIR-V 后跳过验证步骤。intValue0 为该设置编码一个布尔值。|
    |Capability|指定编译目标中可用的附加功能。intValue0 对 CapabilityName 枚举中定义的功能进行编码。|
    |DebugInformation|指定要包含在生成代码中的调试信息级别。intValue0 对在 SlangDebugInfoLevel 枚举中定义的值进行编码。|
    |Optimization|指定优化级别。intValue0 对 SlangOptimizationLevel 枚举中定义的设置值进行编码。|
    |Obfuscate|指定是否启用混淆。启用混淆后，Slang 会从目标代码中移除变量和函数名，并将其替换为哈希值。intValue0 为该设置编码一个布尔值。|
    |VulkanUseGLLayout|启用后，将在原始缓冲区的加载/存储操作中使用 std430 布局，而非 D3D 缓冲区布局。intValue0 为该设置指定一个布尔值。|
    |EmitSpirvViaGLSL|设置后会先生成 GLSL 代码，再通过 glslang 生成最终的 SPIR-V 代码。intValue0 为该设置指定一个布尔值。|
    |EmitSpirvDirectly|启用后，将使用 Slang 的直接生成 SPIR-V 后端，直接从 Slang 中间表示（IR）生成 SPIR-V。intValue0 为该设置指定一个布尔值。|
    |DumpIntermediates|启用后将输出中间源代码。intValue0 为该设置指定一个布尔值。|
    |DebugInformationFormat|指定调试信息的格式。intValue0 是在 SlangDebugInfoFormat 枚举中定义的一个值。|
    |ValidateUniformity|启用后将执行[uniformity analysis(一致性分析)](https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/a1-05-uniformity.html)。|

#### Debugging

- Slang 的 SPIR-V 后端支持使用 [NonSemantic Shader DebugInfo Instructions](https://github.com/KhronosGroup/SPIRV-Registry/blob/main/nonsemantic/NonSemantic.Shader.DebugInfo.100.asciidoc) 生成调试信息
  - SPIR-V 时启用调试信息，可在使用 slangc 工具时指定 `-emit-spirv-directly` 和 `-g2` 参数，或在使用 API 时将 `EmitSpirvDirectly` 设为 1、将 `DebugInformation` 设为 `SLANG_DEBUG_INFO_LEVEL_STANDARD`。
  - 调试功能已通过 RenderDoc 测试。

### Using the Reflection API

#### Compiling a Program

- 首先使用编译API，然后使用`getLayout()`提取reflection信息

#### Types and Variables

- GPU shader programming中，相同的类型可能会有不同的layout，这取决于其使用方式
- `VariableReflection`表示输入程序中的variable（变量）声明
  - 变量包括global shader parameters、struct类型的字段以及entry-point的参数
  - `VariableReflection`不包含layout信息，可以查询其名称和类型
    ```c_cpp
    void printVariable(
        slang::VariableReflection* variable) {
        const char* name = variable->getName();
        slang::TypeReflection* type = variable->getType();
    
        print("name: ");    printQuotedString(name);
        print("type: ");    printType(type);
    }
    ```
- `TypeReflection`表示输入程序中的某种 type（类型）
  - type有多种不同的种类，例如数组、用户定义的struct类型以及 int 等内置类型
  - reflection API 通过 `TypeReflection::Kind` 枚举来表示这些不同的种类
    ```c_cpp
    方法void printType(slang::TypeReflection* type) {
        const char* name = type->getName();
        slang::TypeReflection::Kind kind = type->getKind();
    
        print("name: ");    printQuotedString(name);
        print("kind: ");    printTypeKind(kind);
    }
    ```
    |类型|对应枚举|查询方式|
    |--|--|--|
    |Scalar types（标量类型）|`slang::TypeReflection::Kind::Scalar`|`printScalarType(type->getScalarType());`|
    |Structure types（结构体类型）|`slang::TypeReflection::Kind::Struct`|可以有零个或多个字段（不包含静态）</br>通过`getFieldCount()`和`printVariable(type->getFieldByIndex(id))`枚举所有字段|
    |Arrays（数组）|`slang::TypeReflection::Kind::Array`|通过`printPossiblyUnbounded(type->getElementCount());`和`printType(type->getElementType());`进行查询</br>对于类似`Stuff[]`的数组具有 无限制 大小，`getElementCount()`返回为size_t的最大值|
    |Vectors（向量）|`slang::TypeReflection::Kind::Vector`|通过`printCount(type->getElementCount());`和`printType(type->getElementType());`进行查询|
    |Matrices（矩阵）|`slang::TypeReflection::Kind::Matrix`|通过`printCount(type->getRowCount())`、`printCount(type->getColumnCount())`和`printType(type->getElementType())`查询行数、列数和元素类型|
    |Resources（资源）|`slang::TypeReflection::Kind::Resource`|包括 `StructuredBuffer<int>` 这类简单类型，也有`AppendStructuredBuffer<Stuff>`这类相当复杂的类型 </br>通过`printResourceShape(type->getResourceShape())`、`printResourceAccess(type->getResourceAccess())`和`printType(type->getResourceResultType())`查询其shape、access和result type|
    |Single-Element Containers（单元素容器）|`slang::TypeReflection::Kind::ConstantBuffer`、`ParameterBlock`、`TextureBuffer`、`ShaderStorageBuffer`|可以查询元素类型`printType(type->getElementType())`|
    - resources的result type（结果类型）指的是对该资源执行基本读取操作时所返回的内容。比如`StructuredBuffer<Thing>`的result type为`Thing`，`Texture2D`未明确指定的话默认为`float4`
    - resource 的access （访问方式，SlangResourceAccess）表示着色器代码对资源的读写访问权限。例如，无前缀的Texture2D只读（SLANG_RESOURCE_ACCESS_READ），而RWTexture2D则可以读写（SLANG_RESOURCE_ACCESS_READ_WRITE）。
    - resource的shape（SlangResourceShape）表示该资源的概念层级/维度及其索引方式，可分解为一个基础形状以及一些可能的后缀

#### Layout for Types and Variables

- ==layout units（布局单元）==指的是layout中使用的不同度量单位（例如D3D12中的bytes、t registers和 s registers）
  - 通过`slang::ParameterCategory` 枚举来表示（但是官方避免使用“参数类别”这一术语，因为不太合适）
- `VariableLayoutReflection` 表示为给定变量（其本身是一个 `VariableReflection`）计算出的layout。通过 `getVariable()` 访问底层变量
  - 这个layout存储该变量的**偏移量**（可能存在于多个layout units中），还存储变量中所存储数据的type layout
    ```c_cpp
    void printVarLayout(slang::VariableLayoutReflection* varLayout) {
        print("name"); printQuotedString(varLayout->getName());
        printRelativeOffsets(varLayout);
        key("type layout"); printTypeLayout(varLayout->getTypeLayout());
    }
    ```
  - 偏移量是始终相对于包围该变量的 struct 类型、作用域或其他上下文 的 相对偏移，使用`VariableLayoutReflection::getOffset()`查询任意layout uint中的相对偏移
    ```c_cpp
    void printOffset(slang::VariableLayoutReflection* varLayout, slang::ParameterCategory layoutUnit) {
        print("value: "); print(varLayout->getOffset(layoutUnit));
        print("unit: "); printLayoutUnit(layoutUnit);
    }
    ```
  - 通过 `getCategoryCount()` 和 `getCategoryByIndex()` 查询给定variable layout所使用的layout unit：
    ```c_cpp
    int usedLayoutUnitCount = varLayout->getCategoryCount();
    for (int i = 0; i < usedLayoutUnitCount; ++i) {
        auto layoutUnit = varLayout->getCategoryByIndex(i);
        printOffset(varLayout, layoutUnit);
    }
    ```
- ==space（空间）==是指在某些target和layout uints中，variable的偏移量可能包含一个额外的维度，表示Vulkan/SPIR-V 的描述符集、D3D12/DXIL 的寄存器空间 或 WebGPU/WGSL 的绑定组
  - 使用 `getBindingSpace()` 查询给定layout unit的variable layout的相对空间偏移量
    ```c_cpp
    size_t spaceOffset = varLayout->getBindingSpace(layoutUnit);
    switch(layoutUnit) {
        default: break;
        case slang::ParameterCategory::ConstantBuffer:
        case slang::ParameterCategory::ShaderResource:
        case slang::ParameterCategory::UnorderedAccess:
        case slang::ParameterCategory::SamplerState:
        case slang::ParameterCategory::DescriptorTableSlot:
            print("space: "); print(spaceOffset);
            break;
    }
    ```
- `TypeLayoutReflection`表示为某个类型计算出的layout。通过 `TypeLayoutReflection::getType()` 访问计算该layout所基于的类型
  ```c_cpp
  void printTypeLayout(slang::TypeLayoutReflection* typeLayout) {
      print("name: "); printQuotedString(typeLayout->getName());
      print("kind: "); printTypeKind(typeLayout->getKind());
      printSizes(typeLayout);
  }
  ```
  - type layout 主要是存 该类型的大小。和variable一样，可以查询任意layout uint中的大小
    - 对于特定的layout uint，类型的大小可能是无界的，使用`~size_t(0)`表示
    ```c_cpp
    void printSize(slang::TypeLayoutReflection* typeLayout, slang::ParameterCategory layoutUnit) {
        size_t size = typeLayout->getSize(layoutUnit);
    
        key("value"); printPossiblyUnbounded(size);
        key("unit"); writeLayoutUnit(layoutUnit);
    }
    ```
  - 使用 `getCategoryCount()` 和 `getCategoryByIndex()` 遍历给定type layout所使用的layout unit，和variable layout一样
    ```c_cpp
    void printSizes(slang::TypeLayoutReflection* typeLayout) {
        print("size: ");
        int usedLayoutUnitCount = typeLayout->getCategoryCount();
        for (int i = 0; i < usedLayoutUnitCount; ++i) {
            auto layoutUnit = typeLayout->getCategoryByIndex(i);
            print("- "); printSize(typeLayout, layoutUnit);
        }
    }
    ```
  - type layout可以反映指定layout uint的类型和`TypeLayoutReflection::getAlignment()`的==对齐方式==
    - 通常只有当layout uint为bytes（`slang::ParameterCategory::Uniform`）时，对齐方式才具有实际意义。
    - 给定layout uint的type layout的==stride==是其大小向上取整至对齐值后的结果，用作数组中连续元素之间的间距。通过`TypeLayoutReflection::getStride()`。
    ```c_cpp
    void printTypeLayout(slang::TypeLayoutReflection* typeLayout) {
        if(typeLayout->getSize() != 0) {
            print("alignment in bytes: "); print(typeLayout->getAlignment());
            print("stride in bytes: "); print(typeLayout->getStride());
        }
    }
    ```
  - type layout可能根据类型的种类存储额外信息
    - Structure、Array、Matrix、Single-Element Containers 等类型的type layout，和[Types and Variables](# Types and Variables)的表格里面几乎一样，只不过把带查询的变量从`type`换成`typeLayout`
    - `ConstantBuffer<T>`会隐藏其元素`T`所占用的字节数，而且包含任何**不只是以字节为单位**的数据会发生**渗透**问题；而`ParameterBLock<T>`会隐藏其元素`T`所使用的绑定位、寄存器或插槽。
      - 它们俩的type layout包含容器元素的layout信息，也包含 容器本身的layout信息
      - 
      ```c_cpp
      struct DirectionalLight { // 在D3D12中，占用28字节
          float3 direction;
          float3 intensity;
      }
      ConstantBuffer<DirectionalLight> light; // 在D3#12中，占用一个b寄存器
      
      struct ViewParams {       // 在D3D12中，占用28字节和一个t寄存器；在vulkan中，占用28字节（std140）和1个binding
          float3 cameraPos;
          float3 cameraDir;
          TextureCube envMap;   // 在ViewParams内的偏移量为0个binding；但view.envMap相对于view的偏移量为1个binding
      }
      ConstantBuffer<ViewParams> view;        // 在D3#12中，占用一个b寄存器和一个t寄存器；在vulkan中，占用两个binding
      
      struct Material {         // 在Vulkan中，占用 3 个绑定位
          Texture2D albedoMap;                // material.albedoMap相对于material的偏移量为 0 个binding
          Texture2D glossMap;
          SamplerState sampler;
      }
      ParameterBlock<Material> material;      // 在vulkan中，占用1个space
      ConstantBuffer<Material> material;      // 在vulkan中，占用3个binding，但不对应常量缓冲区
      
      struct PointLight {
          float3 position;
          float3 intensity;
      }
      struct LightingEnvironment {  // 在Vulkan中，占用 316字节和1个binding
          TextureCube envMap;   // lightEnv.envMap的累积binding偏移量为1个binding，LightingEnvironment::envMap的相对偏移量为0个binding
          PointLight pointLights[10];
      }
      ParameterBlock<LightingEnvironment> lightEnv; // 在vulkan中，使用一个描述符set
      ```
    - 对于Single-Element Containers，元素和容器的布局信息都需要支持存储相对于它们整体的偏移信息（而非仅存储大小）
      ```c_cpp
      case slang::TypeReflection::Kind::ConstantBuffer:
      case slang::TypeReflection::Kind::ParameterBlock:
      case slang::TypeReflection::Kind::TextureBuffer:
      case slang::TypeReflection::Kind::ShaderStorageBuffer: {
          print("z: "); printOffsets(typeLayout->getContainerVarLayout());
      
          auto elementVarLayout = typeLayout->getElementVarLayout();
          print("element: "); printOffsets(elementVarLayout);
          print("type layout: "); printTypeLayout(elementVarLayout->getTypeLayout();
      }
      break;
      ```
    - 在single-element container上建议使用`getElementVarLayout()`而不是`getElementTypeLayout()`
- 如果未指定layout uint，所有`TypeLayoutReflection`的方法`getSize()`、`getAlignment()`和g`etStride()`，以及`VariableLayoutReflection::getOffset()`，**默认以字节为单位返回信息**

#### Programs and Scopes

- `ProgramLayout`主要包含==global scope（全局作用域）==，以及多个entry point（可能0个）。
  - 在编译并链接 Slang 程序后，通过 `IComponentType::getLayout()` 获取
  - global scope域和入口点都是==scope==的示例。scope通过`VariableLayoutReflection`表示
- Slang 编译器在编译过程中对全局作用域着色器参数声明所执行的步骤（其中部分步骤为可选步骤）
  - 如果一个shader声明的都是不透明变量的全局作用域参数，Slang 编译器会将所有这些参数声明归为一个 struct 类型，然后仅保留一个这个struct的全局作用域参数
    ```c_cpp
    // shader声明的
    Texture2D diffuseMap;
    TextureCube envMap;
    SamplerState sampler;
    // slang编译器转换后的
    struct Globals {
        Texture2D diffuseMap;
        TextureCube envMap;
        SamplerState sampler;
    }
    uniform Globals globals;
    ```
  - 如果全局作用域参数同时包含不透明类型和普通类型，slang编译器还会归为一个struct，但是包在`ConstantBuffer<>`中
    ```c_cpp
    struct Globals {
        Texture2D diffuseMap;
        TextureCube envMap;
        SamplerState sampler;
    
        float3 cameraPos;
        float3 cameraDir;
    }
    ConstantBuffer<Globals> globals;
    ```
  - 对于 D3D12/DXIL、Vulkan/SPIR-V 和 WebGPU/WGSL 这类目标，如果存在未指定显式空间的全局作用域参数，则会将全局作用域声明封装在一个提供默认空间的 `ParameterBlock<>` 中
  - 如果scope需要同时引入一个constant buffer和一个parameter block，则该scope的表现形式是`ParameterBlock<...>`，而非`ParameterBlock<ConstantBuffer<...>>`，即隐式constant buffer的绑定信息会作为parameter block的容器变量布局的一部分被获取到
  - **建议不要**在需要系统且可靠地反射任何可能输入的着色器代码的应用程序中使用`getParameterCount()` 和 `getParameterByIndex()`。
  - **建议使用** `getGlobalParamsVarLayout()` 而非 `getGlobalParamsTypeLayout()`，考虑全局作用域可能被应用偏移量的情况（同时也为了更统一地处理全局作用域和入口点作用域）。
- `EntryPointReflection`提供某个entry point的信息（比如）
  ```c_cpp
  slang::EntryPointReflection* entryPointLayout = xxx;
  print("stage: "); printStage(entryPointLayout->getStage());
  ```
  - entry point的参数会被组合到一个struct中，随后在需要时会自动封装到constant buffer或parameter block中。比global scope相比还可以声明result type（结果类型）。若存在结果类型，函数的返回值或多或少相当于一个额外的out参数
    ```c_cpp
    printScope(entryPointLayout->getVarLayout());
    
    auto resultVarLayout = entryPointLayout->getResultVarLayout();
    if (resultVarLayout->getTypeLayout()->getKind() != slang::TypeReflection::Kind::None)
        key("result"); printVarLayout(resultVarLayout);
    ```
  - 和global scope一样，**建议不要**使用`getParameterCount()` 和 `getParameterByIndex()` 方法
  - **建议使用** `EntryPointReflection::getVarLayout()` 而非 `::getTypeLayout()`，以便更准确地反映偏移量的计算方式并将其应用于entry point的参数
  - 针对的编译阶段可以获取额外的信息
    ```c_cpp
    if(entryPointLayout->getStage() == SLANG_STAGE_COMPUTE) {
        entryPointLayout->getComputeThreadGroupSize(3, sizes);
        print("thread group size: ");
        print("x: "); print(sizes[0]);
        print("y: "); print(sizes[1]);
        print("z: "); print(sizes[2]);
    }
    ```
  - 用于着色器阶段之间传递的 varying 变量，在layout里会体现为使用以下几类插槽类型名字
    ||用途|
    |--|--|
    |slang::ParameterCategory::VaryingInput|输入参数|
    |slang::ParameterCategory::VaryingOutput|输出参数|
    |slang::ParameterCategory::VaryingInput 和 ::VaryingOutput|inout 参数|
    |无|系统值参数（以使用 SV_* 语义）|
  - 通过 `getSemanticName()` 和 `getSemanticIndex()`获取entry point参数的semantic（语义）
    ```c_cpp
    if (varLayout->getStage() != SLANG_STAGE_NONE) {
        print("semantic: ");
        print("name: "); printQuotedString(varLayout->getSemanticName());
        print("index: "); print(varLayout->getSemanticIndex());
    }
    ```

#### Calculating Cumulative Offsets

- 以上的所有代码都只计算了variable layout的相对offset
- 对于任意给定的layout uint，variable layout的累计偏移量可通过累加指向该变量的==access path（访问路径）==上所有相对偏移量来计算
  ```c_cpp
  struct Material {
      Texture2D albedoMap;
      Texture2D glossMap;         // 偏移量1
      SamplerState sampler;
  }
  struct LightingEnvironment {
      TextureCube environmentMap;
      float3 sunLightDir;
      float3 sunLightIntensity;
  }
  struct Params {
      LightingEnvironment lights; 
      Material material;            // 偏移量1
  }
  uniform Params params;            // 偏移量0
  // 所以params.material.glossMap的偏移量是params（0）、material（1）和 glossMap（1）的偏移量的求和
  ```
  - 可以使用递归+单链表的方式来实现累加
    ```c_cpp
    struct CumulativeOffset {
        int value; // the actual offset
        int space; // the associated space
    };
    struct AccessPathNode { // 单链表
        slang::VariableLayoutReflection* varLayout;
        AccessPathNode* outer;
    };
    struct AccessPath {
        AccessPathNode* leafNode = nullptr;
    };
    CumulativeOffset calculateCumulativeOffset(slang::ParameterCategory layoutUnit, AccessPath accessPath) {
        for(auto node = accessPath.leafNode; node != nullptr; node = node->outer) {
            result.value += node->varLayout->getOffset(layoutUnit);
            result.space += node->varLayout->getBindingSpace(layoutUnit);
        }
    }
    void printOffsets(slang::VariableLayoutReflection* varLayout, AccessPath accessPath) {
        print("cumulative:");
        for (int i = 0; i < usedLayoutUnitCount; ++i) {
            print("- ");
            auto layoutUnit = varLayout->getCategoryByIndex(i);
            printCumulativeOffset(varLayout, layoutUnit, accessPath);
        }
    }
    void printCumulativeOffset(slang::VariableLayoutReflection* varLayout, slang::ParameterCategory layoutUnit, AccessPath accessPath) {
        CumulativeOffset cumulativeOffset = calculateCumulativeOffset(layoutUnit, accessPath);
    
        cumulativeOffset.offset += varLayout->getOffset(layoutUnit);
        cumulativeOffset.space += varLayout->getBindingSpace(layoutUnit);
    
        printOffset(layoutUnit, cumulativeOffset.offset, cumulativeOffset.space);
    }
    ```

- 对于single-element container中的constant buffer，不能在access path的过高层级上累加贡献
- 对于single-element container中的param block，不得对超出封闭参数块的贡献进行求和

#### Determining Whether Parameters Are Used 

- 想要知道使用过的参数 以及由哪些entry point或stage使用，使用`IComponentType::getEntryPointMetadata()` 查询额外==metadata(元数据)==
  ```c_cpp
  slang::IComponentType* program = ...;
  slang::IMetadata* entryPointMetadata;
  program->getEntryPointMetadata(entryPointIndex, /*target index*/0, &entryPointMetadata);
  ```

- 然后对给定的layout uint（及其绝对位置）使用`IMetadata::isParameterLocationUsed()`，就可以知道某参数使用的stage
- 在Metal / WGSL上有个限制：编译时会把顶点 / 片元间传递的 varying 输入打包成一个结构体，元数据只记录整个结构体是否被使用。因为 只要结构体里有一个变量被用，就会判定整个结构体所有变量都被使用，无法精确区分单个变量。
  - SPIR‑V / GLSL 不受影响，因为每个 varying 是独立全局变量，死码消除和元数据都能单独跟踪，可以精确判断每个变量是否被使用。
  - 不变资源类型（例如`DescriptorTableSlot`、`ShaderResource`）在任何target上均不受此限制影响。

### Supported Compilation Targets

#### Background and Terminology

- GPU 通常通过抽象了多种 GPU 处理器架构和版本的 API 进行编程。GPU API 通常会定义一种中间语言，介于 Slang 这类高级语言编译器与 API 驱动程序中针对特定 GPU 的编译器之间。
- GPU 代码执行发生在pipeline的上下文中。一条pipeline包含一个或多个stage以及它们之间的数据流连接。
  - 部分阶段是可编程的，会运行用户通过 Slang 等语言编译而来的、自定义的内核；而另一些阶段则是固定功能的，用户仅能对其进行配置，而非编程。
  - Slang 支持三种不同的pipeline: Rasterization 光栅化（`vertex`和`fragment` stage)、Compute 计算（`compute` stage）、Ray Tracing 光线追踪（`intersection`和`miss` stage）
