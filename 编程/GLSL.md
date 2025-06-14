## 语法：

### layout

- [**`layout(push_constant)`**](https://vkguide.dev/docs/chapter-3/push_constants/)：`push_constant`用于定义一种特殊的常量数据传递方式，它可以简单、高性能地给任何shader发送少量数据。这些数据存储在command buffer中，适合传递一些频繁变化但数据量较小的参数，如变换矩阵、颜色值等。使用`push_constant`可以减少内存带宽的占用，提高性能。

- `layout (constant_id = N)`：一种用于声明 **着色器编译时常量（Specialization Constant）** 的布局限定符（Layout Qualifier），它允许在运行时动态修改着色器中的常量值而无需重新编译着色器

    - 作用一：**动态配置**：通过 `constant_id` 定义的常量可在 Vulkan 管线创建时通过 **Specialization Constants** 机制动态设置（例如调整循环次数、分支条件等）

        ```glsl
        layout(constant_id = 0) const uint LOCAL_SIZE_X = 32;
        layout(local_size_x = LOCAL_SIZE_X) in; // 用常量定义工作组大小
        ```

    - 作用二：**性能优化**：避免运行时分支判断，在编译时优化未使用的代码路径。

        ```c++
        layout(constant_id = 1) const bool USE_PHONG = true;
        void main() {
            if (USE_PHONG) { // Phong 光照逻辑
            } else { // Lambert 光照逻辑
            }
        }
        ```
### 扩展

- 语法： `#extension xxxxxxx : enable`

- `GL_GOOGLE_include_directive`：支持#include

- `GL_EXT_shader_atomic_float`：支持float的原子操作

- 
### 杂

- 结构体中不能包含成员函数，需要结构体外面定义

