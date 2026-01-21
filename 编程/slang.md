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