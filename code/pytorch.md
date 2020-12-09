# 理论

- **模型训练的重点过程就两点：前向传播和反向传播**。前向传播用numpy就能很好完成，深度学习框架解决的核心问题就是反向传播时的梯度计算和更新.[^2]

- 计算图[^2]：用图的方式表示计算过程。例如，左边的是抽象的，右边是pytorch中的

    <img src="images/v2-8441c04bc991aaf0b6d03879f3ad4d98_720w.jpg" alt="img" style="zoom: 33%;" /><img src="images/v2-8aabf74ecd67f6609115b981dcd4b5ae_720w.jpg" alt="img" style="zoom: 61%;" />

# 各种函数

## Tensor

- **torch.Tensor.grad_fn**[^1][^2]：反向传播时，用来计算梯度的函数，即指示梯度函数是哪种类型。Tensor类属性方法。叶子节点通常为None，只有结果节点的grad_fn才有效

- **torch.Tensor.is_leaf**[^2]：标记该tensor是否为叶子节点。Tensor类变量，布尔值。

    所有requires_grad=False的Tensors都为叶子节点

    所有用户显示初始化的Tensors也为叶子节点

    由各种操作(operation)的结果隐式生成的不是叶子节点，比如`a.cuda()`以及加减乘除操作

## torch

- **torch.expand(*sizes)**：返回当前张量在某维扩展更大后的张量

    ```python
    a = torch.rand(2,3)
    b = a.expand(6,3)
    ```

    

- **torch.view()**：相当于numpy中的resize()。例如

    ```python
    a = torch.rand(2,3,4,5)
    b = a.view(3,4,5,-1) # -1表示该维度进行推断，但是只有一个维度能为-1
    # b.size() = [3, 4, 5, 2]
    ```

- 

- 

## torch.autograd

- **torch.autograd.backward()**[^1]：反向传播，根据链式法则叶子节点的梯度。参数如下

    ```python
    torch.autograd.backward(
    		tensors, 
    		grad_tensors=None, 
    		retain_graph=None, 
    		create_graph=False, 
    		grad_variables=None)
    ```

    - 对于一个tensor`z`，`torch.autograd.backward(z)==z.backward()`这两种方式等价
    - grad_tensors为前面算出的梯度，可以理解成求当前梯度时的权重

- **torch.autograd.grad()**[^1]：

## torch.nn

## torch.nn.functional

# 参考资料

[^1]:[Pytorch autograd,backward详解](https://zhuanlan.zhihu.com/p/83172023)
[^2]:[【深度学习理论】一文搞透pytorch中的tensor、autograd、反向传播和计算图](https://zhuanlan.zhihu.com/p/145353262)
[^3]:[PyTorch: 梯度下降及反向传播](https://blog.csdn.net/m0_37306360/article/details/79307354)