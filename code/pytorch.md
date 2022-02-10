# 理论

- **模型训练的重点过程就两点：前向传播和反向传播**。前向传播用numpy就能很好完成，深度学习框架解决的核心问题就是反向传播时的梯度计算和更新.[^2]

- 计算图[^2]：用图的方式表示计算过程。例如，左边的是抽象的，右边是pytorch中的

    <img src="images/v2-8441c04bc991aaf0b6d03879f3ad4d98_720w.jpg" alt="img" style="zoom: 33%;" /><img src="images/v2-8aabf74ecd67f6609115b981dcd4b5ae_720w.jpg" alt="img" style="zoom: 61%;" />

## CNN的反向传播公式[^6]

- **池化层的反向传播**

    <img src="images/v2-76b89f5687c280675964e4ec50288e55_r.jpg" alt="preview" style="zoom:67%;" />

    - **最大池化**：上图中池化后的数字6对应于池化前的红色区域，实际上只有红色区域中最大值数字6对池化后的结果有影响，权重为1，而其它的数字对池化后的结果影响都为0。

        ∴反向传播时，左边红色区域中6的误差等于右边红色区域中的$\delta$误差，而1、1、5对应位置的误差为0。

        因此，最大池化前向传播时，要记录区域的最大值+其位置，方便后续反向传播。

    - 平均池化：其前向传播时区域中每个值对池化后结果贡献的权重都为区域大小的倒数，即$\frac{1}{k*k}$，所以$\delta$误差反向传播回来时，在区域每个位置的delta误差都为池化后delta误差除以区域的大小，即$\frac{\delta}{k*k}$。

- **激活层的反向传播**：$\delta^{l}=\delta^{l+1}\odot \sigma^{\prime}\left(z^{l}\right)$

- **卷积层的反向传播**

    <img src="images/v2-88bd6e9ab498a92847a6bada18f37ee2_720w.jpg" alt="img" style="zoom:67%;" />
    
    - 输入层（$l$层）的$\delta$误差：$\delta^{l}=\delta^{l+1} * Rotate 180\left(w^{l+1}\right)$，即把卷积核旋转180°后与后一层（$l+1$层）的$\delta$误差卷积
      
      <img src="images/image-20201209211717228.png" alt="image-20201209211717228" style="zoom:67%;" />
      
    - 输入层与输出层之间**卷积核的导数**：$\frac{\partial C}{\partial w^{l}}=\delta^{l} * \sigma\left(z^{l-1}\right)$，即卷积核的每个位置，在输入层能影响到的像素与卷积结果进行卷积，上图。注意输入层需要先padding
    
    - 输入层与输出层之间**bias的导数**：$\frac{\partial C}{\partial b^{l}}=\sum_{x} \sum_{y} \delta^{l}$

# 各种操作

## 扩展torch.autograd：自定义层

通过继承子类`torch.autograd.Function`而非`torch.nn.Module`来扩展torch.autograd（自定义层）[^4][^9]，并且必须实现两个静态函数（用`@staticmethod`修饰）[^11][^12]：

- `forward(ctx: Any, *args: Any, **kwargs: Any) → Any`：
    - 输入值可以为任何python的对象
    - 使用`ctx.save_for_backward(a, b)`保存forward()静态方法中的张量, 从而可以在backward()静态方法中调用
- `backward(ctx: Any, *grad_outputs: Any) → Any`：
    - 输入参数的个数=`forward()`中返回值的个数，即每个输入对应了`forward()`中对应返回值的梯度
    - 返回参数的个数=`forward()`中输入值的个数，每个返回值对应了`forward()`输入值的梯度。如果`forward()`中输入有非Ternsor，那么它们在backward()中对应的返回值为None
    - 如果`forward()`的输入不需要梯度（可以通过`ctx.needs_input_grad`获得一个bool的tuple，每个元素表示对应`forward()`中输入是否需要求导），或者这些输入不是`Tensor`，那`backward()`中对应的返回值可以为`None`
    - 通过`a, b = ctx.saved_tensors`重新获得在`forward()`中保存的`Tensor`

ctx和self的区别[^13]：

- ctx是context的缩写, 翻译成"上下文; 环境"，专门用在静态方法中
- self指的是实例对象; 而ctx用在静态方法中, 调用的时候不需要实例化对象, 直接通过类名就可以调用, 所以self在静态方法中没有意义

示例：自定义LinearFunction[^10]

## model结构&数据流可视化

安装 tensorboard (conda) 和 tensorboardX (pip)，然后在代码中添加如下内容[^5]

```python
import torch as t
import torchvision as tv
from tensorboardX import SummaryWriter
class DRConv()
input = t.rand(4, 3, 32, 32)
net = DRConv(3, 10, 3, 9)
with SummaryWriter(comment="DRconv") as w:
    w.add_graph(net, (input,))
```

运行后，会在代码所在目录生成一个`runs`文件。在代码所在目录的终端里运行命令`tensorboard --logdir=runs`，就可以获得浏览器地址，比如`http://localhost:6006/`。从浏览器进去后，选择`GRAPHS`即可

## label变成one hot编码

```python
index = t.argmax(t.rand(4, 5, 32, 32), dim=1) # label大小为N*H*W，每个位置中存放着label的下标

onehot = t.zeros(4, 5, 32, 32).scatter(1, index.unsqueeze(1), 1)  # onehot大小为N*L*H*W，第二个维度中为着label的个数
# PS: 
# index.unsqueeze(1)是为了让其大小变为N*1*H*W；
# 倒数第三个1表示把第二个维度变成onehot编码；
# 倒数第一个1表示填充的数字为1
```

## 并行化读取数据

在[学习tensorflow](https://tf.wiki/)的时候，发现 tf 有一个函数可以并行化读取数据以提高训练效率

<img src="images/image-20210120114828198.png" alt="image-20210120114828198" style="zoom:67%;" />

网上找了一下，pytorch的[Dataloader](https://pytorch.org/docs/stable/data.html#module-torch.utils.data)中有一个类似的参数`prefetch_factor`

> **prefetch_factor** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional**,* *keyword-only arg*) – Number of sample loaded in advance by each worker. `2` means there will be a total of 2 * num_workers samples prefetched across all workers. (default: `2`)

还可以使用 Nvidia 提出的分布式框架 Apex提供的[解决方案](https://zhuanlan.zhihu.com/p/66145913)

## 获取中间层的特征图和梯度

### Hook技术

由于pytorch在计算过程中会舍弃除了叶子节点以外的中间层的特征图和梯度，想要获取他们并进行一定的修改，可以使用Hook技术

hook包含三种函数：

- [register_hook(hook)](https://pytorch.org/docs/stable/autograd.html#torch.Tensor.register_hook)：该函数属于Tensor对象为Tensor注册一个backward hook，用来获取变量的梯度

  其中的hook为函数名称（可以任意），必须遵循如下的格式：`hook(grad) -> Tensor or None`，其中grad为获取的梯度

  具体实例：

  ```python
  import torch
  
  grad_list = []
  def print_grad(grad):
      grad = grad * 2	
      grad_list.append(grad)
  
  x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
  h = x.register_hook(print_grad)    # double the gradient
  out = x.pow(2).sum()
  out.backward()
  print(grad_list) # [tensor([[ 4., -4.], [ 4.,  4.]])]
  
  # 删除hook函数
  h.remove()
  ```

- [register_forward_hook(hook)](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)：该函数属于Module对象，返回为`torch.utils.hooks.RemovableHandle`。用于前向传播中进行hook，可以提取特征图

  在网络执行`forward()`之后，执行hook函数，需要具有如下的形式：

  ```python
  hook(module, input, output) -> None or modified output
  ```

  hook可以修改input和output，但是不会影响**forward**的结果。最常用的场景是需要提取模型的某一层（不是最后一层）的输出特征，但又不希望修改其原有的模型定义文件，这时就可以利用forward_hook函数。

- [register_backward_hook(hook)](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_backward_hook)：该函数属于Module对象，返回为`torch.utils.hooks.RemovableHandle`。用于反向传播中进行hook，可以提取梯度

  每一次module的inputs的梯度被计算后调用hook，hook具有如下的形式：

  ```python
  hook(module, grad_input, grad_output) -> Tensor or Non
  ```

  `grad_input` 和 `grad_output`参数分别表示输入的梯度和输出的梯度，是不能修改的，但是可以通过return一个梯度元组tuple来替代`grad_input`。

  `grad_input`元组包含(`bias的梯度`，`输入x的梯度`，`权重weight的梯度`)，`grad_output`元组包含(`输出y的梯度`)。
  可以在hook函数中通过return来修改`grad_input`

  对于没有参数的Module，比如`nn.ReLU`来说，`grad_input`元组包含(`输入x的梯度`)，`grad_output`元组包含(`输出y的梯度`)。

PS：经过实践，不能将`hook`定义为某个类的成员函数，否则会参数个数会不匹配的（因为会多一个`self`参数）

> 参考：[Pytorch获取中间层信息-hook函数](https://blog.csdn.net/winycg/article/details/100695373)、[官方教程Forward and Backward Function Hooks](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks)

### 自定义forward函数：只能获取中间层的特征

此方法的原理是获取给定模型的每一层，然后在前向传播的时候将想要层的特征保存下来即可

```python
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):	#extracted_layers为一个列表
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
 
    # 自己修改forward函数
    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs
   
extract_list = ["conv1", "maxpool", "layer1", "avgpool", "fc"]
resnet = models.resnet50(pretrained=True)
x = torch.random(1,3,256,256)
extract_result = FeatureExtractor(resnet, extract_list)
print(extract_result(x)[4])  # [0]:conv1  [1]:maxpool  [2]:layer1  [3]:avgpool  [4]:fc
```

> 参考：[PyTorch提取中间层的特征（Resnet）](https://www.codeleading.com/article/8544702154/)

## 参数统计&&共享问题

统计方法有两种：

```python
# 第一种：使用torchsummary
import torchsummary
torchsummary.summary(model, (1, 512, 512))
# 第二种：自定义函数，统计model中可导参数
def count_parameters(model):
	print(sum(p.numel() for p in model.parameters() if p.requires_grad))
```

但是当网络中共享 层（即[参数共享](https://pytorch.org/tutorials/beginner/examples_nn/dynamic_net.html?highlight=share)）时，上面的统计方法都会有问题。例如：

```python
import torch
import torch.nn as nn
import torchsummary
class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,bias=False)
        self.conv2=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,bias=False)
        self.conv3=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,bias=False)
    def forward(self,x):
        x=self.conv1(x)
        out_map=self.conv1(x)
        return out_map
def count_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
model = BaseNet()
torchsummary.summary(model, (1, 512, 512))
count_parameters(model)
```

实际输出参数个数应该为9（因为只有一个3*3的卷积核，没有权重），但是`torchsummary`输出为18，`count_parameters`输出为27

因为`torchsummary`计算时是先把层结构打印下来，然后再统计对各个层的参数求和，`conv1`被调用了两次，所以为18；而在`BaseNet`类里多初始化了`conv2`和`conv3`，即使没有在forward里面调用，但是它也算在`model.parameters()`里面，因此`count_parameters`为27

> 参考教程：[PyTorch几种情况下的参数数量统计](https://zhuanlan.zhihu.com/p/64425750)

# 技巧

## 节省时间

1. **transforms的顺序**：将crop放在color jitter啥的前面，否则会有很多无用的计算

2. **数据放在SSD或者内存里**：将[内存变成硬盘的方式](https://www.jianshu.com/p/6f9b200671bb)为

   ```shell
   mount tmpfs /opt/datasets/ -t tmpfs -o size=16G
   ```

3. 使用GPU加载数据：

## 节省显存

1. **`ReLu`的`inplace`参数**。使用下面的代码可以将model中的`ReLu`全部设置为`inplace=True`

   ```python
   def inplace_relu(m):
       classname = m.__class__.__name__
       if classname.find('ReLU') != -1:
           m.inplace=True
   
   model.apply(inplace_relu)
   ```

   > 参考教程：[Pytorch有什么节省显存的小技巧？ - 郑哲东的回答](https://www.zhihu.com/question/274635237/answer/573633662) 

2. **梯度累加**：即不是每个batch都更新 权重&超参，而是隔几个再更新，这样可以扩大batch_size：
   $$
   \text{真正的batch_size}  = \text{batch_size} * \text{accumulation_steps}
   $$

   ```python
   for i,(features,target) in enumerate(train_loader):
       outputs = model(images)  # 前向传播
       loss = criterion(outputs,target)  # 计算损失
       loss = loss/accumulation_steps   # 可选，如果损失要在训练样本上取平均
   
       loss.backward()  # 计算梯度
       if((i+1)%accumulation_steps)==0:
           optimizer.step()        # 反向传播，更新网络参数
           optimizer.zero_grad()   # 清空梯度
   ```

   > 参考教程：[GPU 显存不足怎么办？](https://zhuanlan.zhihu.com/p/65002487)

3. test阶段清除梯度：一般train阶段，会从正常图片上截取patch作为输入，而test阶段则会使用整张图片输入，这样导致test阶段可能会爆显存。通过清除test阶段的中间变量，可以节省大量显存，比如不保留梯度

   ```python
   with torch.no_grad():
   ```

   PS：[model.eval()和torch.no_grad()的区别](https://blog.csdn.net/qq_38410428/article/details/101102075)
   
4. 输入不要以一次性都转移到GPU上，用到了再转移

## 估计模型显存

$$
显存占用 = 模型自身参数 * n + batch size * 输出参数量 * 2 + 一个batch的输入数据（往往忽略）
$$

其中，n是根据优化算法来定的，如果选用SGD， 则 n = 2， 如果选择Adam， 则 n = 4

一个很棒的实现如下， 我懒得再重新写了，你可以根据这个改一改，问题不大。

```python
# 模型显存占用监测函数
# model：输入的模型
# input：实际中需要输入的Tensor变量
# type_size 默认为 4 默认类型为 float32 

def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums


    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))
```

> 参考教程：[GPU 显存不足怎么办？](https://zhuanlan.zhihu.com/p/65002487)

## 显存跟踪

使用[Pytorch-Memory-Utils](https://github.com/Oldpan/Pytorch-Memory-Utils)工具

## Debug时将Tensor转为图片

```python
# 假设tensor大小为1*3*720*1280
img=(tensor[0].detach().cpu().numpy().transpose(1, 2, 0)*255.0).astype(np.uint8)
# .detach()用于删除grad
# .cpu()用于将数据从GPU转移到CPU
```

> 参考：[Pytorch中Tensor与各种图像格式的相互转化](https://oldpan.me/archives/pytorch-tensor-image-transform)

## 设置model的模式：train or eval

- `model.eval()`：设置为测试模式，不启用 Batch Normalization 和 Dropout，不会取平均，而是用训练好的值。不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大。

- `model.train()`：设置为训练模式，启用 Batch Normalization 和 Dropout，在训练过程中防止网络过拟合

PS：[model.eval()和torch.no_grad()的区别](https://blog.csdn.net/qq_38410428/article/details/101102075)

## 打印宽度增加

可以不换行显示更多的内容，方便对比矩阵内容

```python
torch.set_printoptions(linewidth=1600)
```

> [SET_PRINTOPTIONS官方文档](https://pytorch.org/docs/stable/generated/torch.set_printoptions.html)

## tensorboard

- 排除指定字符串：`^(?!.*Loss).*$`

  <img src="images/image-20211215144532866.png" alt="image-20211215144532866" style="zoom: 67%;" />

# 各种函数

## torch

- [**`Torch.bmn()`**](https://blog.csdn.net/qq_40178291/article/details/100302375)：计算两个矩阵的矩阵乘法，两个矩阵的维度必须为3。输入维度为 $(b \times n \times m)$和$(b \times m \times p)$，则输出维度为$(b \times n \times p)$

## torch.cuda

- [**`torch.cuda.empty_cache()`**](https://blog.csdn.net/qq_29007291/article/details/90451890)：释放不需要的显存，而且可以在`nvidia-smi`中看到

## torch.Tensor

- **`Tensor.grad_fn`**[^1][^2]：反向传播时，用来计算梯度的函数，即指示梯度函数是哪种类型。Tensor类属性方法。叶子节点通常为None，只有结果节点的`grad_fn`才有效

- **`Tensor.is_leaf`**[^2]：标记该tensor是否为叶子节点。Tensor类变量，布尔值。

  所有requires_grad=False的Tensors都为叶子节点

  所有用户显示初始化的Tensors也为叶子节点

  由各种操作(operation)的结果隐式生成的不是叶子节点，比如`a.cuda()`以及加减乘除操作

  叶节点就是由用户创建的 Variable（一般是输入变量） 或 Parameter（网络参数W和b）[^8]

  叶子节点没有下一个可传播的节点，它就是梯度反向传递在这条分路上的终点[^8]

- **`Tensor.is_contiguous`**：Tensor底层一维数组元素的存储顺序与Tensor按行优先一维展开的元素顺序是否一致。多维矩阵，在内存中都是使用内存的1维数组存储的，语义上连续，在实际存储中不一定连续。

- **`Tensor.contiguous()`**：保证tensor在内存上存储时时连续的，如果本来就连续，则无操作。通常在`Tensor.view()`前面使用

- **`Tensor.expand(*sizes)`**：返回当前张量在某维扩展更大后的张量

    ```python
    a = torch.rand(2,3)
    b = a.expand(6,3)
    ```

- [**`Tensor.repeat(*sizes)`**](https://pytorch-cn.readthedocs.io/zh/latest/package_references/Tensor/#repeatsizes)：沿着指定的维度重复tensor。 不同于`expand()`，该函数复制的是tensor中的数据。参数 *sizes (torch.Size ot int...)表示沿着每一维**重复的次数**

- **`Tensor.view()`**：相当于numpy中的resize()，要求Tensor在内存中连续，如果不连续可以使用函数`Tensor.contiguous()`。例如

    ```python
    a = torch.rand(2,3,4,5)
    b = a.view(3,4,5,-1) # -1表示该维度进行推断，但是只有一个维度能为-1
    # b.size() = [3, 4, 5, 2]
    ```

- **`Tensor.permute(*dims)`**：将tensor的维度换位。
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

- [**`nn.ZeroPad2d(padding)`**](https://www.cnblogs.com/wanghui-garcia/p/11265843.html)：使用0填充输入tensor的边界。参数如下：

  - padding（int, tuple）：指定填充的大小。如果是一个整数值a，则所有边界都使用相同的填充数，等价于输入(a,a,a,a)；如果是大小为4的元组，则表示 (padding_left, padding_right, padding_top, padding_bottom)

## torch.nn.functional

一般`import torch.nn.functional as F`

- **`F.unfold(input, kernel_size)`**：在N\*C*H\*W的H\*W平面上取块，然后堆叠，主要用于CV中的patch提取
- [**`F.grid_sample(input, grid, ...)`**](https://pytorch.org/docs/master/generated/torch.nn.functional.grid_sample.html)：通常用在光流、体素流等地方，基本原理是双线性插值 or 三线性插值。grid是用来告诉。以4维的输入为例，input大小为 $(N,C,H_{in},W_{in})$，grid大小为$(N,H_{out},W_{out},2)$，output大小为$(N,C,H_{out},W_{out})$。grid中每个位置$(H_{out},W_{out})$的2个元素，表示在input上面取值的位置（x、y），然后填到output上对应位置。还可以是5维度的输入，多了一个D维度（猜测是depth）
- [**`F.affine_grid(theta, size, ...)`**](https://pytorch.org/docs/master/generated/torch.nn.functional.affine_grid.html#torch.nn.functional.affine_grid)：

## torch.utils.data

- **`torch.utils.data.Dataloader()`**

  - [`pin_memory`]([Pytorch中多GPU训练指北 - Oldpan的个人博客](https://oldpan.me/archives/pytorch-to-use-multiple-gpus))：锁页内存，创建Data Loader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。

    主机中的内存，有两种存在方式，一是锁页，二是不锁页，锁页内存存放的内容在任何情况下都不会与主机的虚拟内存进行交换（注：虚拟内存就是硬盘），而不锁页内存在主机内存不足时，数据会存放在虚拟内存中。显卡中的显存全部是锁页内存,当计算机的内存充足的时候，可以设置pin_memory=True。当系统卡住，或者交换内存使用过多的时候，设置pin_memory=False。因为pin_memory与电脑硬件性能有关，pytorch开发者不能确保每一个炼丹玩家都有高端设备，因此pin_memory默认为False。

    **总结一句就是，如果机子的内存比较大，建议开启pin_memory=Ture，如果开启后发现有卡顿现象或者内存占用过高，此时建议关闭。**

# NVIDIA DALI

## [Getting  started tutorial](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/getting_started.html)

- 打印结果的时候，需要将`TensorList`对象转换成`NumPy`。但是不是所有的的`TensorList`都能转换，使用`.is_dense_tensor()`查看其能否转换成`NumPy`。labels可以使用`.as_tensor()`直接转成tensor，但是imgs不行，因为其中包含了多张img（数量=batch size），需要遍历图像中的每一张图像（`.at()`）

# 参考资料

[^1]:[Pytorch autograd,backward详解](https://zhuanlan.zhihu.com/p/83172023)
[^2]:[【深度学习理论】一文搞透pytorch中的tensor、autograd、反向传播和计算图](https://zhuanlan.zhihu.com/p/145353262)
[^3]:[PyTorch: 梯度下降及反向传播](https://blog.csdn.net/m0_37306360/article/details/79307354)
[^4]:[Pytorch入门学习（八）-----自定义层的实现（甚至不可导operation的backward写法）](https://blog.csdn.net/Hungrier/article/details/78346304)
[^5]:[Pytorch使用Tensorboard可视化网络结构_sunqiande88的博客-CSDN博客](https://blog.csdn.net/sunqiande88/article/details/80155925)
[^6]:[卷积神经网络(CNN)反向传播算法推导](https://zhuanlan.zhihu.com/p/61898234)

[^7]:[卷积神经网络(CNN)Python的底层实现——以LeNet为例](https://zhuanlan.zhihu.com/p/62303214)
[^8]:[Pytorch 中的反向传播](https://zhuanlan.zhihu.com/p/212748204)
[^9]:[探讨Pytorch中nn.Module与nn.autograd.Function的backward()函数](https://oldpan.me/archives/pytorch-nn-module-functional-backward)
[^10]:[Pytorch： 自定义网络层](https://blog.csdn.net/xholes/article/details/81478670)
[^11]:[Extending `torch.autograd`](https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd)
[^12]: [扩展 torch.autograd](https://pytorch-cn.readthedocs.io/zh/latest/notes/extending/#torchautograd)

[^13]: [pytroch中ctx和self的区别](https://blog.csdn.net/littlehaes/article/details/103828130)

- [Oldpan的个人博客](https://oldpan.me/)
  - [浅谈深度学习:如何计算模型以及中间变量的显存占用大小](https://oldpan.me/archives/how-to-calculate-gpu-memory)
  - [如何在Pytorch中精细化利用显存](https://oldpan.me/archives/how-to-use-memory-pytorch)
  - [再次浅谈Pytorch中的显存利用问题(附完善显存跟踪代码)](https://oldpan.me/archives/pytorch-gpu-memory-usage-track)