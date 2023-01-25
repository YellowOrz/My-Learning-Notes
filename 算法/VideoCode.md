# H.265/HEVC

- 整体流程图

  <img src="images/image-20211205202229840.png" alt="image-20211205202229840" style="zoom: 50%;" />

## 预测编码

- 基本过程

  <img src="images/image-20211205202333386.png" alt="image-20211205202333386" style="zoom:50%;" />

- 编码单元划分：CTU（Coding Tree Uint）64*64

  <img src="images/image-20211205202429366.png" alt="image-20211205202429366" style="zoom: 50%;" />

- 预测单元划分：

  <img src="images/image-20211205202525078.png" alt="image-20211205202525078" style="zoom: 50%;" />

- 帧内预测

  - 模式（共35种）：

    - PLANAR模式（模式编号0）：正左边、正上面、左下角、右上角四个像素的加权求和
    - DC模式（模式编号1）：正左边、正上面的平均，边缘还会有额外加权以保证平滑
    - 33种角度模式（模式编号2~34）

  - 步骤：①相邻参考像素的获取；②参考像素的平滑滤波；③预测像素的计算

    <img src="images/image-20211205203732171.png" alt="image-20211205203732171" style="zoom:50%;" />

- 帧间预测

  - 运动矢量MV（motion vector）
  - 步骤：<img src="images/image-20211205204510278.png" alt="image-20211205204510278" style="zoom:50%;" />