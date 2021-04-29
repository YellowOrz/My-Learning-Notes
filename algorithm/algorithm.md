[TOC]

# 别人的教程

- **卡尔曼滤波（Kalman）**：
    - 视频：[【图灵鸡】什么是卡尔曼滤波？](https://www.bilibili.com/video/BV15E411v7rD)、[卡尔曼滤波器的原理以及在matlab中的实现](https://www.bilibili.com/video/BV1vs411z7PX)
    - 文字：[傻瓜也能懂的卡尔曼滤波器（翻译自外网博客）](https://zhuanlan.zhihu.com/p/64539108)
- **马氏距离（马哈拉诺比斯距离，Mahalanobis Distance）**：[维基百科英语](https://en.wikipedia.org/wiki/Mahalanobis_distance)
- **Marching Cubes**：[（中英字幕）Coding Adventure: Marching Cubes](https://www.bilibili.com/video/BV1Ta4y1E7CT)

# 滤波

## 双边滤波

> [bilateral filter双边滤波器的通俗理解](https://blog.csdn.net/guyuealian/article/details/82660826)

一个与空间距离相关的高斯函数与一个灰度距离相关的高斯函数相乘

空间距离：指的是当前点与中心点的欧式距离，用于衡量空间临近程度

$$\mathrm{e}^{-\frac{\left(x_{i}-x_{c}\right)^{2}+\left(y_{i}-y_{c}\right)^{2}}{2 \sigma^{2}}}$$

灰度距离：指的是当前点灰度与中心点灰度的差的绝对值

$$\mathrm{e}^{-\frac{\left(\operatorname{gray}\left(x_{i}, y_{i}\right)-\operatorname{gray}\left(x_{c}, y_{c}\right)\right)^{2}}{2 \sigma^{2}}}$$

## 引导滤波（guided filter）

> [【拜小白opencv】33-平滑处理6——引导滤波/导向滤波（Guided Filter）](https://blog.csdn.net/sinat_36264666/article/details/77990790)

与双边滤波最大的相似之处，就是同样具有保持边缘特性

在引导滤波的定义中，用到了局部线性模型：$q_{i}=a_{k} I_{i}+b_{k}, \forall i \in \omega_{k}$。$\omega_{k}$为窗口

<img src="images/20170916100836138" alt="img" style="zoom:80%;" />

让拟合函数的输出值 q与真实值 p 之间的差距最小，得到
$$
\begin{array}{l}
a_{k}=\frac{\frac{1}{|\omega|} \sum_{i \in \omega_{k}} I_{i} p_{i}-\mu_{k} \bar{p}_{k}}{\sigma_{k}^{2}+\epsilon} \\
b_{k}=\bar{p}_{k}-a_{k} \mu_{k}
\end{array}
$$
由于一个像素会被多个窗口包含，取平均即可
$$
q_{i} =\frac{1}{|\omega|} \sum_{k: i \in \omega_{k}}\left(a_{k} I_{i}+b_{k}\right)=\bar{a}_{i} I_{i}+\bar{b}_{i}
$$

## 盒式滤波（box filter）

> [盒式滤波器Box Filter](https://www.cnblogs.com/lwl2015/p/4460711.html)

主要功能：在给定的滑动窗口大小下，**对每个窗口内的像素值进行快速相加求和**

Boxfilter的原理有点类似Integral Image，而且比它还要快，但是实现步骤比较复杂

均值滤波 是 盒式滤波 归一化后的特殊情况