# 采样一致性初始对齐算法（Sample Consensus Initial Alignment, SCA-IA）

SCA-IA也称为粗匹配。
点云配准一般先粗匹配，给出一个初始预估的刚性（Rigid）变换矩阵，为精匹配（比如ICP）提供初始配准状态。
粗配准流程图如下



```flow
st=>start: 开始
ed=>end: 结束
op1=>operation: 点云降采样
op2=>operation: 提取法向量
op3=>operation: 提取特征(FPFH)
op4=>operation: 根据特征的距离确定点对关系
op5=>operation: 随机选择几个点对计算刚性变换矩阵
cond=>condition: 是否到达迭代次数

st(right)->op1(right)->op2(right)->op3->op4(bottom)->op5->cond
cond(yes,left)->ed
cond(no,left)->op5
```



> 参考资料：[点云配准，采样一致性初始配准算法，SCA-IA第一篇](https://zhuanlan.zhihu.com/p/66019029)、[点云的粗配准和精配准](https://blog.csdn.net/coldplayplay/article/details/78509541)