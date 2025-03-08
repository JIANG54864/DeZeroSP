# DeZeroSP
从零开始构建一个深度学习框架

各模块介绍：
core：核心代码，
1.定义变量（以及子类参数类）和函数的基类；
2.重载常见运算符；
3.临时修改参数的功能

functions：继承函数类，实现诸多具体函数

layers：定义神经网络相关的层，包含Layer父类和诸多具体子类

models：model类在layer层基础上提供了画计算图的功能，继承model类实现具体经典模型

optimizers：更新参数相关功能

datasets：创建、下载数据集相关功能

transforms：数据预处理相关功能

dataloader：批量加载数据相关功能

utils：一些工具类，如绘图工具，数据处理工具等

functions_conv：卷积相关函数

cuda：使用gpu

使用的外部库：numpy, matplotlib,  cupy, Pillow