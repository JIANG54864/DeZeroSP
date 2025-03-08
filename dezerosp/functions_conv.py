# CNN相关函数
import numpy as np
from dezerosp import cuda
from dezerosp.core import Function, as_variable
from dezerosp.utils import pair, get_conv_outsize, get_deconv_outsize
from dezerosp.functions import linear, broadcast_to


# =============================================================================
# [简化版本] 二维卷积（conv2d_simple） / 池化（pooling_simple）
# =============================================================================
def conv2d_simple(x, W, b=None, stride=1, pad=0):
    """
    简单的2D卷积操作。

    参数:
    x: 输入张量，形状为 (N, C, H, W)，其中 N 是批次大小，C 是输入通道数，H 是高度，W 是宽度。
    W: 卷积核张量，形状为 (OC, C, KH, KW)，其中 OC 是输出通道数，C 是输入通道数，KH 是卷积核高度，KW 是卷积核宽度。
    b: 可选的偏置张量，形状为 (OC,)。如果为 None，则不使用偏置。
    stride: 卷积步幅，可以是整数或元组 (SH, SW)。
    pad: 填充大小，可以是整数或元组 (PH, PW)。

    返回:
    y: 卷积结果张量，形状为 (N, OC, OH, OW)，其中 OH 和 OW 是输出特征图的高度和宽度。
    """
    x, W = as_variable(x), as_variable(W)

    Weight = W
    N, C, H, W = x.shape
    OC, C, KH, KW = Weight.shape
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    # 将输入张量转换为列矩阵形式
    col = im2col(x, (KH, KW), stride, pad, to_matrix=True)
    # 将卷积核重塑为矩阵形式并进行转置
    Weight = Weight.reshape(OC, -1).transpose()
    # 执行线性变换（卷积操作）
    t = linear(col, Weight, b)
    # 将结果重塑为输出张量的形状
    y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)
    return y


def pooling_simple(x, kernel_size, stride=1, pad=0):
    """
    简单的2D池化操作。

    参数:
    x: 输入张量，形状为 (N, C, H, W)，其中 N 是批次大小，C 是输入通道数，H 是高度，W 是宽度。
    kernel_size: 池化核大小，可以是整数或元组 (KH, KW)。
    stride: 池化步幅，可以是整数或元组 (SH, SW)。
    pad: 填充大小，可以是整数或元组 (PH, PW)。

    返回:
    y: 池化结果张量，形状为 (N, C, OH, OW)，其中 OH 和 OW 是输出特征图的高度和宽度。
    """
    x = as_variable(x)

    N, C, H, W = x.shape
    KH, KW = pair(kernel_size)
    PH, PW = pair(pad)
    SH, SW = pair(stride)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    # 将输入张量转换为列矩阵形式
    col = im2col(x, kernel_size, stride, pad, to_matrix=True)
    # 将列矩阵重塑为适合池化操作的形式
    col = col.reshape(-1, KH * KW)
    # 执行最大池化操作
    y = col.max(axis=1)
    # 将结果重塑为输出张量的形状
    y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
    return y


# =============================================================================
#  2维卷积/反卷积
# =============================================================================

class Conv2d(Function):
    """
    2D卷积操作类。

    参数:
    stride: 卷积步幅，可以是整数或元组 (SH, SW)。
    pad: 填充大小，可以是整数或元组 (PH, PW)。
    """
    def __init__(self, stride=1, pad=0):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)

    def forward(self, x, W, b):
        """
        前向传播函数。

        参数:
        x: 输入张量，形状为 (N, C, H, W)。
        W: 卷积核张量，形状为 (OC, C, KH, KW)。
        b: 可选的偏置张量，形状为 (OC,)。

        返回:
        y: 卷积结果张量，形状为 (N, OC, OH, OW)。
        """
        xp = cuda.get_array_module(x)

        KH, KW = W.shape[2:]
        # 把应用卷积核的部分取出来展开为一列
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False)

        # 执行卷积操作
        y = xp.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b
        y = xp.rollaxis(y, 3, 1)
        return y

    def backward(self, gy):
        """
        反向传播函数。

        参数:
        gy: 梯度张量，形状为 (N, OC, OH, OW)。

        返回:
        gx: 输入张量的梯度，形状为 (N, C, H, W)。
        gW: 卷积核的梯度，形状为 (OC, C, KH, KW)。
        gb: 偏置的梯度，形状为 (OC,)。
        """
        x, W, b = self.inputs
        # ==== gx ====
        gx = deconv2d(gy, W, b=None, stride=self.stride, pad=self.pad,
                      outsize=(x.shape[2], x.shape[3]))
        # ==== gW ====
        gW = Conv2DGradW(self)(x, gy)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def conv2d(x, W, b=None, stride=1, pad=0):
    """
    2D卷积操作。

    参数:
    x: 输入张量，形状为 (N, C, H, W)。
    W: 卷积核张量，形状为 (OC, C, KH, KW)。
    b: 可选的偏置张量，形状为 (OC,)。
    stride: 卷积步幅，可以是整数或元组 (SH, SW)。
    pad: 填充大小，可以是整数或元组 (PH, PW)。

    返回:
    y: 卷积结果张量，形状为 (N, OC, OH, OW)。
    """
    return Conv2d(stride, pad)(x, W, b)


class Deconv2d(Function):
    """
    2D反卷积操作类。

    参数:
    stride: 反卷积步幅，可以是整数或元组 (SH, SW)。
    pad: 填充大小，可以是整数或元组 (PH, PW)。
    outsize: 输出特征图的大小，可以是元组 (OH, OW)。
    """
    def __init__(self, stride=1, pad=0, outsize=None):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.outsize = outsize

    def forward(self, x, W, b):
        """
        前向传播函数。

        参数:
        x: 输入张量，形状为 (N, C, H, W)。
        W: 卷积核张量，形状为 (C, OC, KH, KW)。
        b: 可选的偏置张量，形状为 (OC,)。

        返回:
        y: 反卷积结果张量，形状为 (N, OC, OH, OW)。
        """
        xp = cuda.get_array_module(x)

        Weight = W
        SH, SW = self.stride
        PH, PW = self.pad
        C, OC, KH, KW = Weight.shape
        N, C, H, W = x.shape
        if self.outsize is None:
            out_h = get_deconv_outsize(H, KH, SH, PH)
            out_w = get_deconv_outsize(W, KW, SW, PW)
        else:
            out_h, out_w = pair(self.outsize)
        img_shape = (N, OC, out_h, out_w)

        # 执行反卷积操作
        gcol = xp.tensordot(Weight, x, (0, 1))
        gcol = xp.rollaxis(gcol, 3)
        y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad,
                         to_matrix=False)
        if b is not None:
            self.no_bias = True
            y += b.reshape((1, b.size, 1, 1))
        return y

    def backward(self, gy):
        """
        反向传播函数。

        参数:
        gy: 梯度张量，形状为 (N, OC, OH, OW)。

        返回:
        gx: 输入张量的梯度，形状为 (N, C, H, W)。
        gW: 卷积核的梯度，形状为 (C, OC, KH, KW)。
        gb: 偏置的梯度，形状为 (OC,)。
        """
        x, W, b = self.inputs

        # ==== gx ====
        gx = conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
        # ==== gW ====
        f = Conv2DGradW(self)
        gW = f(gy, x)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):
    """
    2D反卷积操作。

    参数:
    x: 输入张量，形状为 (N, C, H, W)。
    W: 卷积核张量，形状为 (C, OC, KH, KW)。
    b: 可选的偏置张量，形状为 (OC,)。
    stride: 反卷积步幅，可以是整数或元组 (SH, SW)。
    pad: 填充大小，可以是整数或元组 (PH, PW)。
    outsize: 输出特征图的大小，可以是元组 (OH, OW)。

    返回:
    y: 反卷积结果张量，形状为 (N, OC, OH, OW)。
    """
    return Deconv2d(stride, pad, outsize)(x, W, b)


class Conv2DGradW(Function):
    """
    卷积核梯度计算类。

    参数:
    conv2d: Conv2d 类的实例，用于获取卷积核大小、步幅和填充信息。
    """
    def __init__(self, conv2d):
        W = conv2d.inputs[1]
        kh, kw = W.shape[2:]
        self.kernel_size = (kh, kw)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x, gy):
        """
        前向传播函数。

        参数:
        x: 输入张量，形状为 (N, C, H, W)。
        gy: 梯度张量，形状为 (N, OC, OH, OW)。

        返回:
        gW: 卷积核的梯度，形状为 (OC, C, KH, KW)。
        """
        xp = cuda.get_array_module(x)

        # 将输入张量转换为列矩阵形式
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        # 计算卷积核的梯度
        gW = xp.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
        return gW

    def backward(self, gys):
        """
        反向传播函数。

        参数:
        gys: 梯度张量，形状为 (OC, C, KH, KW)。

        返回:
        gx: 输入张量的梯度，形状为 (N, C, H, W)。
        ggy: 梯度张量的梯度，形状为 (N, OC, OH, OW)。
        """
        x, gy = self.inputs
        gW, = self.outputs

        xh, xw = x.shape[2:]
        gx = deconv2d(gy, gW, stride=self.stride, pad=self.pad,
                      outsize=(xh, xw))
        ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
        return gx, ggy


# =============================================================================
# 最大池化/平均池化
# =============================================================================


class Pooling(Function):
    """
    实现最大池化操作的前向和反向传播。

    Args:
        kernel_size (int or tuple): 池化核的大小。
        stride (int or tuple, optional): 池化操作的步幅，默认为1。
        pad (int or tuple, optional): 输入的填充大小，默认为0。
    """

    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        """
        前向传播函数，执行最大池化操作。

        Args:
            x (numpy.ndarray or cupy.ndarray): 输入数据。

        Returns:
            numpy.ndarray or cupy.ndarray: 池化后的输出。
        """
        # 将输入数据转换为列形式，以便进行池化操作
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)

        # 获取池化后的输出形状
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)

        # 找到每个池化区域的最大值及其索引
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y

    def backward(self, gy):
        """
        反向传播函数，计算最大池化的梯度。

        Args:
            gy (numpy.ndarray or cupy.ndarray): 上一层的梯度。

        Returns:
            numpy.ndarray or cupy.ndarray: 输入的梯度。
        """
        return Pooling2DGrad(self)(gy)


class Pooling2DGrad(Function):
    """
    实现最大池化操作的反向传播。

    Args:
        mpool2d (Pooling): 最大池化层的实例。
    """

    def __init__(self, mpool2d):
        self.mpool2d = mpool2d
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, gy):
        """
        前向传播函数，计算最大池化的梯度。

        Args:
            gy (numpy.ndarray or cupy.ndarray): 上一层的梯度。

        Returns:
            numpy.ndarray or cupy.ndarray: 输入的梯度。
        """
        xp = cuda.get_array_module(gy)

        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)

        # 初始化梯度矩阵
        gcol = xp.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)

        # 根据最大值的索引填充梯度
        indexes = (self.indexes.ravel()
                   + xp.arange(0, self.indexes.size * KH * KW, KH * KW))

        gcol[indexes] = gy.ravel()
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        gcol = xp.swapaxes(gcol, 2, 4)
        gcol = xp.swapaxes(gcol, 3, 5)

        # 将梯度矩阵转换回输入的形状
        gx = col2im_array(gcol, (N, C, H, W), self.kernel_size, self.stride,
                          self.pad, to_matrix=False)
        return gx

    def backward(self, ggx):
        """
        反向传播函数，计算梯度的梯度。

        Args:
            ggx (numpy.ndarray or cupy.ndarray): 上一层的梯度的梯度。

        Returns:
            numpy.ndarray or cupy.ndarray: 输入的梯度的梯度。
        """
        f = Pooling2DWithIndexes(self.mpool2d)
        return f(ggx)


class Pooling2DWithIndexes(Function):
    """
    实现带有索引的最大池化操作。

    Args:
        mpool2d (Pooling): 最大池化层的实例。
    """

    def __init__(self, mpool2d):
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shpae = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, x):
        """
        前向传播函数，执行带有索引的最大池化操作。

        Args:
            x (numpy.ndarray or cupy.ndarray): 输入数据。

        Returns:
            numpy.ndarray or cupy.ndarray: 池化后的输出。
        """
        # 将输入数据转换为列形式
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)

        # 根据索引选择最大值
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
        indexes = self.indexes.ravel()
        col = col[np.arange(len(indexes)), indexes]
        return col.reshape(N, C, OH, OW)


def pooling(x, kernel_size, stride=1, pad=0):
    """
    执行最大池化操作。

    Args:
        x (numpy.ndarray or cupy.ndarray): 输入数据。
        kernel_size (int or tuple): 池化核的大小。
        stride (int or tuple, optional): 池化操作的步幅，默认为1。
        pad (int or tuple, optional): 输入的填充大小，默认为0。

    Returns:
        numpy.ndarray or cupy.ndarray: 池化后的输出。
    """
    return Pooling(kernel_size, stride, pad)(x)


class AveragePooling(Function):
    """
    实现平均池化操作的前向和反向传播。

    Args:
        kernel_size (int or tuple): 池化核的大小。
        stride (int or tuple, optional): 池化操作的步幅，默认为1。
        pad (int or tuple, optional): 输入的填充大小，默认为0。
    """

    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.input_shape = None

    def forward(self, x):
        """
        前向传播函数，执行平均池化操作。

        Args:
            x (numpy.ndarray or cupy.ndarray): 输入数据。

        Returns:
            numpy.ndarray or cupy.ndarray: 池化后的输出。
        """
        self.input_shape = x.shape
        # 将输入数据转换为列形式，以便进行池化操作
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        # 计算每个池化区域的平均值
        y = col.mean(axis=(2, 3))
        return y

    def backward(self, gy):
        """
        反向传播函数，计算平均池化的梯度。

        Args:
            gy (numpy.ndarray or cupy.ndarray): 上一层的梯度。

        Returns:
            numpy.ndarray or cupy.ndarray: 输入的梯度。
        """
        # 将梯度除以池化核的大小，以计算平均梯度
        N, C, OH, OW = gy.shape
        KW, KH = pair(self.kernel_size)
        gy /= (KW * KH)

        # 将梯度广播到池化核的大小
        gcol = broadcast_to(gy.reshape(-1), (KH, KW, N * C * OH * OW))
        gcol = gcol.reshape(KH, KW, N, C, OH, OW).transpose(2, 3, 0, 1, 4, 5)

        # 将梯度矩阵转换回输入的形状
        gx = col2im(gcol, self.input_shape, self.kernel_size, self.stride,
                    self.pad, to_matrix=False)
        return gx


def average_pooling(x, kernel_size, stride=1, pad=0):
    """
    执行平均池化操作。

    Args:
        x (numpy.ndarray or cupy.ndarray): 输入数据。
        kernel_size (int or tuple): 池化核的大小。
        stride (int or tuple, optional): 池化操作的步幅，默认为1。
        pad (int or tuple, optional): 输入的填充大小，默认为0。

    Returns:
        numpy.ndarray or cupy.ndarray: 池化后的输出。
    """
    return AveragePooling(kernel_size, stride, pad)(x)


# =============================================================================
#  im2col / col2im
# =============================================================================
class Im2col(Function):
    """
    将输入图像转换为列矩阵的类，用于卷积操作的优化。

    参数:
    - kernel_size: 卷积核的大小，是一个元组或整数。
    - stride: 卷积步长，表示卷积窗口的移动步幅。
    - pad: 填充大小，用于在图像边缘添加填充。
    - to_matrix: 是否将转换后的数据展平为矩阵。
    """
    def __init__(self, kernel_size, stride, pad, to_matrix):
        # 初始化父类
        super().__init__()
        # 初始化成员变量
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        """
        前向传播，将输入图像转换为列矩阵。

        参数:
        - x: 输入图像，通常是一个四维张量。

        返回:
        - y: 转换后的列矩阵或其相应的数据结构。
        """
        # 记录输入形状
        self.input_shape = x.shape
        # 调用im2col_array进行图像到列矩阵的转换
        y = im2col_array(x, self.kernel_size, self.stride, self.pad,
                         self.to_matrix)
        return y

    def backward(self, gy):
        """
        反向传播，将梯度从列矩阵转换回图像格式。

        参数:
        - gy: 损失函数关于输出的梯度。

        返回:
        - gx: 损失函数关于输入图像的梯度。
        """
        # 调用col2im将梯度转换回图像格式
        gx = col2im(gy, self.input_shape, self.kernel_size, self.stride,
                    self.pad, self.to_matrix)
        return gx


def im2col(x, kernel_size, stride=1, pad=0, to_matrix=True):
    """根据滤波器从图像中提取块。
参数：
    x (`dezerosp.Variable` 或 `ndarray`): 形状为 `(N, C, H, W)` 的输入变量。
    kernel_size (int 或 (int, int)): 滤波器的大小。
    stride (int 或 (int, int)): 滤波器的步幅。
    pad (int 或 (int, int)): 输入数组的空间填充宽度。
    to_matrix (bool): 如果为 True，`col` 将被重塑为形状为 `(N*OH*OW, C*KH*KW)` 的二维数组。
返回：
    `dezerosp.Variable`: 输出变量。如果 `to_matrix` 为 False，输出形状为 `(N, C, KH, KW, OH, OW)`，否则为 `(N*OH*OW, C*KH*KW)`。

符号说明：
- `N` 是批量大小。
- `C` 是输入通道的数量。
- `H` 和 `W` 分别是输入图像的高度和宽度。
- `KH` 和 `KW` 分别是滤波器的高度和宽度。
- `SH` 和 `SW` 是滤波器的步幅。
- `PH` 和 `PW` 是空间填充的大小。
- `OH` 和 `OW` 分别是输出的高度和宽度。
    """
    y = Im2col(kernel_size, stride, pad, to_matrix)(x)
    return y


class Col2im(Function):
    """
    Col2im类用于将展开的矩阵转换回图像格式。

    参数:
    - input_shape: 输入图像的形状。
    - kernel_size: 卷积核的大小。
    - stride: 步幅大小。
    - pad: 填充大小。
    - to_matrix: 是否转换为矩阵形式。
    """
    def __init__(self, input_shape, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        """
        前向传播函数，将输入的矩阵转换回图像格式。

        参数:
        - x: 输入矩阵。

        返回:
        - y: 转换后的图像。
        """
        y = col2im_array(x, self.input_shape, self.kernel_size, self.stride,
                         self.pad, self.to_matrix)
        return y

    def backward(self, gy):
        """
        反向传播函数，计算输入矩阵的梯度。

        参数:
        - gy: 损失函数相对于输出的梯度。

        返回:
        - gx: 损失函数相对于输入矩阵的梯度。
        """
        gx = im2col(gy, self.kernel_size, self.stride, self.pad,
                    self.to_matrix)
        return gx


def col2im(x, input_shape, kernel_size, stride=1, pad=0, to_matrix=True):
    """
    将展开的矩阵转换回图像格式的函数。

    参数:
    - x: 输入矩阵。
    - input_shape: 输入图像的形状。
    - kernel_size: 卷积核的大小。
    - stride: 步幅大小。
    - pad: 填充大小。
    - to_matrix: 是否转换为矩阵形式。

    返回:
    - 转换后的图像。
    """
    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)


# =============================================================================
#  numpy im2col
# =============================================================================
def im2col_array(img, kernel_size, stride, pad, to_matrix=True):
    """
    将图像转换为展开的矩阵格式。

    参数:
    - img: 输入图像。
    - kernel_size: 卷积核的大小。
    - stride: 步幅大小。
    - pad: 填充大小。
    - to_matrix: 是否转换为矩阵形式。

    返回:
    - 转换后的矩阵。
    """
    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    xp = cuda.get_array_module(img)
    if xp != np:
        col = _im2col_gpu(img, kernel_size, stride, pad)
    else:
        img = np.pad(img,
                     ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
                     mode='constant', constant_values=(0,))
        col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col


def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    """
    将展开的矩阵转换回图像格式。

    参数:
    - col: 输入矩阵。
    - img_shape: 输入图像的形状。
    - kernel_size: 卷积核的大小。
    - stride: 步幅大小。
    - pad: 填充大小。
    - to_matrix: 是否转换为矩阵形式。

    返回:
    - 转换后的图像。
    """
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    xp = cuda.get_array_module(col)
    if xp != np:
        img = _col2im_gpu(col, SH, SW, PH, PW, H, W)
        return img
    else:
        img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1),
                       dtype=col.dtype)
        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
        return img[:, :, PH:H + PH, PW:W + PW]



def _im2col_gpu(img, kernel_size, stride, pad):
    """用于 GPU 的 im2col 函数。
    """
    n, c, h, w = img.shape
    kh, kw = pair(kernel_size)
    sy, sx = pair(stride)
    ph, pw = pair(pad)
    out_h = get_conv_outsize(h, kh, sy, ph)
    out_w = get_conv_outsize(w, kw, sx, pw)
    dy, dx = 1, 1
    col = cuda.cupy.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)
    '''定义一个CUDA ElementwiseKernel，用于实现im2col操作。
    参数说明：
      img: 输入图像数据，类型为raw T，表示原始数据指针。
      h: 输入图像的高度。
      w: 输入图像的宽度。
      out_h: 输出特征图的高度。
      out_w: 输出特征图的宽度。
      kh: 卷积核的高度。
      kw: 卷积核的宽度。
      sy: 卷积在高度方向上的步幅。
      sx: 卷积在宽度方向上的步幅。
      ph: 在高度方向上的填充大小。
      pw: 在宽度方向上的填充大小。
      dy: 卷积核在高度方向上的扩张步幅。
      dx: 卷积核在宽度方向上的扩张步幅。
    返回值：
      col: 输出的列矩阵，类型为T。'''
    cuda.cupy.ElementwiseKernel(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T col',
        '''
           // 计算当前线程对应的通道索引c0
           int c0 = i / (kh * kw * out_h * out_w);
           
           // 计算当前线程对应的卷积核高度索引ky
           int ky = i / (kw * out_h * out_w) % kh;
           
           // 计算当前线程对应的卷积核宽度索引kx
           int kx = i / (out_h * out_w) % kw;
           
           // 计算当前线程对应的输出特征图高度索引out_y
           int out_y = i / out_w % out_h;
           
           // 计算当前线程对应的输出特征图宽度索引out_x
           int out_x = i % out_w;
           
           // 根据卷积核位置和步幅计算输入图像的对应位置in_y和in_x
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           
           // 检查计算出的输入图像位置是否在有效范围内
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             // 如果有效，将输入图像对应位置的值赋给col
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             // 如果无效，将col置为0
             col = 0;
           }
        ''',
        'im2col')(img.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)


    return col


def _col2im_gpu(col, sy, sx, ph, pw, h, w):
    """用于 GPU 的 col2im 函数。
    """
    n, c, kh, kw, out_h, out_w = col.shape
    dx, dy = 1, 1
    img = cuda.cupy.empty((n, c, h, w), dtype=col.dtype)

    cuda.cupy.ElementwiseKernel(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img