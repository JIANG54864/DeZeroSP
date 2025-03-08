# 为支持使用gpu相关代码
import numpy as np
gpu_enable = True
try:
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False
from dezerosp import Variable


def get_array_module(x):
    """
    返回 'x' 的类型。

参数：
x （dezero.变量或 numpy.ndarray 或 cupy.ndarray）：根据其值
确定应使用 NumPy 还是 CuPy。

返回：
module： 'cupy' 或 'numpy' 根据参数返回。
    """
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp


def as_numpy(x):
    """转换为 `numpy.ndarray` 。

参数：
x (`numpy.ndarray` 或 `cupy.ndarray`)：可以转换为 `numpy.ndarray` 的任意对象。
返回值：
`numpy.ndarray`：已转换的数组。
    """
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x): # 是否是标量
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)


def as_cupy(x):
    """转换为 `cupy.ndarray` 。

参数：
x (`numpy.ndarray` 或 `cupy.ndarray`)：可以转换为 `cupy.ndarray` 的任意对象。
返回值：
`cupy.ndarray`：已转换的数组。
    """
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        raise Exception('CuPy cannot be loaded. Install CuPy!')
    return cp.asarray(x)