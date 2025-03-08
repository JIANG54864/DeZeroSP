# 其它工具类函数，例如画计算图，梯度调整，文件下载等
import os
import subprocess
import urllib.request
import numpy as np
from dezerosp import as_variable
from dezerosp import Variable
from dezerosp import cuda


# =============================================================================
# 可视化计算图
# =============================================================================
def _dot_var(v, verbose=False):
    # 生成变量的Graphviz DOT语言表示
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)

    return dot_var.format(id(v), name)


def _dot_func(f):
    # 生成表示计算图的字符串
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    ret = dot_func.format(id(f), f.__class__.__name__)

    # for edge
    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        ret += dot_edge.format(id(x), id(f))
    for y in f.outputs:  # y is weakref
        ret += dot_edge.format(id(f), id(y()))
    return ret


def get_dot_graph(output, verbose=True):
    """生成计算图的 Graphviz DOT 文本。

构建一个从输出反向可达的函数和变量的图。要可视化 graphviz DOT 文本，您需要 graphviz 包中的 dot 二进制文件（www.graphviz.org）。

参数：
output（dezerosp.Variable）：用于构建图的输出变量。
verbose（bool）：如果为 True，则生成的 dot 图包含形状和数据类型等附加信息。

返回值：
str：由从输出节点反向可达的节点和边组成的 Graphviz DOT 文本。
    """
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + '}'


def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezerosp')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]  # splitext将文件路径 to_file 分割成两部分：文件名和扩展名
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)



# =============================================================================
# NumPy 的实用函数
# =============================================================================
def sum_to(x, shape):
    """沿指定轴对元素求和，以输出给定形状的数组。

参数：
x (ndarray)：输入数组。
形状：

返回值：
ndarray：形状为 的输出数组。
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """调整梯度的形状以适应dezero.functions.sum的反向传播。

参数:
    gy (dezerosp.Variable): 由输出通过反向传播得到的梯度变量。
    x_shape (tuple): 在sum函数前向传播时使用的形状。
    axis (None 或 int 或 ints的元组): 在sum函数前向传播时使用的轴。
    keepdims (bool): 在sum函数前向传播时是否保持维度。

返回:
    dezerosp.Variable: 形状被适当调整后的梯度变量。
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy


def logsumexp(x, axis=1):
    # 实现数值稳定的计算，防止指数运算溢出
    xp = cuda.get_array_module(x)
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    xp.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    xp.log(s, out=s)
    m += s
    return m


def max_backward_shape(x, axis):
    # 将指定轴的尺寸置为1，其余保持不变。
    if axis is None:
        axis = range(x.ndim)
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = axis

    shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
    return shape


# =============================================================================
# 测试梯度
# =============================================================================
def gradient_check(f, x, *args, rtol=1e-4, atol=1e-5, **kwargs):
    """测试给定函数的反向传播过程。
该函数会自动检查给定函数的反向传播过程。为了验证其正确性，函数会比较通过反向传播计算的梯度与通过数值微分计算的梯度。如果结果在容差范围内，则返回 True，否则返回 False。
参数：
    f (可调用对象): 一个接受 `Variable` 并返回 `Variable` 的函数。
    x (`ndarray` 或 `dezerosp.Variable`): 用于计算梯度的目标 `Variable`。
    *args: 如果 `f` 需要除 `x` 以外的其他变量，可以通过此参数指定。
    rtol (float): 相对容差参数。
    atol (float): 绝对容差参数。
    **kwargs: 如果 `f` 需要关键字变量，可以通过此参数指定。
返回：
    bool: 如果结果在容差范围内，则返回 True，否则返回 False。
    """
    x = as_variable(x)
    x.data = x.data.astype(np.float64)

    num_grad = numerical_grad(f, x, *args, **kwargs)
    y = f(x, *args, **kwargs)
    y.backward()
    bp_grad = x.grad.data

    assert bp_grad.shape == num_grad.shape
    res = array_allclose(num_grad, bp_grad, atol=atol, rtol=rtol)

    if not res:
        print('')
        print('========== FAILED (Gradient Check) ==========')
        print('Numerical Grad')
        print(' shape: {}'.format(num_grad.shape))
        val = str(num_grad.flatten()[:10])
        print(' values: {} ...'.format(val[1:-1]))
        print('Backprop Grad')
        print(' shape: {}'.format(bp_grad.shape))
        val = str(bp_grad.flatten()[:10])
        print(' values: {} ...'.format(val[1:-1]))
    return res


def numerical_grad(f, x, *args, **kwargs):
    """通过有限差分法计算数值梯度。
参数：
    f (可调用对象): 一个接受 `Variable` 并返回 `Variable` 的函数。
    x (`ndarray` 或 `dezerosp.Variable`): 用于计算梯度的目标 `Variable`。
    *args: 如果 `f` 需要除 `x` 以外的其他变量，可以通过此参数指定。
    **kwargs: 如果 `f` 需要关键字变量，可以通过此参数指定。
返回：
    `ndarray`: 梯度。
    """
    eps = 1e-4

    x = x.data if isinstance(x, Variable) else x
    xp = cuda.get_array_module(x)
    if xp is not np:
        np_x = cuda.as_numpy(x)
    else:
        np_x = x
    grad = xp.zeros_like(x)

    it = np.nditer(np_x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx].copy()

        x[idx] = tmp_val + eps
        y1 = f(x, *args, **kwargs)  # f(x+h)
        if isinstance(y1, Variable):
            y1 = y1.data
        y1 = y1.copy()

        x[idx] = tmp_val - eps
        y2 = f(x, *args, **kwargs)  # f(x-h)
        if isinstance(y2, Variable):
            y2 = y2.data
        y2 = y2.copy()

        diff = (y1 - y2).sum()
        grad[idx] = diff / (2 * eps)

        x[idx] = tmp_val
        it.iternext()
    return grad


def array_equal(a, b):
    """如果两个数组具有相同的形状和元素，则返回 True，否则返回 False。
参数：
    a, b (numpy.ndarray 或 cupy.ndarray 或 dezerosp.Variable): 要比较的输入数组。
返回：
    bool: 如果两个数组相等，则返回 True。
    """
    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b
    a, b = cuda.as_numpy(a), cuda.as_numpy(b)
    return np.array_equal(a, b)


def array_allclose(a, b, rtol=1e-4, atol=1e-5):
    """如果两个数组（或变量）在容差范围内逐元素相等，则返回 True。
参数：
    a, b (numpy.ndarray 或 cupy.ndarray 或 dezerosp.Variable): 要比较的输入数组。
    rtol (float): 相对容差参数。
    atol (float): 绝对容差参数。
返回：
    bool: 如果两个数组在给定的容差范围内相等，则返回 True，否则返回 False。
    """
    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b
    a, b = cuda.as_numpy(a), cuda.as_numpy(b)
    return np.allclose(a, b, atol=atol, rtol=rtol)


# =============================================================================
# 下载
# =============================================================================
def show_progress(block_num, block_size, total_size):
    # 进度条
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0: p = 100.0
    if i >= 30: i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end='')


cache_dir = os.path.join(os.getcwd(), '.dezerosp')


def get_file(url, file_name=None):
    """如果缓存中没有，则从 `url` 下载文件。
`url` 处的文件会被下载到 `~/.dezerosp` 目录。
参数：
    url (str): 文件的 URL 地址。
    file_name (str): 文件名。如果指定为 `None`，则使用原始文件名。
返回：
    str: 保存文件的绝对路径。
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path


# =============================================================================
# 其它
# =============================================================================
def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2 * p


def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1


def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError