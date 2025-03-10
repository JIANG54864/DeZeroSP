# 核心代码，1.定义变量（以及子类参数类）和函数的基类；2.重载常见运算符；3.临时修改参数的功能

import weakref
import numpy as np
import contextlib
import dezerosp
import heapq

# =============================================================================
# Config
# =============================================================================
class Config:
    enable_backprop = True
    train = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


def test_mode():
    return using_config('train', False)

# =============================================================================
# Variable / Function
# =============================================================================
try:
    import cupy
    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = (np.ndarray)


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def unchain(self):
        self.creator = None

    def cleargrad(self):
        self.grad = None

    # backward函数实现了反向传播算法，用于计算梯度。
    # 参数:
    #   retain_grad (bool): 如果为True，则保留中间变量的梯度；否则在计算完成后清空梯度。
    #   create_graph (bool): 如果为True，则在反向传播过程中创建计算图，用于高阶导数计算。
    # 返回值:
    #   无返回值。该函数通过修改Variable对象的grad属性来存储梯度。
    def backward_sample(self, retain_grad=False, create_graph=False):
        # 如果当前变量的梯度尚未初始化，则根据数据形状创建一个全1的梯度张量。
        if self.grad is None:
            xp = dezerosp.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        funcs = []
        seen_set = set()

        # 定义辅助函数add_func，用于将函数添加到待处理列表中，并确保按generation排序。
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        # 将当前变量的creator（即生成该变量的函数）添加到待处理列表中。
        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            # 获取当前函数输出变量的梯度。
            gys = [output().grad for output in f.outputs]  # output是弱引用

            # 在指定配置下执行反向传播计算。
            with using_config('enable_backprop', create_graph):
                # 调用函数的backward方法计算输入变量的梯度。
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                # 将计算得到的梯度累加到输入变量的grad属性中。
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    # 如果输入变量有creator，则将其添加到待处理列表中。
                    if x.creator is not None:
                        add_func(x.creator)

            # 如果不需要保留梯度，则清空当前函数输出变量的梯度。
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y是弱引用



    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            xp = dezerosp.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        # 使用优先队列存储待处理函数，按generation排序
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                # 将函数及其generation作为元组加入优先队列
                heapq.heappush(funcs, (-f.generation, f))  # 负号确保按最大值优先
                seen_set.add(f)

        add_func(self.creator)
        while funcs:
            _, f = heapq.heappop(funcs)  # 取出generation最大的函数
            gys = [output().grad for output in f.outputs]  # 获取输出变量的梯度

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None


    def unchain_backward(self):
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezerosp.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return dezerosp.functions.transpose(self, axes)

    @property
    def T(self):
        return dezerosp.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return dezerosp.functions.sum(self, axis, keepdims)

    def to_cpu(self):
        if self.data is not None:
            self.data = dezerosp.cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = dezerosp.cuda.as_cupy(self.data)


class Parameter(Variable):
    pass


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def __lt__(self, other):
        # 补充辈分比较逻辑，以使得可以使用优先队列
        return self.generation < other.generation

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


# =============================================================================
# 算术运算 / 运算符重载
# =============================================================================
class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:  # for broadcaset
            gx0 = dezerosp.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezerosp.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1



def add(x0, x1):
    x1 = as_array(x1, dezerosp.cuda.get_array_module(x0.data))
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:  # for broadcast
            gx0 = dezerosp.functions.sum_to(gx0, x0.shape)
            gx1 = dezerosp.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0, x1):
    x1 = as_array(x1, dezerosp.cuda.get_array_module(x0.data))
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = dezerosp.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezerosp.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1):
    x1 = as_array(x1, dezerosp.cuda.get_array_module(x0.data))
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1, dezerosp.cuda.get_array_module(x0.data))
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:  # for broadcast
            gx0 = dezerosp.functions.sum_to(gx0, x0.shape)
            gx1 = dezerosp.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1, dezerosp.cuda.get_array_module(x0.data))
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1, dezerosp.cuda.get_array_module(x0.data))
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    # 运算符重载，使得可以直接用自然的符号
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = dezerosp.functions.get_item

    Variable.matmul = dezerosp.functions.matmul
    Variable.dot = dezerosp.functions.matmul
    Variable.max = dezerosp.functions.max
    Variable.min = dezerosp.functions.min
