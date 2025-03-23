# 更新参数相关功能
import math
from dezerosp import cuda, Parameter


# =============================================================================
# 优化器（基类）
# =============================================================================
class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks:
            f(params)

        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)


# =============================================================================
# Hook functions
# =============================================================================
# 增加梯度，反向传播使得权重衰减
class WeightDecay:
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, params):
        for param in params:
            param.grad.data += self.rate * param.data

# 梯度裁剪，防止梯度爆炸
class ClipGrad:
    def __init__(self, max_norm):
        self.max_norm = max_norm

    def __call__(self, params):
        total_norm = 0
        for param in params:
            total_norm += (param.grad.data ** 2).sum()
        total_norm = math.sqrt(float(total_norm))

        rate = self.max_norm / (total_norm + 1e-6)
        if rate < 1:
            for param in params:
                param.grad.data *= rate


class FreezeParam:
    def __init__(self, *layers):
        self.freeze_params = []
        for l in layers:
            if isinstance(l, Parameter):
                self.freeze_params.append(l)
            else:
                for p in l.params():
                    self.freeze_params.append(p)

    def __call__(self, params):
        for p in self.freeze_params:
            p.grad = None



# =============================================================================
# SGD / MomentumSGD / AdaGrad / Adam
# =============================================================================
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            xp = cuda.get_array_module(param.data)
            self.vs[v_key] = xp.zeros_like(param.data)

        v = self.vs[v_key]
        # 在物体不受任何力时，物体逐渐减速，对应物理上的地面摩擦或空气阻力。例如momentum=0.9
        v *= self.momentum
        # 物体在梯度方向上受力，在这个力的作用下，物体的速度增加。（负梯度方向上加速
        v -= self.lr * param.grad.data
        param.data += v


class AdaGrad(Optimizer):
    def __init__(self, lr=0.001, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.hs = {}

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = xp.zeros_like(param.data)

        lr = self.lr
        eps = self.eps
        grad = param.grad.data
        h = self.hs[h_key]

        h += grad * grad
        # 参数的元素中变动较大（梯度较大）的元素的学习率将变小
        param.data -= lr * grad / (xp.sqrt(h) + eps)

class Adam(Optimizer):
    # 融合了Momentum和AdaGrad的方法
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self):
        # 计算一阶矩估计的偏差修正项
        fix1 = 1. - math.pow(self.beta1, self.t)
        # 计算二阶矩估计的偏差修正项
        fix2 = 1. - math.pow(self.beta2, self.t)
        # 确保学习率在训练初期不会过大，同时在后期能够稳定收敛。
        return self.alpha * math.sqrt(fix2) / fix1


    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        # 更新一阶矩估计（momentum的估计值）
        # m += (1 - beta1) * (grad - m)
        m = beta1 * m + (1 - beta1) * grad
        # 更新二阶矩估计（梯度的平方的期望值）
        # v += (1 - beta2) * (grad * grad - v)
        v = beta2 * v + (1 - beta2) * (grad * grad)
        # 类似adagrad的思想，更新参数
        param.data -= self.lr * m / (xp.sqrt(v) + eps)
