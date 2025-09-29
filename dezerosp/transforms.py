# 数据变换相关代码
import numpy as np
from PIL import Image
from dezerosp.utils import pair


class Compose:
    """组合多个变换。
参数：
    transforms (list): 变换的列表。
    """
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)
        return img

# =============================================================================
# 用于 NumPy 数组的变换
# =============================================================================
class Normalize:
    """使用均值和标准差对 NumPy 数组进行标准化。
参数：
    mean (float 或 sequence): 所有值的均值，或者是每个通道的均值序列。
    std (float 或 sequence): 所有值的标准差，或者是每个通道的标准差序列。
    """
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        mean, std = self.mean, self.std

        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(self.mean) == 1 else len(self.mean)
            mean = np.array(self.mean, dtype=array.dtype).reshape(*mshape)
        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(self.std) == 1 else len(self.std)
            std = np.array(self.std, dtype=array.dtype).reshape(*rshape)
        return (array - mean) / std


class Flatten:
    """将 NumPy 数组展平。
    """
    def __call__(self, array):
        return array.flatten()


class AsType:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.astype(self.dtype)


ToFloat = AsType


class ToInt(AsType):
    def __init__(self, dtype=int):
        self.dtype = dtype

# =============================================================================
# 用于 PIL 图像的变换
# =============================================================================
class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, img):
        if self.mode == 'BGR':
            img = img.convert('RGB')
            r, g, b = img.split()
            img = Image.merge('RGB', (b, g, r))
            return img
        else:
            return img.convert(self.mode)


class Resize:
    """将输入的 PIL 图像调整为指定的大小。
参数：
    size (int 或 (int, int)): 期望的输出大小。
    mode (int): 期望的插值方式。
    """
    def __init__(self, size, mode=Image.BILINEAR):
        self.size = pair(size)
        self.mode = mode

    def __call__(self, img):
        return img.resize(self.size, self.mode)


class CenterCrop:
    """
    实现了一个图像中心裁剪的功能。首先获取图像的原始宽度和高度（W, H）以及目标裁剪尺寸（OW, OH），
    然后计算裁剪区域的左、右、上、下边界，最后调用`img.crop`方法进行裁剪并返回结果。
    """
    def __init__(self, size):
        self.size = pair(size)

    def __call__(self, img):
        W, H = img.size
        OW, OH = self.size
        left = (W - OW) // 2
        right = W - ((W - OW) // 2 + (W - OW) % 2)
        up = (H - OH) // 2
        bottom = H - ((H - OH) // 2 + (H - OH) % 2)
        return img.crop((left, up, right, bottom))


class ToArray:
    """将 PIL 图像转换为 NumPy 数组。"""
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return img
        if isinstance(img, Image.Image):
            img = np.asarray(img)
            img = img.transpose(2, 0, 1)
            img = img.astype(self.dtype)
            return img
        else:
            raise TypeError


class ToPIL:
    """将 NumPy 数组转换为 PIL 图像。."""
    def __call__(self, array):
        data = array.transpose(1, 2, 0)
        return Image.fromarray(data)


class RandomHorizontalFlip:
    pass