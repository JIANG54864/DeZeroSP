'''
在这个文件中，展示用此框架进行复杂函数的反向传播计算梯度，以及计算高阶导数，
并画出计算图，注意画图需要安装graphviz
'''
import numpy as np
from dezerosp import Variable
from dezerosp.utils import plot_dot_graph
import dezerosp.functions as F



def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z


x1 = Variable(np.array(1.0))
y1 = Variable(np.array(1.0))
z1 = goldstein(x1, y1)
z1.backward()
print('x1.grad:',x1.grad, 'x2.grad:',y1.grad)
x1.name = 'x1'
y1.name = 'y1'
z1.name = 'z1'
plot_dot_graph(z1, verbose=False, to_file='goldstein.png')


# 计算自然对数的7阶导数在x=2处的取值
x2 = Variable(np.array(2.0))
y2 = F.log(x2)
x2.name = 'x2'
y2.name = 'y2'
y2.backward(create_graph=True)

iters = 6

for i in range(iters):
    gx = x2.grad
    print(i+1,'阶导数：',gx)
    x2.cleargrad()
    gx.backward(create_graph=True)

gx = x2.grad
print(i+2,'阶导数：',gx)
gx.name = 'gx' + str(iters + 1)
plot_dot_graph(gx, verbose=False, to_file='log.png')