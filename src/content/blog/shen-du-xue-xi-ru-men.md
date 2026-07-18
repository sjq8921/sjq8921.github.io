---
title: 深度学习入门：基于Python的理论与实现
description: 《深度学习入门：基于Python的理论与实现》（斋藤康毅著）是一本广受好评的深度学习入门书。本文用纯 NumPy 从零实现激活函数、全连接网络、反向传播和 CNN，并附完整可运行代码。
pubDate: 2026-07-18
tags: [深度学习, Python, NumPy, 读书笔记]
---

《深度学习入门：基于Python的理论与实现》（原书名：*ゼロから作るDeep Learning — Pythonで学ぶディープラーニングの理論と実装*）是日本作者斋藤康毅撰写的一本经典教材。全书最大的特色是 **"从零开始制造"**——不依赖 TensorFlow 或 PyTorch，只靠 NumPy 手写所有组件，从感知机一路搭到 CNN，对每个公式和每行代码都追根究底。

## 为什么推荐这本书

- **零框架依赖**：所有模型用纯 NumPy 实现，绝无黑盒
- **逐层递进**：感知机 → 全连接 → 反向传播 → CNN，每一章都建立在前一章之上
- **代码即教材**：理论推导与代码实现一一对应，读代码等于读数学

## 内容概览

| 章节 | 主题 | 核心内容 |
|------|------|----------|
| 1 | Python 入门 | NumPy 基础与绘图 |
| 2 | 感知机 | 与/或/与非门, XOR 的局限性 |
| 3 | 神经网络 | 激活函数、三层前向网络、softmax 输出 |
| 4 | 神经网络的学习 | 损失函数、数值梯度、梯度下降、mini-batch |
| 5 | 误差反向传播 | 计算图、链式法则、各层的 backward 实现 |
| 6 | 学习技巧 | 优化器、权重初始化、Dropout、BN |
| 7 | 卷积神经网络 | im2col、卷积层、池化层、完整 CNN |
| 8 | 深度学习 | 更深的网络与更大规模数据 |

## 适合谁读

- 有 Python 基础、想真正理解神经网络内部机制的同学
- 用框架调参调腻了、想搞清楚底层到底在算什么的工程师
- 面试前需要把反向传播和 CNN 的细节过一遍的求职者

---

# 从零实现：完整代码走读

以下代码均取自本书配套 Jupyter Notebook，按章节组织，可以直接运行。

## 1. 激活函数

激活函数决定了神经元的"发放"方式。下面用 NumPy 实现三种最常用的激活函数并可视化。

```python
import numpy as np
import matplotlib.pylab as plt
```

### 阶跃函数

二值输出，只在越过阈值时翻转——感知机的核心，但不可导。

```python
def step_function(x):
    return np.array(x > 0, dtype=int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
```

![阶跃函数](/assets/blog/shen-du-xue-xi-ru-men/output_2_0.png)

### Sigmoid 函数

平滑、可导，将输入压缩到 (0, 1)——早期神经网络的标配。

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
```

![Sigmoid 函数](/assets/blog/shen-du-xue-xi-ru-men/output_4_0.png)

### ReLU 函数

现代深度网络的默认选择——简单、高效、缓解梯度消失。

```python
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1.0, 5.0)
plt.show()
```

![ReLU 函数](/assets/blog/shen-du-xue-xi-ru-men/output_6_0.png)

## 2. 输出层

### 恒等函数

回归任务直接用。

```python
def identity_function(x):
    return x
```

### Softmax 函数

分类任务的标配输出层，将 logits 转为概率分布。注意用 `c = np.max(a)` 做数值稳定处理，防止 `exp` 溢出。

```python
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
```

## 3. 三层神经网络的完整前向传播

下面这段代码可以说是一本书的缩影——初始化权重 → sigmoid 激活 → 逐层前向 → softmax 输出，一个 2 输入、3 隐藏、2 输出的网络就搭建完成了。

```python
def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)  # [0.31682708 0.69627909]
```

## 4. 损失函数

### 均方误差（MSE）

回归任务的经典损失，衡量预测值与真实值之间欧氏距离的平方。

```python
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)
```

### 交叉熵误差

分类任务的标准损失，对数似然的负值。`delta = 1e-7` 防止 `log(0)`。

```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size
```

## 5. 梯度与梯度下降

### 数值梯度（中心差分）

解析求导太麻烦时，用数值方法近似。中心差分比前向差分误差更小（O(h²) vs O(h)）。

```python
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad
```

### 梯度下降法

沿着负梯度方向走一小步（学习率 `lr`），迭代 `step_num` 次。

```python
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x
```

## 6. 两层神经网络 + Mini-batch 训练

把前述所有组件拼在一起：权重初始化、前向预测、损失计算、精度评估、数值梯度、mini-batch SGD。这就是一个完整的、可训练的分类器。

```python
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
```

**Mini-batch 训练循环**——从 MNIST 中每次随机抽取 100 张图像，计算梯度、更新参数，周期性地输出准确率。

```python
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
iters_num = 10000            # 可调大以充分训练
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    batch_mask = np.random.choice(x_train.shape[0], batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % 1000 == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        print(f"[iter {i}] train acc: {train_acc:.3f}, "
              f"test acc: {test_acc:.3f}")
```

## 7. 误差反向传播——逐层实现

数值梯度虽然直观，但计算代价太高。反向传播是深度学习效率的基石。下面用 **计算图** 的思想，为每种运算分别实现 `forward` 和 `backward`。

### 乘法层 & 加法层

计算图中最基础的两种节点：

```python
class MulLayer:
    def __init__(self):
        self.x, self.y = None, None

    def forward(self, x, y):
        self.x, self.y = x, y
        return x * y

    def backward(self, dout):
        return dout * self.y, dout * self.x


class AddLayer:
    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout
```

### ReLU 层

正向屏蔽负值，反向时对应位置梯度为零。

```python
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout
```

### Sigmoid 层

Sigmoid 的反向传播有一个优雅的结论：`dx = dout * y * (1 - y)`，其中 `y` 就是 forward 的输出。

```python
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out) * self.out
```

### Affine 层

全连接层的矩阵运算版本。正向是 `x·W + b`，反向需要两个关键的矩阵导数：`dx = dout · Wᵀ`、`dW = xᵀ · dout`。

```python
class Affine:
    def __init__(self, W, b):
        self.W, self.b = W, b
        self.x, self.dW, self.db = None, None, None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
```

### Softmax-with-Loss 层

将 softmax 和交叉熵合并为一层后，反向传播退化为一个极其简洁的表达式：`y - t`（预测值 - 真实标签）。这是全书最漂亮的结论之一。

```python
class SoftmaxWithLoss:
    def __init__(self):
        self.loss, self.y, self.t = None, None, None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size
```

## 8. 卷积神经网络（CNN）

全连接层处理图像有两个致命问题：参数量爆炸 + 忽略空间结构。CNN 用**卷积核滑动**和**池化下采样**解决了这两个问题。

### 输出尺寸公式

输入尺寸 (H, W)、滤波器尺寸 (FH, FW)、填充 P、步幅 S，输出尺寸为：

$$
\begin{aligned}
OH &= \frac{H + 2P - FH}{S} + 1 \\[4pt]
OW &= \frac{W + 2P - FW}{S} + 1
\end{aligned}
$$

### im2col：卷积实现的关键技巧

卷积本质上就是"把滑动窗口展开成列向量，然后做矩阵乘法"。`im2col` 把 4D 图像张量转换成 2D 矩阵，让卷积变成高效的 `np.dot` 运算。

### 卷积层

```python
from common.util import im2col

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W, self.b = W, b
        self.stride, self.pad = stride, pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_w = self.W.reshape(FN, -1).T
        out = np.dot(col, col_w) + self.b
        return out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
```

### 池化层

对每个窗口取最大值（Max Pooling），大幅降低空间维度，同时保留最显著特征。

```python
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h, self.pool_w = pool_h, pool_w
        self.stride, self.pad = stride, pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)
        out = np.max(col, axis=1)
        return out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
```

### 完整 CNN 网络：SimpleConvNet

把所有层串起来：`Conv → ReLU → Pool → Affine → ReLU → Affine → SoftmaxWithLoss`。这就是一个完整的、可以用 MNIST 训练的卷积神经网络。

```python
import pickle
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5,
                             'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10,
                 weight_init_std=0.01):

        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size +
                            2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) ** 2)

        # 权重初始化
        self.params = {
            'W1': weight_init_std * np.random.randn(
                filter_num, input_dim[0], filter_size, filter_size),
            'b1': np.zeros(filter_num),
            'W2': weight_init_std * np.random.randn(
                pool_output_size, hidden_size),
            'b2': np.zeros(hidden_size),
            'W3': weight_init_std * np.random.randn(
                hidden_size, output_size),
            'b3': np.zeros(output_size),
        }

        # 构建层（OrderedDict 保证顺序）
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(
            self.params['W1'], self.params['b1'],
            conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(
            self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(
            self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        return self.last_layer.forward(self.predict(x), t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        return acc / x.shape[0]

    def gradient(self, x, t):
        # 前向
        self.loss(x, t)

        # 反向（逐层 backward，利用计算图自动求导）
        dout = self.last_layer.backward(1)
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        return {
            'W1': self.layers['Conv1'].dW,
            'b1': self.layers['Conv1'].db,
            'W2': self.layers['Affine1'].dW,
            'b2': self.layers['Affine1'].db,
            'W3': self.layers['Affine2'].dW,
            'b3': self.layers['Affine2'].db,
        }

    def save_params(self, file_name="params.pkl"):
        with open(file_name, 'wb') as f:
            pickle.dump(dict(self.params), f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params[f'W{i+1}']
            self.layers[key].b = self.params[f'b{i+1}']
```

### CNN 训练 MNIST

用 Adam 优化器训练 5 个 epoch，30 个 5×5 卷积核、100 个隐藏单元，在 MNIST 上轻松达到 98%+ 的测试准确率。

```python
from dataset.mnist import load_mnist
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=False)

network = SimpleConvNet(
    input_dim=(1, 28, 28),
    conv_param={'filter_num': 30, 'filter_size': 5,
                'pad': 0, 'stride': 1},
    hidden_size=100, output_size=10, weight_init_std=0.01)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=5, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

network.save_params("params.pkl")
print("Saved Network Parameters!")
```

**训练曲线**：训练和测试准确率随 epoch 稳步上升，5 个 epoch 后测试准确率达到 **98.2%**。

![CNN 训练曲线](/assets/blog/shen-du-xue-xi-ru-men/output_48_1.png)

## 总结

这本书之所以成为经典，在于它**把深度学习的"黑盒"彻底拆开**。读完这本书你会理解：

- 为什么反向传播比数值梯度快几百倍（计算图 + 链式法则）
- 为什么 CNN 处理图像比全连接强那么多（参数共享 + 局部感受野）
- 为什么 Softmax-with-Loss 层的反向传播是 `y - t`（数学的优雅）

如果你正在啃这本书，建议**边读边敲代码**——亲手 debug 过 `im2col` 的维度变换，才算真正理解了卷积。
