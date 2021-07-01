import numpy as np

network_sizes = [2, 3, 2]

# 初始化网络
sizes = network_sizes
num_layers = len(sizes)
biases = [np.random.randn(h, 1) for h in sizes[1:]]
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


def loss_der(net_y, real_y):
    # 返回损失函数的偏导，损失采用最小二乘损失函数
    # L=1/2 * (real_y-net_y)^2 对net_y偏导是:net_y-real_y
    return (net_y - real_y)


def sigmod(z):
    # 激活函数sigmod
    return 1.0 / (1.0 + np.exp(z))


def sigmod_der(z):
    # sigmod 函数的导数
    return sigmod(z) * (1 - sigmod(z))


def backprop(x, y):
    # 反向传播过程

    delta_w = [np.zeros(w.shape) for w in weights]
    delta_b = [np.zeros(b.shape) for b in biases]

    # 前向传播
    activarion = x  # 输入数据作为第一次激活值
    activarions = [x]  # 存储网络中的激活值
    zs = []  # 存储加权输入值（wx+b）
    # 循环迭代求出每层网络的a和z
    for w, b in zip(weights, biases):
        z = np.dot(w, activarion) + b
        activarion = sigmod(z)
        activarions.append(activarion)
        zs.append(z)

    # 反向传播
    # 计算输出层误差
    delta_L = loss_der(activarions[-1], y) * sigmod(zs[-1])
    delta_b[-1] = delta_L
    delta_w[-1] = np.dot(delta_L, activarions[-2].T)

    delta_l = delta_L
    for l in range(2, num_layers):
        # 计算第一层误差
        z = zs[-l]
        sp = sigmod_der(z)
        delta_l = np.dot(weights[-l + 1].T, delta_l) * sp
        delta_b[-l] = delta_l
        delta_w[-l] = np.dot(delta_l, activarions[-l - 1].T)

    return delta_w, delta_b


def main():
    # 产生训练数据
    train_x = np.random.randn(2).reshape(2, 1)
    train_y = np.array([0, 1]).reshape(2, 1)
    w, b = backprop(train_x, train_y)
    print("w:", w)
    print("b:", b)


if __name__ == '__main__':
    main()
