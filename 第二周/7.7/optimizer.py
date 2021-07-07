import numpy as np

# y=wx+b
# 损失采用最小二乘法
# n=10个样本：(0.0,0.9)、(0.5,2.1)、(0.8,2.7)、(1.1,3.1)、(1.5,4.1)、(1.9,4.8 )、(2.2,5.1)、(2.4,5.9)、(2.6,6.0)、(3.0,7.0)
# learning rate为0.01
# 实现下列优化器
# 1、SGD
# 2、Momentum
# 3、RMSprop
# 4、Adam
# 初始化(w,b)=(0,0)
# SGD方法迭代100次所达到的损失值为基准，比较其他几种优化方法达到该值所需要的迭代轮次


data = [[0.0, 0.9], [0.5, 2.1], [0.8, 2.7], [1.1, 3.1], [1.5, 4.1], [1.9, 4.8], [2.2, 5.1], [2.4, 5.9], [2.6, 6.0],
        [3.0, 7.0]]
data = np.array(data)
w, b = 0, 0
X = data[:, 0]
Y = data[:, 1]


def SGD(learning_rate):
    sgd_w, sgd_b = 0, 0
    for i in range(1, 100):
        loss = 0
        for x, y in zip(X, Y):
            y_hat = sgd_w * x + sgd_b
            loss += (y - y_hat) ** 2 / 2
            sgd_w = sgd_w - learning_rate * (y_hat - y) * x
            sgd_b = sgd_b - learning_rate * (y_hat - y)
        if i == 99:
            print(loss / 10)


def Momentum(learning_rate, beta):
    mom_w, mom_b, dw, db = 0., 0., 0., 0.
    for i in range(1, 100):
        loss = 0
        for x, y in zip(X, Y):
            y_hat = mom_w * x + mom_b
            loss += (y - y_hat) ** 2 / 2
            dw = beta * dw + (1 - beta) * (y_hat - y) * x
            db = beta * db + (1 - beta) * (y_hat - y)
            mom_w = mom_w - learning_rate * dw
            mom_b = mom_b - learning_rate * db
        if i == 99:
            print(loss / 10)


def RMSprop(learning_rate, beta):
    rms_w, rms_b, sdw, sdb = 0., 0., 0., 0.
    esp = 1e-8
    for i in range(1, 100):
        loss = 0
        for x, y in zip(X, Y):
            y_hat = rms_w * x + rms_b
            loss += (y - y_hat) ** 2 / 2
            dw = (y_hat - y) * x
            db = y_hat - y
            sdw = beta * sdw + (1 - beta) * (dw ** 2)
            sdb = beta * sdb + (1 - beta) * (db ** 2)
            rms_w = rms_w - learning_rate * dw / (np.sqrt(sdw) + esp)
            rms_b = rms_b - learning_rate * db / (np.sqrt(sdb) + esp)
        if i == 99:
            print(loss / 10)


def Adam(learning_rate, beta1, beta2):
    ada_w, ada_b, sdw, sdb, vdw, vdb = 0., 0., 0., 0., 0., 0.
    esp = 1e-8
    for i in range(1, 100):
        loss = 0
        for x, y in zip(X, Y):
            y_hat = ada_w * x + ada_b
            loss += (y - y_hat) ** 2 / 2
            dw = (y_hat - y) * x
            db = y_hat - y
            # Momentum
            vdw = beta1 * vdw + (1 - beta1) * dw
            vdb = beta1 * vdb + (1 - beta1) * db
            # RMSprop
            sdw = beta2 * sdw + (1 - beta2) * (dw ** 2)
            sdb = beta2 * sdb + (1 - beta2) * (db ** 2)

            vdw_corrent = vdw / (1 - beta1 ** 2)
            vdb_corrent = vdb / (1 - beta1 ** 2)

            sdw_corrent = sdw / (1 - beta1 ** 2)
            sdb_corrent = sdb / (1 - beta2 ** 2)

            ada_w = ada_w - learning_rate * vdw_corrent / (np.sqrt(sdw_corrent) + esp)
            ada_b = ada_b - learning_rate * vdb_corrent / (np.sqrt(sdb_corrent) + esp)
        if i == 99:
            print(loss / 10)


SGD(0.001)
Momentum(0.001, 0.9)
RMSprop(0.001, 0.9)
Adam(0.001, 0.9, 0.999)
