import numpy as np
import random
import pickle
import platform
import os


# 加载序列文件
def load_pickle(f):
    version = platform.python_version_tuple()  # 判断python的版本
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version:{}".format(version))


# 处理原数据
def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        # reshape()是在不改变矩阵的数值的前提下修改矩阵的形状,transpose()对矩阵进行转置
        Y = np.array(Y)
        return X, Y


# 返回可以直接使用的数据集
def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))  # os.path.join()将多个路径组合后返回
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)  # 这个函数用于将多个数组进行连接
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte



# import  os
# print(os.path.abspath('.') )
# print(os.path.abspath('..'))

datasets = 'E:\PythonProject\datasets\CIFAR-10\cifar-10-batches-py'
X_train, Y_train, X_test, Y_test = load_CIFAR10(datasets)
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', Y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', Y_test.shape)
