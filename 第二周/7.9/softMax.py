import numpy as np


def softMax(X):
    X = X - max(X)  # 防止乘e之后数值过大
    X = np.exp(X)
    sum = np.sum(X)
    X = X / sum
    return X


input = np.array([2, -1, 4, 3])
output = softMax(input)
print(output)
