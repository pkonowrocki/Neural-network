import autograd.numpy as np

alpha = 0.1

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def linear(x):
    return x

def tanh(x):
    return (np.exp(2*x) - 1) / (np.exp(2*x) +1)

def ReLU(x):
    return np.maximum(x, np.zeros(1))

def LReLU(x):
    return np.maximum(alpha*x, x)

def swish(x):
    return x / (1.+np.exp(-x))

def softmax(x):
    x = x - x.max(axis=0, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=0, keepdims=True)

def square(x):
    return x**2