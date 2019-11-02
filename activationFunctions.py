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
    m = np.max(x)
    y = np.exp(x-m)
    return y / y.sum()

def square(x):
    return x**2

def cube(x):
    return x**3

def polly(x):
    powers = np.arange(1, x.shape[0]+1, 1).reshape(x.shape)
    return  np.power(x, powers)
