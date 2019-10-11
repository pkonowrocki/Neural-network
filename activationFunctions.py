import autograd.numpy as np

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def linear(x):
    return x

def tanh(x):
    return (np.exp(2*x) - 1) / (np.exp(2*x) +1)

def ReLU(x):
    return np.maximum(x, np.zeros(1))

def LReLU(x):
    return np.maximum(0.1*x, x)

def swish(x):
    return x/(1+np.exp(-x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))