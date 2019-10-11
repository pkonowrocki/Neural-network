import autograd.numpy as np

def binaryCrossEntropy(yhat, y):
    return -np.sum(y*np.log(yhat))/yhat.shape[0]

def l1(yhat, y):
    return np.sum(np.abs(y-yhat))/yhat.shape[0]

def l2(yhat, y):
    return np.sum((y-yhat)**2)/yhat.shape[0]

def crossEntropy(yhat, y):
    return -np.sum(y*np.log(yhat))/yhat.shape[0]