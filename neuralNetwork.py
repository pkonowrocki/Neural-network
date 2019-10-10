# https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
#import numpy as np
from datetime import datetime
import autograd.numpy as np
import autograd 

class NeuralNetwork:
    def __init__(self, seed = datetime.now()):
        np.random.seed(seed)
        self.paramsValues = {}
        self.numberOfLayers = 0
        self.memory = {}
        self.gradsValues = {}
    
    def printNetwork(self):
        print(self.paramsValues)
    
    def addLayer(self, inputSize, outputSize, activationFunction, bias = True):
        self.numberOfLayers = self.numberOfLayers + 1
        self.paramsValues['W' + str(self.numberOfLayers)] = np.random.rand(outputSize, inputSize)*0 + 0.1
        if (bias):
            self.paramsValues['b' + str(self.numberOfLayers)] = np.random.rand(outputSize, 1)*0.1
        self.paramsValues['F' + str(self.numberOfLayers)] = activationFunction
        self.paramsValues['dF' + str(self.numberOfLayers)] = autograd.grad(activationFunction, 0)

    def singleLayerForward(self, Aprev, Wcurr, bcurr, activationFunction):
        dot = np.dot(Wcurr, Aprev)
        Zcurr = np.add(dot, bcurr)
        return activationFunction(Zcurr), Zcurr

    def forward(self, X):
        Acurr = X
        for i in range(self.numberOfLayers):
            idx = i + 1
            Aprev = Acurr
            activationFunction = self.paramsValues['F' + str(idx)]
            Wcurr = self.paramsValues['W' + str(idx)]
            if (('b'+str(idx)) in self.paramsValues.keys()):
                bcurr = self.paramsValues['b' + str(idx)]
            else:
                bcurr = np.zeros((Wcurr.shape[0], 1))
            Acurr, Zcurr = self.singleLayerForward(Aprev, Wcurr, bcurr, activationFunction)
            
            self.memory['A' + str(idx - 1)] = Aprev
            self.memory['Z' + str(idx)] = Zcurr
        return Acurr
    
    def singleLayerBackward(self, dAcurr, Wcurr, bcurr, Zcurr, Aprev, backwardActivationFunc):
        m = Aprev.shape[1]

        dZ = list()
        for z in list(Zcurr):
            dZ.append(backwardActivationFunc(z))
        dZ = np.array(dZ)
        dZcurr = dZ*dAcurr
        dWcurr = np.dot(dZcurr, Aprev.T) / m
        dbcurr = np.sum(dZcurr, axis = 1, keepdims = True) / m
        dAprev = np.dot(Wcurr.T, dZcurr)

        return dAprev, dWcurr, dbcurr
    
    def backward(self, Yhat, Y):
        m = Y.shape[1]
        Y = Y.reshape(Yhat.shape)

        dL = autograd.grad(self.costFunction, 0)
        dAprev = dL(Yhat, Y)

        for i in reversed(list(range(self.numberOfLayers))):
            idx = i+1

            dAcurr = dAprev
            Aprev = self.memory['A' + str(idx - 1)]
            Zcurr = self.memory['Z' + str(idx)]
            Wcurr = self.paramsValues['W' + str(idx)]
            
            if (('b'+str(idx)) in self.paramsValues.keys()):
                bcurr = self.paramsValues['b' + str(idx)]
            else:
                bcurr = np.zeros((Wcurr.shape[0], 1))

            dAprev, dWcurr, dbcurr = self.singleLayerBackward(dAcurr, Wcurr, bcurr, Zcurr, Aprev, self.paramsValues['dF'+str(idx)])

            self.gradsValues['dW'+str(idx)] = dWcurr
            
            if (('b'+str(idx)) in self.paramsValues.keys()):
                self.gradsValues['db'+str(idx)] = dbcurr
        
        return self.gradsValues

    def setCostFunction(self, costFunction):
        self.costFunction = costFunction

    def getCostValue(self, Yhat, Y):
        cost = self.costFunction(Yhat, Y)
        return cost

    def update(self, learningRate):
        for i in range(self.numberOfLayers):
            idx = i+1
            self.paramsValues['W'+str(idx)] -= learningRate*self.gradsValues['dW'+str(idx)]
            if('b'+str(idx) in self.paramsValues.keys()):
                self.paramsValues['b'+str(idx)] -= learningRate*self.gradsValues['db'+str(idx)]
    

if __name__ == "__main__":
    net = NeuralNetwork(0)
    net.addLayer(2, 10, lambda x : 1. / (1. + np.exp(-x)), True)
    net.addLayer(10, 5, lambda x : 1. / (1. + np.exp(-x)), True)
    net.addLayer(5, 2, lambda x : x, True)
    net.setCostFunction(lambda yhat, y : np.sum((y-yhat)**2))

    for _ in range(10):
        result = net.forward(np.array([[0, 1]]).T)
        cost = net.getCostValue(result, np.array([[1, 1]]).T)
        backward = net.backward(result,  np.array([[1, 1]]).T)
        print('result: ', result)
        print('cost value: ', cost)

        net.update(0.1)