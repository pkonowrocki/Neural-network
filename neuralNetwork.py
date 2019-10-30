import time
import autograd.numpy as np
import autograd 
import activationFunctions as F
import lossFunctions as L
import networkPrinter as printer
import matplotlib.pyplot as plt
import csvReader as reader

class NeuralNetwork:
    def __init__(self, seed = None, momentumSize = 5):
        if(seed is not None):
            np.random.seed(seed)
        self.paramsValues = {}
        self.functions = {}
        self.numberOfLayers = 0
        self.memory = {}
        self.gradsValues = {}
        self.momentumValues = []
        self.momentumSize = momentumSize

        # additional info for displaying
        self.layerSize = {}
        self.biggestLayerSize = 0
    
    def addLayer(self, inputSize, outputSize, activationFunction, bias = True):
        self.numberOfLayers = self.numberOfLayers + 1
        self.paramsValues['W' + str(self.numberOfLayers)] = np.random.rand(outputSize, inputSize)*2 - 1
        if (bias):
            self.paramsValues['b' + str(self.numberOfLayers)] = np.random.rand(outputSize, 1)*2 - 1
            
        self.functions['F' + str(self.numberOfLayers)] = activationFunction
        self.functions['dF' + str(self.numberOfLayers)] = autograd.grad(activationFunction, 0)

        # additional info for displaying
        if (self.numberOfLayers is 1):
            self.layerSize[0] = inputSize
            if self.biggestLayerSize < inputSize:
                self.biggestLayerSize = inputSize
        self.layerSize[self.numberOfLayers] = outputSize
        if self.biggestLayerSize < outputSize:
                self.biggestLayerSize = outputSize

    def _singleLayerForward(self, Aprev, Wcurr, bcurr, activationFunction):
        dot = np.dot(Wcurr, Aprev)
        Zcurr = np.add(dot, bcurr)
        return activationFunction(Zcurr), Zcurr

    def forward(self, X):
        Acurr = X
        for i in range(self.numberOfLayers):
            idx = i + 1
            Aprev = Acurr
            activationFunction = self.functions['F' + str(idx)]
            Wcurr = self.paramsValues['W' + str(idx)]
            if (('b'+str(idx)) in self.paramsValues.keys()):
                bcurr = self.paramsValues['b' + str(idx)]
            else:
                bcurr = np.zeros((Wcurr.shape[0], 1))
            Acurr, Zcurr = self._singleLayerForward(Aprev, Wcurr, bcurr, activationFunction)
            
            self.memory['A' + str(idx - 1)] = Aprev
            self.memory['Z' + str(idx)] = Zcurr
        return Acurr
    
    def _singleLayerBackward(self, dAcurr, Wcurr, bcurr, Zcurr, Aprev, backwardActivationFunc):
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

        dAprev = self.dL(Yhat, Y)

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

            dAprev, dWcurr, dbcurr = self._singleLayerBackward(dAcurr, Wcurr, bcurr, Zcurr, Aprev, self.functions['dF'+str(idx)])

            self.gradsValues['dW'+str(idx)] += dWcurr
            
            if (('b'+str(idx)) in self.paramsValues.keys()):
                self.gradsValues['db'+str(idx)] += dbcurr

        return self.gradsValues

    def clearGradsValues(self):
        for k in self.paramsValues.keys():
            self.gradsValues['d'+k] = 0

    def setCostFunction(self, costFunction):
        self.costFunction = costFunction
        self.dL = autograd.grad(self.costFunction, 0)

    def getCostValue(self, Yhat, Y):
        cost = self.costFunction(Yhat, Y)
        return cost

    def update(self, learningRate, momentumRate):
        if(momentumRate != 0 and self.momentumSize != 0):
            momentum = self._getMomentum()

        for i in range(self.numberOfLayers):
            idx = i+1
            if(momentumRate != 0 and self.momentumSize != 0):
                self.paramsValues['W'+str(idx)] += momentumRate*momentum['dW'+str(idx)] - learningRate*self.gradsValues['dW'+str(idx)]
                if('b'+str(idx) in self.paramsValues.keys()):
                    self.paramsValues['b'+str(idx)] += momentumRate*momentum['db'+str(idx)] - learningRate*self.gradsValues['db'+str(idx)]
            else:
                self.paramsValues['W'+str(idx)] -= learningRate*self.gradsValues['dW'+str(idx)]
                if('b'+str(idx) in self.paramsValues.keys()):
                    self.paramsValues['b'+str(idx)] -= learningRate*self.gradsValues['db'+str(idx)]
        if(momentumRate != 0 and self.momentumSize != 0):
            self._addToMomentum(self.gradsValues)

    def _addToMomentum(self, gradsValues):
        if(len(self.momentumValues) == self.momentumSize):
            self.momentumValues.pop(0)
        self.momentumValues.append(gradsValues)

    def _getMomentum(self):
        momentum = {}
        if(len(self.momentumValues) == 0):
            for k in self.paramsValues.keys():
                momentum['d'+k] = 0
        else:
            for k in self.gradsValues.keys():
                for i in self.momentumValues:
                    if (k in momentum.keys()):
                        momentum[k] += i[k]
                    else:
                        momentum[k] = i[k]
                momentum[k] = momentum[k] / len(self.momentumValues)
        return momentum

    def _trainOnce(self, X, Y, batchSize, learningRate, momentumRate):
        self.clearGradsValues()
        err = []
        for i in range(len(X)):
            result = self.forward(X[i])
            cost = self.getCostValue(result, Y[i])
            self.backward(result, Y[i])
            err.append(cost)
            if batchSize == 0:
                self.update(learningRate, momentumRate)
                self.clearGradsValues()
            else:
                if((i+1)%batchSize == 0):
                    self.update(learningRate, momentumRate)
                    self.clearGradsValues()
                elif(i+1 == len(X)):
                    self.update(learningRate, momentumRate)
                    self.clearGradsValues()
        return np.mean(err), np.std(err)

    def train(self, X, Y, epochs = 100, batchSize = 0, learningRate = 0.1, momentumRate = 0, showNodes = False, showError = True):
        if showNodes or showError:
            printer.initialize()

        trainingError = []
        trainingStd = []
        for e in range(epochs):
            mean, std = self._trainOnce(X, Y, batchSize, learningRate, momentumRate)
            trainingError.append(mean)
            trainingStd.append(std)
            print('Epoch: ',e+1, '\terr mean: ', mean, '\terr std:', std)
            
            if showError:
                plt.figure(1)
                printer.print_error(trainingError)
            
            if showNodes: 
                plt.figure(69)     
                printer.print_network(self)
            
            if showError or showNodes:
                printer.wait(0.01)

    def validate(self, X, Y):
        testError = []
        for i in range(len(X)):
            result = self.forward(X[i])
            cost = self.getCostValue(result, Y[i])
            testError.append(cost)
        return np.mean(testError), np.std(testError), testError

    def trainAndValidate(self, Xtrain, Ytrain, Xvalidation, Yvalidation, epochs = 100, batchSize = 0, learningRate = 0.1, momentumRate = 0, showNodes = False, showError = True):
        if showNodes or showError:
            printer.initialize()

        trainingError = []
        trainingStd = []
        validateError = []
        validateStd = []

        for e in range(epochs):
            meanT, stdT = self._trainOnce(Xtrain, Ytrain, batchSize, learningRate, momentumRate)
            trainingError.append(meanT)
            trainingStd.append(stdT)
            meatV, stdV, errorV = self.validate(Xvalidation, Yvalidation)
            validateError.append(meatV)
            validateStd.append(stdV)
            if showError:
                plt.figure(1)
                printer.print_error(trainingError, validateError)
            
            if showNodes: 
                plt.figure(69)     
                printer.print_network(self)
            
            if showError or showNodes:
                printer.wait(0.01)

    def kFoldsTrainAndValidate(self, Xtrain, Ytrain, k = 4, epochs = 100, batchSize = 0, learningRate = 0.1, momentumRate = 0, showNodes = False, showError = True, print = None, showEvery = 1):
        X = []
        Y = []
        for _ in range(k):
            X.append([])
            Y.append([])
        for i in range(len(Xtrain)):
            X[i%k].append(Xtrain[i])
            Y[i%k].append(Ytrain[i])

        if showNodes or showError:
            printer.initialize()

        trainingError = []
        trainingStd = []
        validateError = []
        validateStd = []

        for e in range(epochs):
            validationFold = 0
            meanT = []
            for fold in range(k):
                if validationFold == fold:
                    meatV, stdV, errorV = self.validate(X[fold], Y[fold])
                    validateError.append(meatV)
                    validateStd.append(stdV)
                mean, stdT = self._trainOnce(X[fold], Y[fold], batchSize, learningRate, momentumRate)
                meanT.append(mean)
            
            validationFold = (validationFold+1)%k
            trainingError.append(np.mean(meanT))
            trainingStd.append(stdT)

            if showError and (e+1)%showEvery==0:
                plt.figure(1)
                printer.print_error(trainingError, validateError)
            
            if showNodes and (e+1)%showEvery==0: 
                plt.figure(69)     
                printer.print_network(self)
            
            if print is not None and (e+1)%showEvery==0:
                print()

            if (showError or showNodes or print is not None) and (e+1)%showEvery==0:
                printer.wait(0.01)
        
