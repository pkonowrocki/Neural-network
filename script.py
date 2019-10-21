import csvReader as reader
import neuralNetwork as nn
import activationFunctions as F
import lossFunctions as L
import networkPrinter as printer

def classification3Classes():
    Xtrain, Ytrain = reader.readClassification3ClassesFile('classification\data.three_gauss.train.100.csv')
    Xtest, Ytest = reader.readClassification3ClassesFile('classification\data.three_gauss.test.100.csv')
    net = nn.NeuralNetwork(momentumSize=1, seed=0)
    net.addLayer(2, 9, F.linear, True)
    net.addLayer(9, 15, F.LReLU, True)
    net.addLayer(15, 6, F.LReLU, True)
    net.addLayer(6, 3, F.tanh, True)
    net.setCostFunction(L.l2)
    #define own validation and train sets
    # net.trainAndValidate(Xtrain[20:], Ytrain[20:], Xtrain[0:20], Ytrain[0:20], epochs=10, learningRate=0.1, showError=True, showNodes=False)
    #automatic k-folds validation method
    net.kFoldsTrainAndValidate(Xtrain, Ytrain, k=10, epochs=20, learningRate=0.1, showError=True, showNodes=False)
    mean, std, error = net.validate(Xtest, Ytest)
    printer.print_classification(net, Xtest, Ytest)

def classification2Classes():
    Xtrain, Ytrain = reader.readClassificationFile('classification\data.simple.train.100.csv')
    Xtest, Ytest = reader.readClassificationFile('classification\data.simple.test.100.csv')
    net = nn.NeuralNetwork(momentumSize=1, seed=0)
    net.addLayer(2, 9, F.linear, True)
    net.addLayer(9, 15, F.LReLU, True)
    net.addLayer(15, 6, F.LReLU, True)
    net.addLayer(6, 1, F.tanh, True)
    net.setCostFunction(L.l2)
    #define own validation and train sets
    # net.trainAndValidate(Xtrain[20:], Ytrain[20:], Xtrain[0:20], Ytrain[0:20], epochs=10, learningRate=0.1, showError=True, showNodes=False)
    #automatic k-folds validation method
    net.kFoldsTrainAndValidate(Xtrain, Ytrain, k=10, epochs=20, learningRate=0.1, showError=True, showNodes=False)
    mean, std, error = net.validate(Xtest, Ytest)
    printer.print_classification(net, Xtest, Ytest) 

def regression():
    Xtrain, Ytrain = reader.readRegressionFile('regression\data.activation.train.100.csv')
    Xtest, Ytest = reader.readRegressionFile('regression\data.activation.test.100.csv')
    net = nn.NeuralNetwork(momentumSize=1, seed=0)
    net.addLayer(1, 2, F.linear, True)
    net.addLayer(2, 1, F.linear, True)
    net.setCostFunction(L.l2)
    #define own validation and train sets
    # net.trainAndValidate(Xtrain[20:], Ytrain[20:], Xtrain[0:20], Ytrain[0:20], epochs=10, learningRate=0.1, showError=True, showNodes=False)
    #automatic k-folds validation method
    net.kFoldsTrainAndValidate(Xtrain, Ytrain, k=10, epochs=20, learningRate=0.1, showError=True, showNodes=False)
    mean, std, error = net.validate(Xtest, Ytest)

if __name__ == "__main__":
    # classification3Classes()
    # classification2Classes()
    regression()
    input('Click enter')


