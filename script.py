import csvReader as reader
import neuralNetwork as nn
import activationFunctions as F
import lossFunctions as L
import networkPrinter as printer

def classification3Classes():
    Xtrain, Ytrain = reader.readClassification3ClassesFile('classification\data.three_gauss.train.100.csv')
    Xtest, Ytest = reader.readClassification3ClassesFile('classification\data.three_gauss.test.100.csv')
    net = nn.NeuralNetwork(momentumSize=0, seed=0)
    net.addLayer(2, 5, F.linear, True)
    net.addLayer(5, 5, F.LReLU, True)
    net.addLayer(5, 10, F.linear, True)
    net.addLayer(10, 3, F.sigmoid, False)
    net.setCostFunction(L.l2)
    #define own validation and train sets
    # net.trainAndValidate(Xtrain[20:], Ytrain[20:], Xtrain[0:20], Ytrain[0:20], epochs=10, learningRate=0.1, showError=True, showNodes=False)
    #automatic k-folds validation method
    net.kFoldsTrainAndValidate(Xtrain, Ytrain, k=15, epochs=100, learningRate=0.1,
        showError=True, 
        showNodes=False,
        print=lambda : printer.print_classification(net, Xtrain, Ytrain, size=1, dS=0.1),
        showEvery=10)
    mean, std, error = net.validate(Xtest, Ytest)

def classification2Classes():
    Xtrain, Ytrain = reader.readClassificationFile('classification\data.simple.train.100.csv')
    Xtest, Ytest = reader.readClassificationFile('classification\data.simple.test.100.csv')
    net = nn.NeuralNetwork(momentumSize=1, seed=0)
    net.addLayer(2, 8, F.linear, True)
    net.addLayer(8, 4, F.swish, True)
    net.addLayer(4, 2, F.linear, True)
    net.addLayer(2, 1, F.sigmoid, True)
    net.setCostFunction(L.l1)
    #define own validation and train sets
    # net.trainAndValidate(Xtrain[20:], Ytrain[20:], Xtrain[0:20], Ytrain[0:20], epochs=10, learningRate=0.1, showError=True, showNodes=False)
    #automatic k-folds validation method
    net.kFoldsTrainAndValidate(Xtrain, Ytrain, k=5, epochs=30, learningRate=0.2, 
        showError=True, 
        showNodes=False,
        print=lambda : printer.print_classification(net, Xtrain, Ytrain, size=1, dS=0.02),
        showEvery=2)
    mean, std, error = net.validate(Xtest, Ytest)

def regression():
    Xtrain, Ytrain = reader.readRegressionFile('regression\data.activation.train.100.csv')
    Xtest, Ytest = reader.readRegressionFile('regression\data.activation.test.100.csv')
    net = nn.NeuralNetwork(momentumSize=1, seed=305)
    net.addLayer(1, 1, F.tanh, True)
    net.addLayer(1, 1, F.linear, True)
    net.setCostFunction(L.l1)
    #define own validation and train sets
    # net.trainAndValidate(Xtrain[20:], Ytrain[20:], Xtrain[0:20], Ytrain[0:20], epochs=10, learningRate=0.1, showError=True, showNodes=False)
    #automatic k-folds validation method
    net.kFoldsTrainAndValidate(Xtrain, Ytrain, k=10, batchSize=2, epochs=2000, learningRate=2e-3, momentumRate=1e-10, 
        showError=True, 
        showNodes=True,
        print = lambda : printer.print_regression(net, Xtrain, Ytrain),
        showEvery= 400)
    mean, std, error = net.validate(Xtest, Ytest)

if __name__ == "__main__":
    classification3Classes()
    # classification2Classes()
    # regression()
    input('Click enter')


