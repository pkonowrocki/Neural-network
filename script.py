import csvReader as reader
import neuralNetwork as nn
import activationFunctions as F
import lossFunctions as L
import networkPrinter as printer

def classification3Classes():
    Xtrain, Ytrain = reader.readClassification3ClassesFile('classification\data.three_gauss.train.100.csv')
    Xtest, Ytest = reader.readClassification3ClassesFile('classification\data.three_gauss.test.100.csv')
    net = nn.NeuralNetwork(momentumSize=0, seed=0)
    net.addLayer(2, 10, F.linear, True)
    net.addLayer(10, 7, F.linear, True)
    net.addLayer(7, 5, F.linear, True)
    net.addLayer(5, 3, F.sigmoid, False)
    net.setCostFunction(L.l1)
    #define own validation and train sets
    # net.trainAndValidate(Xtrain[20:], Ytrain[20:], Xtrain[0:20], Ytrain[0:20], epochs=10, learningRate=0.1, showError=True, showNodes=False)
    #automatic k-folds validation method
    net.kFoldsTrainAndValidate(Xtrain, Ytrain, k=5, epochs=50, learningRate=0.1,
        showError=True, 
        showNodes=True,
        print=lambda : printer.print_classification(net, Xtrain, Ytrain, size=1.5, dS=0.005),
        showEvery=50)
    mean, std, error = net.validate(Xtest, Ytest)
    print(mean, std)
    printer.print_accuracy(net, Xtest, Ytest)

def classification2Classes():
    Xtrain, Ytrain = reader.readClassificationFile('classification\data.XOR.train.100.csv')
    Xtest, Ytest = reader.readClassificationFile('classification\data.XOR.test.100.csv')
    net = nn.NeuralNetwork(momentumSize=1, seed=0)
    net.addLayer(2, 10, F.tanh, True)
    net.addLayer(10, 1, F.sigmoid, True)
    net.setCostFunction(L.l1)
    #define own validation and train sets
    # net.trainAndValidate(Xtrain[20:], Ytrain[20:], Xtrain[0:20], Ytrain[0:20], epochs=10, learningRate=0.1, showError=True, showNodes=False)
    #automatic k-folds validation method
    net.kFoldsTrainAndValidate(Xtrain, Ytrain, k=5, epochs=500, learningRate=0.5, 
        showError=True, 
        showNodes=False,
        print=lambda : printer.print_classification(net, Xtest, Ytest, size=1.1, dS=0.1),
        showEvery=10)
    mean, std, error = net.validate(Xtest, Ytest)
    print(mean, std)
    printer.print_accuracy(net, Xtest, Ytest)

def regression():
    Xtrain, Ytrain = reader.readRegressionFile('regression\data.square.train.100.csv')
    Xtest, Ytest = reader.readRegressionFile('regression\data.square.test.100.csv')
    net = nn.NeuralNetwork(momentumSize=1, seed=100)
    net.addLayer(1, 2, F.polly, bias=True)
    net.addLayer(2, 1, F.linear, bias=True)

    net.setCostFunction(L.l1)
    #define own validation and train sets
    # net.trainAndValidate(Xtrain[20:], Ytrain[20:], Xtrain[0:20], Ytrain[0:20], epochs=10, learningRate=0.1, showError=True, showNodes=False)
    #automatic k-folds validation method
    net.kFoldsTrainAndValidate(Xtrain, Ytrain, k=20, batchSize=1, epochs=2000, learningRate=6e-4, momentumRate=2e-5, 
        showError=True, 
        showNodes=False,
        print = lambda : printer.print_regression(net, Xtrain, Ytrain, size=5, dx=0.25),
        showEvery= 10)
    mean, std, error = net.validate(Xtest, Ytest)
    print(mean, std)

if __name__ == "__main__":
    # classification3Classes()
    classification2Classes()
    # regression()
    input('Click enter')
