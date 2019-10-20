import csvReader as reader
import neuralNetwork as nn
import activationFunctions as F
import lossFunctions as L
import networkPrinter as printer

if __name__ == "__main__":


    # Xtrain, Ytrain = reader.readClassification3ClassesFile('classification\data.three_gauss.train.100.csv')
    # Xtest, Ytest = reader.readClassification3ClassesFile('classification\data.three_gauss.test.100.csv')
    Xtrain, Ytrain = reader.readClassificationFile('classification\data.simple.train.100.csv')
    Xtest, Ytest = reader.readClassificationFile('classification\data.simple.test.100.csv')
    
    net = nn.NeuralNetwork(momentumSize=1, seed=0)
    net.addLayer(2, 2, F.linear, False)
    net.addLayer(2, 3, F.sigmoid, True)
    net.setCostFunction(L.l2)

    #define own validation and train sets
    # net.trainAndValidate(Xtrain[20:], Ytrain[20:], Xtrain[0:20], Ytrain[0:20], epochs=10, learningRate=0.1, showError=True, showNodes=False)
    
    #automatic k-folds validation method
    net.kFoldsTrainAndValidate(Xtrain, Ytrain, k=10, epochs=20, learningRate=0.1, showError=True, showNodes=False)

    mean, std, error = net.validate(Xtest, Ytest)
    printer.print_classification(net, Xtest, Ytest)

    input('Click enter')