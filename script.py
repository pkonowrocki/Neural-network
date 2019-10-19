import csvReader as reader
import neuralNetwork as nn
import activationFunctions as F
import lossFunctions as L
import networkPrinter as printer

if __name__ == "__main__":
    Xtrain, Ytrain = reader.readClassificationFile('classification\data.simple.train.100.csv')
    Xtest, Ytest = reader.readClassificationFile('classification\data.simple.test.100.csv')
    net = nn.NeuralNetwork(momentumSize=1, seed=0)
    net.addLayer(2, 5, F.linear, False)
    net.addLayer(5, 1, F.sigmoid, True)
    net.setCostFunction(L.l2)

    net.trainAndValidate(Xtrain, Ytrain, Xtest[0:20], Ytest[0:20], epochs=100, learningRate=1, showError=True, showNodes=False)

    mean, std = net.validate(Xtest[20:], Ytest[20:])
    print(mean, std)
    input("Press Enter to continue...")