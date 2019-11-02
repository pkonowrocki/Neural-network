# https://www.kaggle.com/c/digit-recognizer/data
import csvReader as reader
import neuralNetwork as nn
import activationFunctions as F
import lossFunctions as L
import networkPrinter as printer

def _test(e, Xtest, net, name):
    Ytest = []
    for x in Xtest:
        Ytest.append(net.forward(x))
    reader.saveMnistResult(name+'Epoch'+str(e)+'.csv', Ytest)
    print('Saved epoch ', e)

def test0():
    name = 'test0'
    Xtrain, Ytrain = reader.readmnistAsOneLineTraining('mnist\\train.csv')
    print('Training set ready, size: ', len(Xtrain))
    net = nn.NeuralNetwork(momentumSize=0)
    net.addLayer(784, 10, F.softmax, True)
    net.setCostFunction(L.crossEntropy)
    print('Starting training...')
    Xtest = reader.readmnistAsOneLineTest('mnist\\test.csv')
    net.kFoldsTrainAndValidate(Xtrain, Ytrain,
        k=6, 
        epochs=100, 
        learningRate=1e-2,
        batchSize=100,
        showError=True, 
        showNodes=False,
        print=  lambda e: _test(e, Xtest, net, name),
        showEvery=5,
        name = name)

def test1():
    name = 'test1'
    Xtrain, Ytrain = reader.readmnistAsOneLineTraining('mnist\\train.csv')
    Xtrain = Xtrain[1:3]
    Ytrain = Ytrain[1:3]
    print('Training set ready, size: ', len(Xtrain))
    net = nn.NeuralNetwork(momentumSize=0, seed=1)
    net.addLayer(784, 200, F.sigmoid, False)
    net.addLayer(200, 80, F.sigmoid, False)
    net.addLayer(80,10, F.softmax, False)
    net.setCostFunction(L.l2)
    print('Starting training...')
    Xtest = reader.readmnistAsOneLineTest('mnist\\test.csv')
    net.kFoldsTrainAndValidate(Xtrain, Ytrain,
        k=2, 
        epochs=100, 
        learningRate=1e-2,
        batchSize=100,
        showError=True, 
        showNodes=False,
        #print=  lambda e: _test(e, Xtest, net, name),
        showEvery=10,
        name = None)

def test2():
    name = 'test2'
    Xtrain, Ytrain = reader.readmnistAsOneLineTraining('mnist\\train.csv')
    print('Training set ready, size: ', len(Xtrain))
    net = nn.NeuralNetwork(momentumSize=0)
    net.addLayer(784, 200, F.sigmoid, False)
    net.addLayer(200, 80, F.tanh, False)
    net.addLayer(80,10, F.softmax, False)
    net.setCostFunction(L.l2)
    print('Starting training...')
    Xtest = reader.readmnistAsOneLineTest('mnist\\test.csv')
    net.kFoldsTrainAndValidate(Xtrain, Ytrain, 
        k=100, 
        epochs=3, 
        learningRate=1e-4,
        batchSize=10,
        showError=False, 
        showNodes=False,
        print=  lambda e: _test(e, Xtest, net, name),
        showEvery=1,
        name = name)
    

if __name__ == "__main__":
    test0()
    # test1()
    # test2()
    input('Click enter')
