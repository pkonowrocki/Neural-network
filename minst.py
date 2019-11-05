# https://www.kaggle.com/c/digit-recognizer/data
import csvReader as reader
import neuralNetwork as nn
import activationFunctions as F
import lossFunctions as L
import networkPrinter as printer
import numpy as np
import matplotlib.pyplot as plt

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

def reduceMNIST(n):
    Xtrain, Ytrain = reader.readmnistAsOneLineTraining('mnist\\train.csv')
    Xtest = reader.readmnistAsOneLineTest('mnist\\test.csv')
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 784)
    Xtrain = np.array(Xtrain).reshape((42000, 784))
    pca = pca.fit(Xtrain)
    cum_var_explained = np.cumsum(pca.explained_variance_)/np.sum(pca.explained_variance_)
    plt.figure(10000)
    plt.title("Wykres wytłumaczonej wariancji zbioru")
    plt.ylabel("Część wytłumaczonej wariancji")
    plt.xlabel("Liczba wymiarów")
    plt.grid(True)
    plt.plot(cum_var_explained)
    plt.show()

    # pca = PCA(n_components = n)
    # Xtrain = pca.fit_transform(np.array(Xtrain).reshape((42000, 784)))
    # np.savetxt('mnist\\trainPCA'+str(n)+'.csv', np.round(Xtrain, 5), delimiter=',')
    # Xtest = np.array(Xtest).reshape((28000, 784))
    # Xtest = np.round(pca.transform(Xtest), 5)
    # np.savetxt('mnist\\testPCA'+str(n)+'.csv', Xtest, delimiter=',')

def test1():
    _, Ytrain = reader.readmnistAsOneLineTraining('mnist\\train.csv')
    Xtrain = reader.readmnistAsOneLinePCA('mnist\\trainPCA300.csv')
    Xtest = reader.readmnistAsOneLinePCA('mnist\\testPCA300.csv')
    name = 'test1'
    print('Training set ready, size: ', len(Xtrain))
    net = nn.NeuralNetwork(momentumSize=0)
    net.addLayer(300, 10, F.softmax, True)
    net.setCostFunction(L.crossEntropy)
    print('Starting training...')
    net.kFoldsTrainAndValidate(Xtrain, Ytrain,
        k=6, 
        epochs=2, 
        learningRate=1e-2,
        batchSize=100,
        showError=True, 
        showNodes=False,
        print=  lambda e: _test(e, Xtest, net, name),
        showEvery=1,
        name = name)

def test2():
    _, Ytrain = reader.readmnistAsOneLineTraining('mnist\\train.csv')
    Xtrain = reader.readmnistAsOneLinePCA('mnist\\trainPCA300.csv')
    Xtest = reader.readmnistAsOneLinePCA('mnist\\testPCA300.csv')
    name = '2test'
    print('Training set ready, size: ', len(Xtrain))
    net = nn.NeuralNetwork(momentumSize=0)
    net.addLayer(300, 100, F.LReLU, True)
    net.addLayer(100, 10, F.softmax, True)
    net.setCostFunction(L.crossEntropy)
    print('Starting training...')
    net.kFoldsTrainAndValidate(Xtrain, Ytrain,
        k=6, 
        epochs=100, 
        learningRate=1e-2,
        batchSize=100,
        showError=True, 
        showNodes=False,
        print=  lambda e: _test(e, Xtest, net, name),
        showEvery=50,
        name = name)
    
def test3():
    _, Ytrain = reader.readmnistAsOneLineTraining('mnist\\train.csv')
    Xtrain = reader.readmnistAsOneLinePCA('mnist\\trainPCA300.csv')
    Xtest = reader.readmnistAsOneLinePCA('mnist\\testPCA300.csv')
    name = '3test'
    print('Training set ready, size: ', len(Xtrain))
    net = nn.NeuralNetwork(momentumSize=0)
    net.addLayer(300, 150, F.tanh, True)
    net.addLayer(150, 70, F.LReLU, True)
    net.addLayer(70, 10, F.softmax, True)
    net.setCostFunction(L.crossEntropy)
    print('Starting training...')
    net.kFoldsTrainAndValidate(Xtrain, Ytrain,
        k=6, 
        epochs=100, 
        learningRate=1e-2,
        batchSize=100,
        showError=True, 
        showNodes=False,
        print=  lambda e: _test(e, Xtest, net, name),
        showEvery=25,
        name = name)
    

if __name__ == "__main__":
    # test0()
    reduceMNIST(300)
    # test1()
    # test3()
    # test2()
    input('Click enter')
