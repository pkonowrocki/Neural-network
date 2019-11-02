import csv
import numpy as np

def readClassificationFile(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        X = []
        Y = []
        for row in csv_reader:
            X.append(np.array([[row[0], row[1]]]).T.astype(np.float))
            if int(row[2])==1:
                Y.append(np.array([[1]]).T)
            else:
                Y.append(np.array([[0]]).T)
    return X,Y

def readClassification3ClassesFile(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        X = []
        Y = []
        for row in csv_reader:
            X.append(np.array([[row[0], row[1]]]).T.astype(np.float))
            if int(row[2])==1:
                Y.append(np.array([[1, 0, 0]]).T)
            elif int(row[2])==2:
                Y.append(np.array([[0, 1, 0]]).T)
            else:
                Y.append(np.array([[0, 0, 1]]).T)
    return X,Y

def readRegressionFile(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        X = []
        Y = []
        for row in csv_reader:
            X.append(np.array([[row[0]]]).T.astype(np.float))
            Y.append(np.array([[float(row[1])]]).T)
    return X,Y

def readmnistAsOneLineTraining(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        X = []
        Y = []
        for row in csv_reader:
            y = np.zeros((10,1))
            y[int(row[0])-1,0] = 1
            Y.append(y)
            X.append((np.array([row[1:785]]).T.astype(float))/255)
    return X,Y

def readmnistAsOneLineTest(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        X = []
        for row in csv_reader:
            X.append((np.array([row[0:784]]).T.astype(float))/255)
    return X

def readmnistAsOneLinePCA(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        X = []
        for row in csv_reader:
            X.append((np.array([row[0:784]]).T.astype(float)))
    return X

def saveMnistResult(path, Y):
    with open(path, 'w', newline='') as csvfile:
        fields = ['ImageId', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for i in range(len(Y)):
           writer.writerow({'ImageId' : i+1, 'Label' : np.argmax(Y[i])}) 

def readmnistAsPictureTraining(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        X = []
        Y = []
        for row in csv_reader:
            y = np.zeros((10,1))
            y[int(row[0]),0] = 1
            Y.append(y)
            X.append(np.array([row[1:785]]).T.astype(int).reshape((28,28)))
    return X,Y