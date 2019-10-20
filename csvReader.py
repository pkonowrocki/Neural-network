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
            X.append(np.array([[row[0], row[1]]]).T.astype(np.float))
            Y.append(np.array([[float(row[2])]]).T)
    return X,Y
