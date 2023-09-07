from statistics import stdev
import sys
from  math  import  log
from sklearn import svm
from sklearn.datasets import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron

LAMBDA = 1
EPOCHS = 5
ALLCLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

def input():
    global w, b
    filetest = ".\iris-py-test.csv"
    filetrain = ".\iris-py-train.csv"
    train_dataset = []
    train_classLabels = []
    test_dataset = []
    test_classLabels = []
        
    with open(filetrain ,'r') as f:
        count = 0
        for line in f:
            if(count == 0):
                count+=1
                continue
            temp = line.strip().split(",")
            attributes = list(map(float, temp[:-1]))
            noob = set(attributes)
            # print(len(noob))
            theClass = ALLCLASSES.index(temp[-1])
            train_dataset.append(attributes)
            train_classLabels.append(theClass)
    
    train_dataset = np.array(train_dataset)
    train_classLabels = np.array(train_classLabels)

    with open(filetest ,'r') as f:
        count = 0
        for line in f:
            if(count == 0):
                count+=1
                continue
            temp = line.strip().split(",")
            attributes = list(map(float, temp[0:4]))
            noob = set(attributes)
            # print(len(noob))
            theClass = ALLCLASSES.index(temp[-1])
            test_dataset.append(attributes)
            test_classLabels.append(theClass)
    w = np.zeros(1+train_dataset.shape[1])
    
    test_dataset = np.array(test_dataset)
    test_classLabels = np.array(test_classLabels)
    return train_dataset, train_classLabels, test_dataset, test_classLabels

def testClf(clf, x2, y2):
    labels = clf.predict(x2)
    correct = 0
    for i in range(len(labels)):
        print("Point: ", x2[i], " | Predicted: ", labels[i], " | Actual: ", y2[i])
        if(labels[i] == y2[i]):
            correct+=1
    print("Accuracy: ", str(correct/len(labels)) + "%")

x1, y1, x2, y2 = input()
clf = Perceptron()
clf.fit(x1, y1)
testClf(clf, x2, y2)


# perceptronTrain(x1, y1)
# perceptronTest(x2, y2)
# perceptronTest(x1, y1)
