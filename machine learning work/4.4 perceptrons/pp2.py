from statistics import stdev
import sys
from  math  import  log
from sklearn import svm
from sklearn.datasets import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
import math
import pickle

LAMBDA = 1
EPOCHS = 5
w = [] #changes when it gets the input
b = []
ALLCLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
classDict = {"Iris-setosa": -1,
             "Iris-versicolor": 0,
             "Iris-virginica": 1
             }

def initWeights():
    global w, b
    # w = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    # b = [0, 0, 0]
    w = [0, 0, 0, 0]
    b = 0

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
            theClass = classDict[temp[-1]]
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
            theClass = classDict[temp[-1]]
            test_dataset.append(attributes)
            test_classLabels.append(theClass)
    
    test_dataset = np.array(test_dataset)
    test_classLabels = np.array(test_classLabels)
    return train_dataset, train_classLabels, test_dataset, test_classLabels

def hardLimitFunc(n):
    if(n == 0):
        return 0
    elif(n > 0):
        return 1
    else:
        return -1


def perceptronTrain(trainSet, trainLabels):
    global w, b
    stabilizeNum = len(trainSet)
    countStable = 0
    for epochs in range(EPOCHS):
        for i in range(len(trainSet)):
            pt = np.array(trainSet[i]).T
            label = trainLabels[i]
            # print("weight: ", w)
            # print("point: ", pt)
            val = np.matmul(w, pt) + b
            newVal = hardLimitFunc(val)
            if(newVal != label):
                countStable = 0
                error = label - newVal
                w = w + (error * LAMBDA * pt.T)
                b = b + (error * LAMBDA)
            else:
                countStable += 1
            if(countStable >= stabilizeNum):
                break
    print(w, b)

def perceptronTest(testSet, testLabels):
    global w, b
    correct = 0
    for i in range(len(testSet)):
        pt = np.array(testSet[i]).T
        label = testLabels[i]
        print("weight: ", w)
        print("point: ", pt)
        val = np.matmul(w, pt) + b
        newVal = hardLimitFunc(val)
        print("Point: ", pt, " | Expected Class: ", label, " | Classification: ", newVal)
        if(newVal == label):
            correct+=1
    print("Total Accuracy: ", str(correct/len(testSet) * 100) + "%")
    print("Classes are the following: Iris-setosa is -1, Iris-versicolor is 0, and Iris-virginica 1")
    print("Weight and Bias: ", w, "|", b)


x1, y1, x2, y2 = input()
initWeights()
perceptronTrain(x1, y1)
perceptronTest(x2, y2)
# perceptronTest(x1, y1)
