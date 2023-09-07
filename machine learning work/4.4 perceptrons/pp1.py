from statistics import stdev
import sys
from  math  import  log
from sklearn import svm
from sklearn.datasets import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split

LAMBDA = 1
EPOCHS = 5
w = np.array([0,0])
b = 0
show = True

def newInput():
    blobsX, blobsY = make_blobs(n_samples=100, centers=2, n_features=2)
    trainX, testX, trainY, testY = train_test_split(blobsX, blobsY, test_size=0.33)
    with open('savedPoints.txt', 'w') as convert_file:
        stringVal = ""
        for i in range(len(blobsX)):
            stringVal = str(blobsX[i][0]) + "," + str(blobsX[i][1]) + "," + str(blobsY[i])
            convert_file.write(stringVal + "\n")

    df = DataFrame(dict(x=blobsX[:,0], y=blobsX[:,1], label=blobsY))
    colors = {0:'red', 1:'blue'}
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    if(show == True):
        pyplot.show()

    return trainX, trainY, testX, testY

def getOldInput():
    blobsX = []
    blobsY = []
    with open('savedPoints.txt') as f:
        for line in f:
            indexData = line.strip().split(",")
            blobsX.append(list(map(float, indexData[:-1])))
            blobsY.append(int(indexData[-1]))
    blobsX = np.array(blobsX)
    blobsY = np.array(blobsY)
    df = DataFrame(dict(x=blobsX[:,0], y=blobsX[:,1], label=blobsY))
    colors = {0:'red', 1:'blue'}
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    if(show == True):
        pyplot.show()

    trainX, testX, trainY, testY = train_test_split(blobsX, blobsY, test_size=0.33)
    return trainX, trainY, testX, testY

def hardLimitFunc(n):
    if(n > 0):
        return 1
    else:
        return 0
    
def func2(n):
    if(n != 0):
        return 1
    else:
        return 0

def perceptronTrain(trainSet, trainLabels):
    global w, b
    stabilizeNum = len(trainSet)
    countStable = 0
    for epochs in range(EPOCHS):
        for i in range(len(trainSet)):
            pt = np.array(trainSet[i]).T
            label = trainLabels[i]
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
    correct = 0
    for i in range(len(testSet)):
        pt = np.array(testSet[i]).T
        label = testLabels[i]
        val = np.matmul(w, pt) + b
        newVal = hardLimitFunc(val)
        print("Point: ", pt, " | Expected Class: ", label, " | Classification: ", newVal)
        if(newVal == label):
            correct+=1
    print("Total Accuracy: ", str(correct/len(testSet) * 100) + "%")
    print("Weight and Bias: ", w, "|", b)
    return correct/len(testSet) * 100


# x1, y1, x2, y2 = newInput()

x1, y1, x2, y2 = getOldInput()
perceptronTrain(x1, y1)
val = perceptronTest(x2, y2)
total = val
show = False
for i in range(9):
    x1, y1, x2, y2 = getOldInput()
    perceptronTrain(x1, y1)
    val = perceptronTest(x2, y2)
    total += val
print("Average Accuracy over 5 random test and train sets: ", str(total/10) + "%")
