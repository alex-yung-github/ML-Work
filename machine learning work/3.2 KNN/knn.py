import numpy as np
import math
import json
import random
import statistics

train_dataset = []
train_classLabels = []

test_dataset = []
test_classLabels = []

ALLCLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
testDict = {}
testIndex = 0
indexOrderforTrain = []
counter = 0
N = 12 #number of neighbors

def getData():
    global train_dataset, train_classLabels, test_dataset, test_classLabels
    filetest = ".\iris-py-test.csv"
    filetrain = ".\iris-py-train.csv"
        
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
            theClass = temp[-1]
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
            attributes = list(map(float, temp[:-1]))
            noob = set(attributes)
            # print(len(noob))
            theClass = temp[-1]
            test_dataset.append(attributes)
            test_classLabels.append(theClass)
    
    test_dataset = np.array(test_dataset)
    test_classLabels = np.array(test_classLabels)



def printData():
    print("TRAIN: ")
    print("Dataset instances w/o class label:")
    print(train_dataset)
    print("Shape: ", train_dataset.shape)
    print()
    print("Class Labels: ")
    print(train_classLabels)
    print("Shape: ", train_classLabels.shape)
    print("--------------------------------------------")

    print()
    print("TEST: ")
    print("Dataset instances w/o class label:")
    print(test_dataset)
    print("Shape: ", test_dataset.shape)
    print()
    print("Class Labels: ")
    print(test_classLabels)
    print("Shape: ", test_classLabels.shape)
    print()

def getDistance(x1, x2): #finds euclidian distance)
    totalDist = 0
    for i in range(len(x1)):
        totalDist += (x1[i] - x2[i]) ** 2
    totalDist = math.sqrt(totalDist)
    return totalDist

def get_neighbors(train, trainLabels, test_row, num_neighbors):
    distances = list()
    for train_index in range(len(train)):
        trainVal = train[train_index]
        dist = getDistance(test_row, trainVal)
        distances.append((dist, trainLabels[train_index]))
    distances.sort()
    # print(distances)
    distances = distances[0:num_neighbors]
    # print(distances)
    return distances

def classify():
    global train_classLabels, train_dataset, test_dataset
    classification = []
    for i in test_dataset:
        # print(i)
        tempNeighbors = get_neighbors(train_dataset, train_classLabels, i, N)
        nearClasses = []
        for x in tempNeighbors:
            nearClasses.append(x[1])
        decidingList = []
        for i in ALLCLASSES:
            decidingList.append(nearClasses.count(i))
        print(decidingList)
        best = max(decidingList)
        indexofBest = decidingList.index(best)
        classification.append(ALLCLASSES[indexofBest])
    return classification

def checkAccuracy(test_labels, classifications, data):

    classDict = dict()
    confusionMatrixDict = dict()
    for i in range(len(ALLCLASSES)):
        classDict[ALLCLASSES[i]] = i
        confusionMatrixDict[i] = []

    correct = 0
    for i in range(len(data)):
        actualLabel = test_labels[i]
        label = classifications[i]
        if(label == actualLabel):
            correct += 1
        confusionMatrixDict[classDict[actualLabel]].append(classDict[label])
        print("Correct Label: ", actualLabel, " | Actual Label: ", label)
    
    print("Correct: ", correct, " | total", len(data))
    print("Percentage Correct: ", correct/len(data))

    for i in range(len(ALLCLASSES)):
        print(ALLCLASSES[i] + "(" + str(i) + ")",  end = "  ")
    print()
    print("     ", "0  1  2")
    precisions, recalls = macroAverage(confusionMatrixDict)
    for i in confusionMatrixDict:
        temp = confusionMatrixDict[i]
        print(i, "   ", temp.count(0), temp.count(1), temp.count(2), "Precision:", precisions[i] )
    print("Recalls:  ", recalls[0], recalls[1], recalls[2])

    print("Macroaverage Precision:", statistics.mean(precisions))
    print("Macroaverage Recall:", statistics.mean(recalls))

    mA = microAverage(confusionMatrixDict)
    totalSum = sum(mA)
    pooledPrecision = mA[0]/ (mA[0] + mA[2])
    pooledRecall = mA[0] / (mA[0] + mA[3])
    print("Microaverage Precision: ", pooledPrecision)
    print("Microaverage Recall: ", pooledRecall)
        
def macroAverage(confusionMatrixDict):
    count = 0        
    precisions = []
    colList = []
    allRows = []
    for i in confusionMatrixDict:
        tempRowList = []
        temp = confusionMatrixDict[i]
        for x in range(len(ALLCLASSES)):
            tempRowList.append(temp.count(x))
        total = sum(tempRowList)
        precisions.append(tempRowList[i]/total)
        allRows.append(tempRowList)
    
    recalls = []
    count = 0
    for colData in zip(*allRows):
        total = sum(colData)
        val = colData[count]/total
        recalls.append(round(val, 2))
        count+=1

    return (precisions, recalls)

def microAverage(confusionMatrixDict):
    tempHolder = []
    for i in range(len(ALLCLASSES)):
        truePos = confusionMatrixDict[i].count(i)
        trueNeg = 0
        for x in range(len(ALLCLASSES)):
            if(x != i):
                val =  (len(confusionMatrixDict[x]) - confusionMatrixDict[x].count(i))
                trueNeg += val
        falsePos = 0
        for x in range(len(ALLCLASSES)):
            if(x != i):
                val = confusionMatrixDict[i].count(x)
                falsePos += val

        falseNeg = 0
        for x in range(len(ALLCLASSES)):
            if(x != i):
                for z in range(len(ALLCLASSES)):
                    if(z == i):
                        val = confusionMatrixDict[x].count(z)
                        falseNeg += val

        tempHolder.append([truePos, trueNeg, falsePos, falseNeg])

    pooledVals = [0, 0, 0, 0]
    for vals in tempHolder:
        for ind in range(len(vals)):
            pooledVals[ind] += vals[ind]
    return pooledVals
        

        
getData()
classifications = classify()
checkAccuracy(test_classLabels, classifications, test_dataset)
# printData()


