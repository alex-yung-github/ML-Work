import numpy as np
import pandas as pd
import math

dataset = []
classLabels = []
ALLCLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
testDict = {}
testIndex = 0

def getData(userinput):
    global dataset, classLabels, testDict, testIndex
    file = ""
    if(userinput == "test"):
        file = ".\iris-py-test.csv"
    elif(userinput == "train"):
        file = ".\iris-py-train.csv"
        
    with open(file ,'r') as f:
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
            dataset.append(attributes)
            classLabels.append(theClass)
    
    if(userinput == "test"):
        with open("savedData.txt", 'r') as f:
            count =0 
            for line in f:
                if(count == 0):
                    count +=1
                    testIndex = int(line.strip())
                    continue
                temp = line.strip().split(",")
                # print(temp)
                testDict[temp[0]] = temp[1]

def split():
    global dataset, label
    splitData = dict()
    for i in range(len(dataset)):
        temp = dataset[i]
        label = classLabels[i]
        if(label not in splitData.keys()):
            splitData[label] = []
        splitData[label].append(temp)
    return splitData
    
def statistics(data):
    allStats = {}
    for flower in data:
        allStats[flower] = []

    for flower in data:
        for colData in zip(*data[flower]):
            # print(colData)
            stats = []
            mean = sum(colData)/float(len(colData))
            std = math.sqrt(sum([(x-mean)**2 for x in colData]) / float(len(colData)-1))
            length = len(colData)
            stats.append(mean)
            stats.append(std)
            stats.append(length)
            allStats[flower].append(stats)
    
    saveData(allStats)
    # print(allStats)  
    return allStats

def saveData(stats):
    with open("savedDataNaiveBayes.txt",'w') as f:
        for i in stats:
            f.write(i)
            for l in range(len(stats[i])):
                f.write("," + ",".join(list(map(str, stats[i][l]))))
            f.write("\n")
        # f.write(str(error))
        f.close()

def probabilityDensityFunc(n, mean, std):
    exponent = np.exp(-((n-mean)**2 / (2 * std**2 )))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

def generatePredictions(classStats):
    global dataset

    predictions = []
    for item in dataset:
        temppredictions = []
        for i in classStats:
            totalVal = 0
            for statplace in range(len(classStats[i])):
                freq = len(classStats[i][-1])/len(dataset)
                totalVal += math.log(freq)
                stat = classStats[i][statplace]
                totalVal += math.log(probabilityDensityFunc(item[statplace], stat[0], stat[1]))
            temppredictions.append((totalVal, i))
        predictions.append(max(temppredictions))
    
    onlypredictions = []
    for i in predictions:
        onlypredictions.append(i[1])
    return onlypredictions

def getTestStats():

    predictionDict = {}
    with open("savedDataNaiveBayes.txt", 'r') as f:
        for line in f:
            temp = line.strip().split(",")
            cLabel = temp[0]
            predictionDict[cLabel] = []
            for i in range(1,len(temp), 3):
                predictionDict[cLabel].append(list(map(float, temp[i:i+3])))
    return predictionDict
            # print(temp)

def checkPerformance(predictions):
    correct = 0
    classDict = dict()
    confusionMatrixDict = dict()
    for i in range(len(ALLCLASSES)):
        classDict[ALLCLASSES[i]] = i
        confusionMatrixDict[i] = []

    for i in range(len(predictions)):
        label = predictions[i]
        actualLabel = classLabels[i]
        if(label == actualLabel):
            correct += 1
        confusionMatrixDict[classDict[actualLabel]].append(classDict[label])
        # print("Correct Label: ", actualLabel, " | Actual Label: ", label)
    
    print("Correct: ", correct, " | total", len(predictions))
    print("Percentage Correct: ", correct/len(predictions))
    for i in range(len(ALLCLASSES)):
        print(ALLCLASSES[i] + "(" + str(i) + ")",  end = "  ")
    print()
    print("     ", "0  1  2")
    for i in confusionMatrixDict:
        temp = confusionMatrixDict[i]
        print(i, "   ", temp.count(0), temp.count(1), temp.count(2))




# mean = sum(colData)/float(len(colData))
# avg = mean(colData)
# var = math.sqrt(sum([(x-avg)**2 for x in colData]) / float(len(colData)-1))

userInput = input("train or test on iris dataset (insert 'train' or 'test'): ")
if(userInput != "train" and  userInput != "test"):
    print("Assuming you meant train...")
    userInput = "train"
getData(userInput)
if(userInput == "train"):
    split_data = split()
    stats = statistics(split_data)
elif(userInput == "test"):
    st = getTestStats()
    predictions = generatePredictions(st)
    checkPerformance(predictions)



