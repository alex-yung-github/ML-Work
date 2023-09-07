import numpy as np
import math
import json
import random


dataset = []

# classLabels = trainClassLabelsforDebug
classLabels = []
ALLCLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
testDict = {}
testIndex = 0
indexOrderforTrain = []
counter = 0

N = 5 #only works for 4 or 5 bins; number of bins for each var
testingDictionary = dict()

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
    
    dataset = np.array(dataset)
    classLabels = np.array(classLabels)
    # print(testDict)


def printData():
    print("Dataset instances w/o class label:")
    print(dataset)
    print("Shape: ", dataset.shape)
    print()
    print("Class Labels: ")
    print(classLabels)
    print("Shape: ", classLabels.shape)
    print()


def findEntropyBefore(data):
    tempdict = {}
    for i in data:
        # print("BIG I: ", i)
        if(i not in tempdict.keys()):
            tempdict[i] = 1
        else:
            tempdict[i] += 1

    entropy = 0
    for i in tempdict:
        entropy += (tempdict[i]/len(data)) * math.log2(tempdict[i]/len(data))
    entropy *= -1

    return entropy

def bin():
    global dataset, classLabels
    var1 = []
    var2 = []
    var3 = []
    var4 = []
    allvars = [var1, var2, var3, var4]
    for i in dataset:
        for var in range(len(allvars)):
            allvars[var].append(i[var])

    allBounds = []
    intBounds = []
    for var in allvars:
        #equal depth bins
        var.sort()
        length = len(var)
        depth = int(length/N)
        bounds = []
        tempintbounds = []
        place = 0
        for i in range(N):
            temp = var[place:place+depth]
            minimum = min(temp)
            if(i + 1 == N):
                maximum = round(max(temp)+.1, 1) #add .1 so that last bin includes all.
            else:
                maximum = round(max(temp), 1)
            bound = str(minimum) + " to "  + str(maximum)
            bounds.append(bound)
            tempintbounds.append((minimum, maximum))
            place = place + depth
        allBounds.append(bounds)
        intBounds.append(tempintbounds)
        # print(allBounds)
        
    newDataset = []
    
    for i in range(len(dataset)):
        vars = dataset[i]
        newbins = []
        for t in range(len(vars)):
            binClass = intBounds[t]
            stringbinClass = allBounds[t]
            tempvar = vars[t]
            for x in range(len(binClass)):
                # print(binClass)
                # print(tempvar)
                if(tempvar >= binClass[x][0] and tempvar <= binClass[x][1]):
                    newbins.append(stringbinClass[x])
                    break

        newDataset.append(newbins)
    return (newDataset, allBounds)

def binTest(boundings):
    global dataset
    
    intBounds = []
    allBounds = []
    for bigbound in boundings:
        tempbounds = []
        tempAllBound = []
        for i in bigbound:
            tempAllBound.append(i)
            thing1 = i[0:3].strip()
            thing2 = i[-4:].strip()
            # print(thing1, thing2)
            tempbounds.append((float(thing1), float(thing2)))
        intBounds.append(tempbounds)
        allBounds.append(tempAllBound)
    
    # print(intBounds)
    # print(allBounds)
    
    newData = []
    for i in range(len(dataset)):
        vars = dataset[i]
        dataPoint = []
        for part in range(len(vars)):
            tempBounds = intBounds[part]
            tempVar = vars[part]
            for temp in range(len(tempBounds)):
                if(tempVar >= tempBounds[temp][0] and tempVar <= tempBounds[temp][1]):
                    dataPoint.append(allBounds[part][temp])
                    break
        newData.append(dataPoint)
    # print(newData)
            
    return newData

def findEntropyAfter(nodes):
    weights = []
    totalLength = 0
    for i in nodes:
        totalLength += len(i)

    for i in nodes:
        weight = len(i)/totalLength
        weights.append(weight)

    entropyAfter = 0
    for i in range(len(nodes)):
        entropyAfter += (weights[i] * findEntropyBefore(nodes[i]))

    return entropyAfter

def infoGain(entropyBefore, entropyAfter):
    return entropyBefore - entropyAfter

def process(listt):
    stuff = []
    for i in ALLCLASSES:
        for d in listt:
            if(d[0] == i):
                stuff.append(i)
                break
    return stuff


def sortEntropy(parentDataLabels, data, bounds):
    totalOrder = []
    for i in range(len(bounds)):
        storage = dict()
        entropyCalcLabelStorage = dict()
        tempBounds = bounds[i]
        for x in tempBounds:
            storage[x] = []
            entropyCalcLabelStorage[x] = []
        for dataindex in range(len(data)):
            dataAttr = data[dataindex][i]
            tempClass = parentDataLabels[dataindex]
            # if(dataAttr not in storage):
            #     storage[dataAttr] = []
            #     entropyCalcLabelStorage[dataAttr] = []
            storage[dataAttr].append((tempClass, dataindex))
            entropyCalcLabelStorage[dataAttr].append(tempClass)
        nodes = []
        for x in entropyCalcLabelStorage:
            nodes.append(entropyCalcLabelStorage[x])
        entropyAfter = findEntropyAfter(nodes)
        entropyBefore = findEntropyBefore(parentDataLabels)
        infoGain = entropyBefore - entropyAfter
        totalOrder.append((infoGain, i, storage))
    totalOrder.sort(reverse=True)
    return totalOrder
    # print("Total order: ", totalOrder)
    # print(ruleset)

def getNextIndex(totalOrder, usedindices):
    index = None
    store = None
    for i in totalOrder:
        if(i[1] not in usedindices):
            index = i[1]
            store = i
            break
    return index, store

def createTree(data, initlabels, usedindices, bounds):
    global indexOrderforTrain, counter
    ruleset = {}
    # print(data)
    totalOrder = sortEntropy(initlabels, data, bounds)
    index, store = getNextIndex(totalOrder, usedindices)
    if(index == None):
        if(counter == 0):
            indexOrderforTrain = usedindices
            counter+=1
        tempthingforclasses = []
        for a in ALLCLASSES:
            tempthingforclasses.append((initlabels.count(a), a))
        tempVal = max(tempthingforclasses)
        return tempVal[1]
    newUsedIndices = usedindices
    newUsedIndices.append(index)
    tempbounds =  bounds[index]
    store = store[2]
    for i in range(len(tempbounds)):
        b = tempbounds[i]
        tempStore = store[b]
        if(len(tempStore) == 0):
            ruleset[b] = random.choice(ALLCLASSES)	
            continue
        classTuple = getEachClass(tempStore)
        # print(classTuple)
        tempValPure = getIfPure(classTuple)
        if(tempValPure is not None):
            ruleset[b] = ALLCLASSES[tempValPure]
        else:
            nextInitLabels = getNextLabels(tempStore)
            nextData = getNextData(tempStore, data)
            ruleset[b] = createTree(nextData, nextInitLabels, newUsedIndices, bounds)
    return ruleset

def getNextLabels(tStore):
    toReturnList = []
    for i in tStore:
        toReturnList.append(i[0])
    return toReturnList

def getNextData(tStore, originalData):
    toReturnList = []
    for i in tStore:
        index = i[1]
        vals = originalData[index]
        toReturnList.append(vals)
    return toReturnList
        

def getEachClass(listOfTuples):
    totalList = [0 for i in ALLCLASSES]
    for i in listOfTuples:
        tempClass = i[0]
        ind = ALLCLASSES.index(tempClass)
        totalList[ind] = totalList[ind] + 1
    return totalList

def getIfPure(classtup):
    index = None
    for i in range(len(classtup)):
        num = classtup[i]
        if(num != 0):
            if(index == None):
                index = i
            else:
                return None
    return index

def printTotalOrder(totalOrder):
    count = 0
    for i in totalOrder:
        print(count, " : ", i)
        print()
        count+=1
    
def saveRuleset(ruleset, bounds):
    with open('savedDataDTKey.txt', 'w') as convert_file:
        convert_file.write(json.dumps(ruleset))
    with open('savedDataDTIndex.txt', 'w') as convert_file:
        stringVal = ""
        for i in indexOrderforTrain:
            stringVal += str(i) + " "
        convert_file.write(stringVal + "\n")
    print(bounds)
    with open('savedDataDTBounds.txt', 'w') as convert_file:
        for i in bounds:
            for x in i:
                convert_file.write(str(x) + "\n")

def getTestingDict():
    global testingDictionary
    indexData = []
    count = 0
    with open('savedDataDTKey.txt') as f:
        data = f.read()
        testingDictionary = json.loads(data)
    with open('savedDataDTIndex.txt') as f:
        for line in f:
            indexData = line.strip().split(" ")
    # reconstructing the data as a dictionary
    return indexData

def runDecisionTrees(data, indexData, testKey):
    chosenLabels = []
    # print(data)
    for i in data:
        tempthing = testKey
        while(tempthing not in ALLCLASSES):
            tempC = 0
            for x in i:
                if(x in tempthing.keys()):
                    tempC = x
                    break

            tempthing = tempthing[tempC]
        chosenLabels.append(tempthing)
    print(chosenLabels)
    checkAccuracy(chosenLabels, data)

def getBounds():
    bounds = []
    with open('savedDataDTBounds.txt') as f:
        count = 0
        tempBounds = []
        for line in f:
            indexData = line.strip()
            if(count < N):
                tempBounds.append(indexData)
                count += 1
            else:
                count = 1
                bounds.append(tempBounds)
                tempBounds = [indexData]
        bounds.append(tempBounds)
    return bounds

def checkAccuracy(testLabels, data):
    
    classDict = dict()
    confusionMatrixDict = dict()
    for i in range(len(ALLCLASSES)):
        classDict[ALLCLASSES[i]] = i
        confusionMatrixDict[i] = []

    correct = 0
    for i in range(len(data)):
        actualLabel = classLabels[i]
        label = testLabels[i]
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
    for i in confusionMatrixDict:
        temp = confusionMatrixDict[i]
        print(i, "   ", temp.count(0), temp.count(1), temp.count(2))


userInput = input("train or test on iris dataset (insert 'train' or 'test'): ")
if(userInput != "train" and  userInput != "test"):
    print("Assuming you meant train...")
    userInput = "train"
getData(userInput)
if(userInput == "train"):
    newData, bounds = bin()
    ruleset = createTree(newData, classLabels, [], bounds)
    # ruleset = createTree(trainDataforDebug, trainClassLabelsforDebug, [], trainBoundsforDebug)
    saveRuleset(ruleset, bounds)
elif(userInput == "test"):
    boundings = getBounds()
    newData = binTest(boundings)
    indexData = getTestingDict()
    print(boundings)
    runDecisionTrees(newData, indexData, testingDictionary)
    # print("index", indexData)
    # print(testingDictionary)

    # print("[", end = '')
    # for i in classLabels:
    #     print("'" + i + "'", ", ", end = '')
    # print("]")