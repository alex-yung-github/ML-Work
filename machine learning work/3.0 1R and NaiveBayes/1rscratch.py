import numpy as np


dataset = []
classLabels = []
ALLCLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
testDict = {}
testIndex = 0

N = 5 #only works for 4 or 5 bins; number of bins for each var

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
        with open("savedData1R.txt", 'r') as f:
            count =0 
            for line in f:
                if(count == 0):
                    count +=1
                    testIndex = int(line.strip())
                    continue
                temp = line.strip().split(",")
                # print(temp)
                testDict[temp[0]] = temp[1]
    
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

def binTest():
    intBounds = []
    allBounds = []
    for i in testDict:
        allBounds.append(i)
        thing1 = i[0:3].strip()
        thing2 = i[-4:].strip()
        # print(thing1, thing2)
        intBounds.append((float(thing1), float(thing2)))
    
    newData = []
    for i in range(len(dataset)):
        vars = dataset[i]
        tempvar = vars[testIndex]
        for x in range(len(intBounds)):
            if(tempvar >= intBounds[x][0] and tempvar <= intBounds[x][1]):
                newData.append(allBounds[x])
                break
    return(newData)

def train1r(data, allBounds):
    global classLabels
    # yes = []
    # for i in data:
    #     temp = len(i)
    #     yes.append(temp)
    # print(yes)
    # print(data[24])
    # print(allBounds)
    allBoundErrors = []
    for modifier in range(len(allBounds)):
        # print(modifier)
        boundErrors = []
        for bound in allBounds[modifier]:
            tempdict = {}
            for i in range(len(data)):
                # print(i, modifier)
                # print(data[i][modifier])
                tempattribute = data[i][modifier]
                # print(tempattribute, bound)
                if(tempattribute == bound):
                    templabel = classLabels[i]
                    if(templabel in tempdict.keys()):
                        tempdict[templabel] = tempdict[templabel] +1
                    else:
                        tempdict[templabel] = 1
            tempBoundResults = []
            for i in ALLCLASSES:
                if(i in tempdict):
                    tempBoundResults.append(tempdict[i])
                else:
                    tempBoundResults.append(0)
            tempBoundResults.append(sum(tempBoundResults))
            boundErrors.append(tempBoundResults)
        allBoundErrors.append(boundErrors)
    # print(allBoundErrors)

    totalDict = {}
    overallError = float('inf')
    index = 0
    for i in range(len(allBoundErrors)):
        stats = allBoundErrors[i]
        temperror = 0
        tempdict = {}
        for l in range(len(stats)):
            templist = stats[l].copy()
            templist = templist[:-1]
            temp = max(templist)
            end = stats[l][-1]
            place = stats[l].index(temp)
            classChosen = ALLCLASSES[place]
            temperror += float(1-(temp/end))
            tempdict[allBounds[i][l]] = classChosen
        if(temperror < overallError):
            totalDict = tempdict
            overallError = temperror
            index = i
    #         print(totalDict)
    # print(totalDict)
    # print(overallError)
    saveData(totalDict, index, overallError)

def saveData(dict, index, error):
    with open("savedData1R.txt",'w') as f:
        f.write(str(index))
        f.write('\n')
        for i in dict:
            f.write(i + "," + dict[i])
            f.write("\n")
        # f.write(str(error))
        f.close()

def run1r(data):

    classDict = dict()
    confusionMatrixDict = dict()
    for i in range(len(ALLCLASSES)):
        classDict[ALLCLASSES[i]] = i
        confusionMatrixDict[i] = []

    correct = 0
    for i in range(len(data)):
        actualLabel = classLabels[i]
        label = testDict[data[i]]
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
    train1r(newData, bounds)
elif(userInput == "test"):
    newData = binTest()
    run1r(newData)

