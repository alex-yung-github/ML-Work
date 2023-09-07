import numpy as np

def andWB():
    w = np.array([1, 1])
    b = [-1.5]
    testlabels = [0, 0, 0, 1]
    return (w, b, testlabels)

def orWB():
    w = np.array([2, 2])
    b = -1
    testlabels = [0, 1, 1, 1]
    return (w, b, testlabels)

def nandWB():
    w = np.array([-1,-1])
    b = 2
    testlabels = [1, 1, 1, 0]
    return (w, b, testlabels)

def threeInpOR():
    w = np.array([2,2,2])
    b = -1
    testLabels = [0,1,1,1,1,1,1,1]
    return (w, b, testLabels)

def printFancyT(testSet, testLabels):
    for i in range(len(testSet)):
        bigmen = testSet[i]
        val = testLabels[i]
        print("  ", end = "")
        for i in bigmen:
            print(i, "   ", end = "")
        print("| ", val, end = "")
        print()

def hardLimitFunc(n):
    if(n > 0):
        return 1
    else:
        return 0

def perceptronTest(testSet, testLabels, w, b):
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


# testSet = [(0, 0), (0, 1), (1, 0), (1, 1)]
# print("And Perceptron")
# w, b, testlabels = andWB()
# printFancyT(testSet, testlabels)
# perceptronTest(testSet, testlabels, w, b)

# testSet = [(0, 0), (0, 1), (1, 0), (1, 1)]
# print("Or Perceptron")
# w, b, testlabels = orWB()
# printFancyT(testSet, testlabels)
# perceptronTest(testSet, testlabels, w, b)

# testSet = [(0, 0), (0, 1), (1, 0), (1, 1)]
# print("Nand Perceptron")
# w, b, testlabels = nandWB()
# printFancyT(testSet, testlabels)
# perceptronTest(testSet, testlabels, w, b)

testSet = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
print("3 Input OR Perceptron")
w, b, testlabels = threeInpOR()
printFancyT(testSet, testlabels)
perceptronTest(testSet, testlabels, w, b)