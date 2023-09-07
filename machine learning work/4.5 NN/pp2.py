import numpy as np

def xorWB():
    b1 = np.array([-1, 3])
    b2 = np.array([-3])

    w1 = np.array([[2, -2], [2, -2]])
    w2 = np.array([[2], [2]])
    testlabels = [0, 1, 1, 0]
    weightList = []
    biasList = []
    weightList.append(w1)
    weightList.append(w2)
    biasList.append(b1)
    biasList.append(b2)
    return (weightList, biasList, testlabels)

def xnorWB():
    b1 = np.array([-1, 1])
    b2 = np.array([-1])

    w1 = np.array([[1, -1], [1, -1]])
    w2 = np.array([[2], [2]])
    testlabels = [1, 0, 0, 1]
    weightList = []
    biasList = []
    weightList.append(w1)
    weightList.append(w2)
    biasList.append(b1)
    biasList.append(b2)
    return (weightList, biasList, testlabels)

def func(n):
    return 1 if n > 0 else 0 

def printFancyT(testSet, testLabels):
    for i in range(len(testSet)):
        bigmen = testSet[i]
        val = testLabels[i]
        print("  ", end = "")
        for i in bigmen:
            print(i, "   ", end = "")
        print("| ", val, end = "")
        print()

def perceptronTest(testSet, testLabels, wList, bList):
    correct = 0
    for i in range(len(testSet)):
        pt = np.array(testSet[i]).T
        label = testLabels[i]
        newVal = network(func, pt, wList, bList)
        print("Point: ", pt, " | Expected Class: ", label, " | Classification: ", newVal)
        if(newVal == label):
            correct+=1
    print("Total Accuracy: ", str(correct/len(testSet) * 100) + "%")
    print("Weight and Bias: ", w, "|", b)

def network (A, x, w_list, b_list):
    vectorizedF = np.vectorize(A)
    ai = np.array(x)
    for i in range(len(w_list)):
        wi = w_list[i]
        bi = b_list[i]
        temp = ai@wi + bi
        ai = vectorizedF(temp)
    asdf = ai[0]
    if(asdf > .5):
        asdf = 1
    else:
        asdf = 0
    return asdf


testSet = [(0, 0), (0, 1), (1, 0), (1, 1)]
print("XOR Perceptron")
w, b, testlabels = xorWB()
printFancyT(testSet, testlabels)
perceptronTest(testSet, testlabels, w, b)

testSet = [(0, 0), (0, 1), (1, 0), (1, 1)]
print("XNOR Perceptron")
w, b, testlabels = xnorWB()
printFancyT(testSet, testlabels)
perceptronTest(testSet, testlabels, w, b)
