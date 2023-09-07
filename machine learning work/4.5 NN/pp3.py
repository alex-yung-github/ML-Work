import numpy as np
import itertools
from sklearn.neural_network import MLPClassifier

def specialWB():
    b1 = np.array([-2,-3,-2,-2,-4,-3])
    b2 = np.array([-.5])

    w1=np.array([[-1,-1,-1,1,1,1],
                 [-1,1,1,-1,-1,1],
                 [1,-1,1,-1,1,-1],
                 [1,1,-1,-1,1,-1],
                 [-1,1,-1,1,1,1]]).T
    w2 = np.array([1,1,1,1,1,1])
    testlabels = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    weightList = []
    biasList = []
    weightList.append(w1)
    weightList.append(w2)
    biasList.append(b1)
    biasList.append(b2)
    return (weightList, biasList, testlabels)


def func(n):
    return 1 if n >= 0 else 0

def truthT(bits, n):
    tempN = bin(n)
    binaryNum = tempN[2:]
    while(len(binaryNum) < 2**bits):
        binaryNum = "0" + binaryNum
    # print(binaryNum)
    toReturn = []
    zeTruth = list(itertools.product([0, 1], repeat=bits))
    zeTruth.sort(reverse = True)

    for i in range(len(binaryNum)):
        val = binaryNum[i:i+1]
        bigman = zeTruth[i]
        toReturn.append((bigman, int(val)))
    return toReturn

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
        pt = np.array(testSet[i])
        label = testLabels[i]
        # print("anotha one")
        newVal = network(func, pt, wList, bList)
        print("Point: ", pt, " | Expected Class: ", label, " | Classification: ", newVal)
        if(newVal == label):
            correct+=1
    print("Total Accuracy: ", str(correct/len(testSet) * 100) + "%")
    print("Weight and Bias: ", w, "|", b)

def network (A, x, w_list, b_list):
    vectorizedF = np.vectorize(A)
    ai = np.array(x).transpose()
    for i in range(len(w_list)):
        wi = w_list[i]
        bi = b_list[i]
        temp = np.matmul(wi, ai) + bi
        ai = vectorizedF(temp)
        # print(ai)
    return ai

def perpTest2(clf, x, y):
    correct = 0
    for i in range(len(x)):
        tempLabel = clf.predict([x[i]])
        print("Point: ", x[i], " | Expected Class: ", y[i], " | Classification: ", tempLabel)
        if(tempLabel == y[i]):
            correct+=1
    print("Total Accuracy: ", str(correct/len(testSet) * 100) + "%")
    print("Weight and Bias: ", w, "|", b)
        

def trainWeightsandBiases(testSet, testlabels):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)
    clf.fit(testSet, testlabels)
    correct = 0
    for i in range(len(testSet)):
        label = clf.predict([testSet[i]])
        actualLabel = testlabels[i]
        print("Point: ", testSet[i], " | Expected Class: ", label, " | Classification: ", actualLabel)
        if(label == actualLabel):
            correct += 1
    print("Total Accuracy w/Scikit learn w & b: ", str(correct/len(testSet) * 100) + "%")
# for the inputs
# thing = truthT(5, 4)
# print("[", end = '')
# indexes = []
# thing.reverse()
# for i in thing:
#     if(i[0] == (0,0,1,1,0) or i[0] == (0,1,0,1,1) or i[0] == (0,1,1,0,0) or i[0] == (1,0,0,0,1) or i[0] == (1,0,1,1,1) or i[0] == (1,1,0,0,1)):
#         indexes.append(thing.index(i))
#     print(i[0], end = ', ')
# print("]")
# print("[", end = '')
# for i in range(len(thing)):
#     print(1, end = ", ") if i in indexes else print(0, end = ", ")
# print("]")
testSet = [(0, 0, 0, 0, 0), (0, 0, 0, 0, 1), (0, 0, 0, 1, 0), (0, 0, 0, 1, 1), (0, 0, 1, 0, 0), (0, 0, 1, 0, 1), (0, 0, 1, 1, 0), (0, 0, 1, 1, 1), (0, 1, 0, 0, 0), (0, 1, 0, 0, 1), (0, 1, 0, 1, 0), (0, 1, 0, 1, 1), (0, 1, 1, 0, 0), (0, 1, 1, 0, 1), (0, 1, 1, 1, 0), (0, 1, 1, 1, 1), (1, 0, 0, 0, 0), (1, 0, 0, 0, 1), (1, 0, 0, 1, 0), (1, 0, 0, 1, 1), (1, 0, 1, 0, 0), (1, 0, 1, 0, 1), (1, 0, 1, 1, 0), (1, 0, 1, 1, 1), (1, 1, 0, 0, 0), (1, 1, 0, 0, 1), (1, 1, 0, 1, 0), (1, 1, 0, 1, 1), (1, 1, 1, 0, 0), (1, 1, 1, 0, 1), (1, 1, 1, 1, 0), (1, 1, 1, 1, 1)]
w, b, testlabels = specialWB()
printFancyT(testSet, testlabels)
print("Weights and Biases from Scratch: ")
perceptronTest(testSet, testlabels, w, b)
print("Weights and Biases from Scikit-Learn Library Trained: ")
trainWeightsandBiases(testSet, testlabels)





