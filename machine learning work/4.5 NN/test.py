from statistics import stdev
import sys
from  math  import  log
import string
import random
import itertools
import ast

LAMBDA = 1

def inputP():
    point = sys.argv[1]
    point = ast.literal_eval(point) 
    return point

def truthT(n, labels):
    zeTruth = list(itertools.product([0, 1], repeat=n))
    zeTruth.sort()

    toReturn = []
    for i in range(len(zeTruth)):
        val = labels[i]
        bigman = zeTruth[i]
        toReturn.append((bigman, int(val)))
    return toReturn

def printFancyT(table):
    bruh = table[0]
    bruh1 = bruh[0]
    print(" ", end = "")
    for i in range(len(bruh1)):
        print("In", i, end = " ")
    print(" | Output ")

    for i in range(len(table)):
        temp = table[i]
        bigmen = temp[0]
        val = temp[1]
        print("  ", end = "")
        for i in bigmen:
            print(i, "   ", end = "")
        print("| ", val, end = "")
        print()

def perceptron(A, w, b, x):
    wx = 0
    for i, j in zip(w, x):
            wx += i * j
    wxb = wx+b
    return A(wxb)

def step(num):
    if(num > 0):
        return 1
    else:
        return 0

def check(n, w, b):
    truth = truthT(len(w), n)
    # printFancyT(truth)
    inp = list(itertools.product([0, 1], repeat=len(w)))
    inp.sort(reverse = True)
    end = []
    for i in inp:
        val = perceptron(step, w, b, i)
        end.append(val)
    correct = 0
    for i in range(len(truth)):
        vals = truth[i]
        trueval = vals[1]
        if(trueval == end[i]):
            correct+=1
    return correct / len(truth)

def getTable(bits):
    # tempN = bin(n)
    # binaryNum = tempN[2:]
    # while(len(binaryNum) < 2**bits):
    #     binaryNum = "0" + binaryNum
    # print(binaryNum)
    toReturn = []
    zeTruth = list(itertools.product([0, 1], repeat=bits))
    zeTruth.sort(reverse = True)
    return zeTruth

def training(initw, initb, numEpochs, data, n):
    w = initw
    b = initb
    actualVals = truthT(len(w), n)
    currEpoch = ()
    count = 0
    for epoch in range(numEpochs):
        initw = w
        initb = b
        for row in range(len(data)):
            thing = data[row]
            val = perceptron(step, w, b, thing)
            error = actualVals[row][1] - val
            if(error != 0):
                finalthing = [i * LAMBDA * error for i in thing]
                w = [finalthing[k] + w[k] for k in range(len(finalthing))]
                # finalthing = [thing[0] * LAMBDA * error, thing[1] * LAMBDA * error]
                # w = [w[0] + finalthing[0], w[1] + finalthing[1]]
                b = b + (error * LAMBDA)
        if(initw == w and initb == b):
            count+=1
        else:
            count = 0
        if(count >= 2):
            break
    thingay = (check(n, w, b))
    return ((w, b), thingay)


def algorithm(n, bits, epochs):
    w = [0] * bits
    data = getTable(bits)
    d = training(w, 0, epochs, data, n)
    return d

def generalAlg(bits, epochs):
    w = [0] * bits
    combos = 2**(2**bits)
    data = getTable(bits)
    count = 0
    for i in range(combos):
        d = training(w, 0, epochs, data, i)
        print(i)
        print("Accuracy: ", d[1])
        print("Weight and Bias: ", d[0])
        if(d[1] == 1):
            count += 1
    print(count, "/", combos)

#XOR HAPPENS HERE


# param = input()
# param = (50101, (3, 2, 3, 1), -4)
# param = (8, (1, 1), -1.5)
# final = check(param[0], param[1], param[2])
# print(final)

# inp = inputperp2()
# generalAlg(inp, 100)
# generalAlg(4, 100)
labels = [1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1]
truth = truthT(4, labels)
print(truth)