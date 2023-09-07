import sys
from turtle import circle
import numpy as np
import math

LAMBDA = 1.5
circleDataX = []
circleDataY = []

def inp1():
    epochs = 3
    trainfile = "data.txt"
    return (epochs, trainfile)

def getTrainSet(file):
    toReturn = []
    with open(file, "r") as r:
        for line in r:
            temp = line.strip()
            toReturn.append(temp)
    return toReturn

def videonetwork():
    w1 = np.array([[1, -.5], [1,.5]])
    w2 = np.array([[1, 2], [-1, -2]])
    b1 = np.array([[1, -1]])
    b2 = np.array([[-.5, .5]])
    return ([0, w1, w2], [0, b1, b2])

def f(n):
    asdf = 1 / (1+math.e**(-1 * n))
    return asdf

def fprimo(n):
    asdf = (math.e**(-1*n))/((1+(math.e**(-1 * n)))**2)
    return asdf

def findError(x, y, wList, bList):
    vectorizedF = np.vectorize(f)
    ai = np.array(x)
    alist = []
    dotList = []
    alist.append(ai)
    dotList.append(ai)
    for i in range(1, len(wList)):
        # ai = ai.transpose()
        # print(ai)
        wi = wList[i]
        bi = bList[i]
        dot = ai@wi + bi
        dotList.append(dot)
        ai = vectorizedF(dot)
        alist.append(ai)
        # print(ai)
        # print("temp: ", temp)

    fprime = np.vectorize(fprimo)
    vecX = np.array(x)
    vecY = np.array(y)
    deltaN = fprime(dotList[-1]) * (vecY - alist[-1])

    v1 = np.linalg.norm(vecY-ai)
    asdf = (1/2) * (v1**2)
    # for i in range(len(y[0])):
    #     asdf += ((1/2) * (y[0][i]-ai[0][i])**2)

    deltas = [0] * len(wList)
    deltas[-1] = deltaN
    newWList = []
    newBList = []
    for w in range(len(wList)-2, 0, -1):
        deltaL = fprime(dotList[w]) * (deltas[w+1]@(wList[w+1].T))
        # deltaTemp = fprime(ai) * (vecY - ai)
        deltas[w] = deltaL

    newBList.append(0)
    newWList.append(0)
    for l in range(1, len(wList)):
        bNew = bList[l] + (LAMBDA*deltas[l])
        wNew = wList[l] + (LAMBDA*((alist[l-1].T)@deltas[l]))
        newWList.append(wNew)
        newBList.append(bNew)
        
    return (asdf, newWList, newBList)

def realbackprop(epochs, x, y, wList, bList):
    newW = wList
    newB = bList
    print()
    for i in range(epochs):
        print("Epoch ", i+1)
        for l in range(len(x)):
            sheesh = findError([x[l]], [y[l]], newW, newB)
            # print(sheesh[0])
            newW = sheesh[1]
            newB = sheesh[2]
        # print()
    return (newW, newB)

def run(x, wList, bList):
    vectorizedF = np.vectorize(f)
    ai = np.array(x)
    for i in range(1, len(wList)):
        wi = wList[i]
        bi = bList[i]
        temp = ai@wi + bi
        ai = vectorizedF(temp)
    asdf = ai[0]
    return (x,  asdf)

def beeground(num):
    toReturn = 0
    if(num > .5):
        toReturn =1 
    else:
        toReturn = 0
    return toReturn
    

def sumStuff():
    w1 = 2 * np.random.rand(2, 2) - 1
    w2 = 2 * np.random.rand(2, 2) - 1
    b1 = 2 * np.random.rand(1, 2) - 1
    b2 = 2 * np.random.rand(1, 2) - 1
    return ([0, w1, w2], [0, b1, b2])

def getCircleData(file):
    global circleDataX, circleDataY
    with open(file, "r") as r:
        for line in r:
            temp = line.strip()
            data = temp.split(" ")
            val1 = float(data[0])
            val2 = float(data[1])
            circleDataX.append([val1, val2])
            # distance = getDistance(val1, val2)
            if(val1**2 + val2**2 < 1):
                circleDataY.append([1])
            else:
                circleDataY.append([0])
            
def circleWandB():
    b1 = np.array([[1.35, 1.35, 1.35, 1.35]])
    b2 = np.array([[-3]])

    w1 = np.array([[-1, 1, -1, 1], [1,-1,-1,1]])
    w2 = np.array([[1], [1], [1], [1]])
    return ([0, w1, w2], [0, b1, b2])

def checkFunctionCircle(n, wL, bL):
    count = 0
    for i in range(n):
        point = 2 * np.random.rand(1, 2) - 1
        val = run(point, wL, bL)
        # print(val)
        t = beeground(val[1][0])
        # print("Real Value: ", end = "")
        if(np.linalg.norm(point)< 1):
            print("1", end = " ")
            if(t == 1):
                print("cool")
                count+=1
            else:
                print("notcool")
        else:
            print("0", end = " ")
            if(t == 0):
                print("cool")
                count+=1
            else:
                print("notcool")
        print()
    print("Percentage Correct", count/n)

def checkFunctionCircle2(wL, bL):
    count = 0
    for i in range(len(circleDataX)):
        point = circleDataX[i]
        val = run(point, wL, bL)
        t = beeground(val[1][0])
        if(np.linalg.norm(point)< 1):
            if(t == 1):
                count+=1
        else:
            if(t == 0):
                count+=1
    print("Percentage Correct", count/10000)
    print("Number Correct", count)

def inputio():
    val = sys.argv[1]
    return val
# def getDistance(pt1, pt2):
#     return math.sqrt(pt1**2 + pt2**2)

# For part 1
# vals1 = [[2, 3]]
# vals2 = [[.8, 1]]
# stuff = videonetwork()
# wL = stuff[0]
# bL = stuff[1]
# realbackprop(2, vals1, vals2, wL, bL)

# #For part 2
# vals1 = [[0,0], [0,1], [1, 0], [1, 1]]
# vals2 = [[0,0], [0,1], [0,1], [1, 0]]
# summystuff = sumStuff()
# wSum = summystuff[0]
# bSum = summystuff[1]
# finalstuff = realbackprop(500, vals1, vals2, wSum, bSum)
# useW = finalstuff[0]
# useB = finalstuff[1]
# print(run([[1, 1]], useW, useB))

#For part 3
# getCircleData("10000_pairs.txt")
# circlestuff = circleWandB()
# wCircle = circlestuff[0]
# bCircle = circlestuff[1]
# finalvals = realbackprop(40, circleDataX, circleDataY, wCircle, bCircle)
# print(finalvals)
# checkFunctionCircle(500, finalvals[0], finalvals[1])

inp = inputio()
if(inp == "S"):
    vals1 = [[0,0], [0,1], [1, 0], [1, 1]]
    vals2 = [[0,0], [0,1], [0,1], [1, 0]]
    summystuff = sumStuff()
    wSum = summystuff[0]
    bSum = summystuff[1]
    finalstuff = realbackprop(500, vals1, vals2, wSum, bSum)
    useW = finalstuff[0]
    useB = finalstuff[1]
    print(run([[0, 0]], useW, useB))
    print(run([[0, 1]], useW, useB))
    print(run([[1, 0]], useW, useB))
    print(run([[1, 1]], useW, useB))
elif(inp == "C"):
    getCircleData("10000_pairs.txt")
    circlestuff = circleWandB()
    wCircle = circlestuff[0]
    bCircle = circlestuff[1]
    finalvals = realbackprop(40, circleDataX, circleDataY, wCircle, bCircle)
    print(finalvals)
    checkFunctionCircle2(finalvals[0], finalvals[1])
