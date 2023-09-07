import sys
import numpy as np
import math
import pickle

# # write python dict to a file
# mydict = {'a': 1, 'b': 2, 'c': 3}
# output = open('myfile.pkl', 'wb')
# pickle.dump(mydict, output)
# output.close()

# # read python dict back from the file
# pkl_file = open('myfile.pkl', 'rb')
# mydict2 = pickle.load(pkl_file)
# pkl_file.close()

LAMBDA = 1.2
mDataX = []
mDataY = []
mDataXTest = []
mDataYTest = []

errorList = []
weightTracker = []
accuracyList = []
count = 0

def f(n): #function which the calculated value wp + b runs through
    asdf = 1 / (1+math.e**(-1 * n))
    return asdf

def fprimo(n): #derivative of the previous function used to compute the error.
    asdf = (math.e**(-1*n))/((1+(math.e**(-1 * n)))**2)
    return asdf

def findError(x, y, wList, bList): #Runs the forward pass, calculation of mean squared error, and backward pass
    global count
    vectorizedF = np.vectorize(f) #vectorizes the function so it can be applied to numpy matrices
    ai = np.array(x) #converts the input to a numpy array
    alist = [] # for history purposes 
    dotList = [] # for history purposes
    alist.append(ai)
    dotList.append(ai)
    #FORWARD PASS
    for i in range(1, len(wList)): #runs over each layer 
        # ai = ai.transpose()
        # print(ai)
        wi = wList[i] #gets the weights and biases from the current layer
        bi = bList[i]
        dot = ai@wi + bi #calculates the value n (not through the activation func yet)
        dotList.append(dot) #saves the value n for future use in the backward pass
        ai = vectorizedF(dot) #funs the value n through the activation function f (call this a)
        alist.append(ai) #saves the value a for backward pass
        # print(ai)
        # print("temp: ", temp)

    fprime = np.vectorize(fprimo) #vectorizes the derivative of the activation function 
    vecX = np.array(x) #vectorizes the inputs and labels as numpy arrays so operations can be performed
    vecY = np.array(y)
    # print(dotList[-1])
    # [CALCULATING MEAN SQUARED ERROR]
    deltaN = fprime(dotList[-1]) * (vecY - alist[-1]) #
    v1 = np.linalg.norm(vecY-ai)
    asdf = (1/2) * (v1**2)
    # for i in range(len(y[0])):
    #     asdf += ((1/2) * (y[0][i]-ai[0][i])**2)

    #UPDATING WEIGHTS IN A BACKWARD PASS
    deltas = [0] * len(wList)
    deltas[-1] = deltaN
    newWList = []
    newBList = []
    #calculating the partial derivative for weights and biases in each hidden layer (delta is the partial derivative of E-total with respect to all the weights in the layer as a matrix)
    for w in range(len(wList)-2, 0, -1):
        deltaL = fprime(dotList[w]) * (deltas[w+1]@(wList[w+1].T))
        # deltaTemp = fprime(ai) * (vecY - ai)
        deltas[w] = deltaL

    newBList.append(0)
    newWList.append(0)

    for l in range(1, len(wList)): #using the partial derivatives of the total error, this part calculates new weights and biases for each layer
        bNew = bList[l] + (LAMBDA*deltas[l])
        wNew = wList[l] + (LAMBDA*((alist[l-1].T)@deltas[l]))
        newWList.append(wNew)
        newBList.append(bNew)
    count += 1
    global errorList, weightTracker #this portion of the code saves a certain list of weights and the loss so that later it can be graphed. (I saved multiple weights instead of just one just to see multiple weights change for personal testing)
    if(count >= 1000):
        errorList.append(asdf)
        weightTracker.append(wNew[0])
        count = 0
    
    return (asdf, newWList, newBList) #returns updated weights and biases


def realbackprop(epochs, x, y, wList, bList): #performs the backpropagation 
    global mDataXTest, mDataYTest, accuracyList
    newW = wList
    newB = bList
    getMNISTTestData("mnist_test.csv") #saves the testing data to the global lists
    print()
    accuracyCounter= 500
    for i in range(epochs): #maximum number of times it runs over all the points in the dataset 
        print("Epoch ", i+1)
        for l in range(len(x)):  #runs through each point
            sheesh = findError([x[l]], [y[l]], newW, newB) #this function doesn't just find the error. it runs the forward pass, computes the squared mean error, and updates the weight through the backward pass
            newW = sheesh[1] #newW and newB are updated weights and biases
            newB = sheesh[2]
            accuracyCounter += 1
            if(accuracyCounter >= 500): #every 500 points trained on and on the first point, I print out the accuracy on the testing set and save it for future graphing.
                tempAcc = run(mDataXTest, mDataYTest, newW, newB)
                accuracyList.append(tempAcc)
                accuracyCounter = 0
            # print(newW, newB)
        print("Saving epoch...") #code that runs every epoch and dumps the weights and biases into a file that contains the most current w and b
        accOutput = open('accuracySave.pkl', 'wb')
        pickle.dump(accuracyList, accOutput)
        output = open('testSave.pkl', 'wb')
        pickle.dump((newW, newB), output)
        output.close()
        graphOutput = open('errorGraph.pkl', 'wb')
        global errorList
        pickle.dump((errorList), graphOutput)
        graphOutput.close()
        weightOutput = open('weightGraph.pkl', 'wb')
        global weightTracker
        pickle.dump((weightTracker), weightOutput)
        weightOutput.close()
    return (newW, newB) #when number of epochs completes, it returns the weights and biases

def getMNISTTrainData(file): #gets the mnist data from the mnist_train data (for training) (plus preprocessing to make it in a good format for training)
    global mDataX, mDataY
    with open(file, "r") as r:
        count = 0
        for line in r:
            temp = line.strip()
            data = temp.split(",")
            trueval = int(data[0])
            datalist = data[1:]
            datalist = np.array(np.float_(datalist))/255
            mDataX.append(datalist.tolist())
            # distance = getDistance(val1, val2)
            truelist = [0] * 10
            truelist[trueval] = 1
            mDataY.append(truelist)
            count+=1

def getMNISTTestData(file): #gets the mnist data from the mnist_train.csv file (plus preprocessing to make it in a good format for testing)
    global mDataXTest, mDataYTest
    with open(file, "r") as r:
        count = 0
        for line in r:
            temp = line.strip()
            data = temp.split(",")
            trueval = [int(data[0])]
            datalist = data[1:]
            datalist = np.array(np.float_(datalist))/255
            mDataXTest.append(datalist)
            # distance = getDistance(val1, val2)
            mDataYTest.append(trueval)
            count+=1

def mnistWandB(): #initialize the initial weights and biases for mnist
    w1 = 2 * np.random.rand(784, 300) - 1
    w2 = 2 * np.random.rand(300, 100) - 1
    w3 = 2 * np.random.rand(100, 10) - 1

    b1 = 2 * np.random.rand(1, 300) - 1
    b2 = 2 * np.random.rand(1, 100) - 1
    b3 = 2 * np.random.rand(1, 10) - 1

    return ([0, w1, w2, w3], [0, b1, b2, b3])

def getWandBPKL(file): #pulls the weights and biases from previous runs
    pkl_file = open(file, 'rb')
    temp = pickle.load(pkl_file)
    pkl_file.close()
    weight = temp[0]
    bias = temp[1]
    return (weight, bias)

def run(x, y, wList, bList): #uses the weights and biases on the testing set and records the accuracy
    vectorizedF = np.vectorize(f)
    correct = 0
    for h in range(len(x)):
        ai = np.array(x[h])
        for i in range(1, len(wList)):
            wi = wList[i]
            bi = bList[i]
            temp = ai@wi + bi
            ai = vectorizedF(temp)
        asdf = ai[0]
        # print(np.argmax(asdf), y[h])
        if(np.argmax(asdf) == y[h]):
            correct += 1
    finalValForTest = correct/len(x)
    print("Accuracy on Test Data: ", finalValForTest)
    return finalValForTest

stff = input("Train New (N), Continue Train (C), or Test (T) ? ")
if(stff == "N"): #trains new weights and biases from scratch
    getMNISTTrainData("mnist_train.csv") #pulls the training data
    mstuff = mnistWandB() #initializes the weights and biases
    # mstuff = getWandBPKL("saveWandB.pkl")
    wCircle = mstuff[0] #spllits the weights and biases into their own lists
    bCircle = mstuff[1]
    finalvals = realbackprop(1, mDataX, mDataY, wCircle, bCircle) #runs backpropagation
    print(finalvals)
elif(stff == "T"): #tests on testing data using the training and testing data from saved file
    stuff = getWandBPKL("testSave.pkl")  #pulls the weights and biases
    # print(stuff)
    inp = getMNISTTestData("mnist_test.csv") #pulls the testing data from csv file
    print(run(mDataXTest, mDataYTest, stuff[0], stuff[1])) #runs and prints accuracy of the weights and biases
elif(stff == "C"): #continues training the w and b from the saved file. Very similar to when training a new set, except we are pulling the saved weights and biases instead of initializing new ones.
    ff = input("File Name?")
    getMNISTTrainData("mnist_train.csv")
    wandB = getWandBPKL(ff)
    wCircle = wandB[0]
    bCircle = wandB[1]
    finalvals = realbackprop(1, mDataX, mDataY, wCircle, bCircle)
    print(finalvals)