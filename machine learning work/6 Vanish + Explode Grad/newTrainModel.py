import sys
import numpy as np
import math
import pickle
from sklearn.neural_network import MLPClassifier
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
NUMCLASSES = 2

# errorList = []
# weightTracker = []
gradientList = []
accuracyList = []
count = 0

def f(n): #function which the calculated value wp + b runs through
    asdf = 1 / (1+math.e**(-1 * n))
    return asdf

def fprimo(n): #derivative of the previous function used to compute the error.
    asdf = (math.e**(-1*n))/((1+(math.e**(-1 * n)))**2)
    return asdf

def getTrainData(file): #gets the mnist data from the mnist_train data (for training) (plus preprocessing to make it in a good format for training)
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
            mDataY.append(trueval)

def getTestData(file): #gets the mnist data from the mnist_train.csv file (plus preprocessing to make it in a good format for testing)
    global mDataXTest, mDataYTest
    with open(file, "r") as r:
        count = 0
        for line in r:
            temp = line.strip()
            data = temp.split(",")
            trueval = int(data[0])
            datalist = data[1:]
            datalist = np.array(np.float_(datalist))/255
            mDataXTest.append(datalist)
            # distance = getDistance(val1, val2)
            mDataYTest.append(trueval)
            count+=1


clf = MLPClassifier(random_state=1, max_iter=300)
clf.fit(mDataX, mDataY)