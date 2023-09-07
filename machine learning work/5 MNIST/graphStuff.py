import sys
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt 

#graphing accuracy vs number of points tested
accFile =  "accuracySave.pkl"
accFilePickle = open(accFile, 'rb')
accTemp = pickle.load(accFilePickle)
accFilePickle.close()
accList = accTemp
xVals = []
for i in range(len(accList)):
    xVals.append(i)
plt.scatter(xVals, accList)
plt.title("Accuracy vs Points Tested")
plt.xlabel("Points Tested (1 = 500)")
plt.ylabel("Accuracy (%)")
plt.show()

#graphing loss vs number of points tested
file =  "errorGraph.pkl"
pkl_file = open(file, 'rb')
temp = pickle.load(pkl_file)
pkl_file.close()
errorList = temp
print(errorList)
xVals = []
for i in range(len(errorList)):
    xVals.append(i)
plt.scatter(xVals, errorList)
plt.title("Loss vs Points Tested")
plt.xlabel("Points Tested (1 = 1000 points)")
plt.ylabel("Loss ")
plt.show()

#graphing the loss versus the tracked weight.
file =  "weightGraph.pkl"
pkl_file = open(file, 'rb')
temp = pickle.load(pkl_file)
pkl_file.close()
weights = np.array(temp)
print(weights.shape)
trackedWeight = []
for i in weights:
    trackedWeight.append(i[0])
print(trackedWeight)
plt.scatter(trackedWeight, errorList)
plt.title("Loss vs Tracked Weight")
plt.xlabel("Tracked Weight Value")
plt.ylabel("Loss")
plt.show()
