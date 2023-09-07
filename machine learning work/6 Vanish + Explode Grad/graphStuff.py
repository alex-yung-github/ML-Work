import sys
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt 

#graphing the loss versus the tracked weight.
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