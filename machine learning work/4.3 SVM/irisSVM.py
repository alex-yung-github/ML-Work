import numpy as np
import math
import json
import random
import statistics
from sklearn import svm
from sklearn.datasets import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
import numpy as np
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

train_dataset = []
train_classLabels = []

test_dataset = []
test_classLabels = []

ALLCLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
testDict = {}
testIndex = 0
N = 12 #number of neighbors

def getData():
    global train_dataset, train_classLabels, test_dataset, test_classLabels
    filetest = ".\iris-py-test.csv"
    filetrain = ".\iris-py-train.csv"
        
    with open(filetrain ,'r') as f:
        count = 0
        for line in f:
            if(count == 0):
                count+=1
                continue
            temp = line.strip().split(",")
            attributes = list(map(float, temp[:-3]))
            noob = set(attributes)
            # print(len(noob))
            theClass = ALLCLASSES.index(temp[-1])
            train_dataset.append(attributes)
            train_classLabels.append(theClass)
    
    train_dataset = np.array(train_dataset)
    train_classLabels = np.array(train_classLabels)

    with open(filetest ,'r') as f:
        count = 0
        for line in f:
            if(count == 0):
                count+=1
                continue
            temp = line.strip().split(",")
            attributes = list(map(float, temp[0:2]))
            noob = set(attributes)
            # print(len(noob))
            theClass = ALLCLASSES.index(temp[-1])
            test_dataset.append(attributes)
            test_classLabels.append(theClass)
    
    test_dataset = np.array(test_dataset)
    test_classLabels = np.array(test_classLabels)

getData()
df = DataFrame(dict(x=train_dataset[:,0], y=train_dataset[:,1], label=train_classLabels))

# colors = {ALLCLASSES[0]:'red', ALLCLASSES[1]:'blue', ALLCLASSES[2]:"green"}
# fig, ax = pyplot.subplots()
# grouped = df.groupby('label')
# for key, group in grouped:
#     group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
# pyplot.show()

clf = svm.SVC()
clf.fit(train_dataset, train_classLabels)
correct = 0
for i in range(len(train_dataset)):
    tempList = np.array([train_dataset[i]])
    tempVal = clf.predict(tempList)
    if(tempVal == train_classLabels[i]):
        correct += 1
    # print("Point: ", blobsX[i], " | SVM Classified as: ", tempVal, " | ", "Actual Class: ", blobsY[i])
print("Train Accuracy: ", str(correct) + "%")

correct = 0
for i in range(len(test_dataset)):
    tempList = np.array([test_dataset[i]])
    tempVal = clf.predict(tempList)
    if(tempVal == test_classLabels[i]):
        correct += 1
    # print("Point: ", blobsX[i], " | SVM Classified as: ", tempVal, " | ", "Actual Class: ", blobsY[i])
print("Test Accuracy: ", str(correct) + "%")

plot_decision_regions(train_dataset, train_classLabels, clf=clf)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('SVM on Iris')
plt.show()