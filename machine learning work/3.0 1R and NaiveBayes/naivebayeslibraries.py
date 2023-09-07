from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from mlxtend.classifier import OneRClassifier
from mlxtend.plotting import plot_confusion_matrix
import numpy as np

trainDataset = []
testDataset = []
trainClassLabels = []
testClassLabels = []
ALLCLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
# testDict = {}
# testIndex = 0

N = 5 #only works for 4 or 5 bins; number of bins for each var

def getData():
    global trainDataset, testDataset, trainClassLabels, testClassLabels
    file2 = ".\iris-py-test.csv"
    file1 = ".\iris-py-train.csv"
        
    with open(file1 ,'r') as f:
        count = 0
        for line in f:
            if(count == 0):
                count+=1
                continue
            temp = line.strip().split(",")
            attributes = list(map(float, temp[:-1]))
            noob = set(attributes)
            # print(len(noob))
            theClass = temp[-1]
            trainDataset.append(np.array(attributes))
            trainClassLabels.append(ALLCLASSES.index(theClass))
    
    with open(file2 ,'r') as f:
        count = 0
        for line in f:
            if(count == 0):
                count+=1
                continue
            temp = line.strip().split(",")
            attributes = list(map(float, temp[:-1]))
            noob = set(attributes)
            # print(len(noob))
            theClass = temp[-1]
            testDataset.append(np.array(attributes))
            testClassLabels.append(ALLCLASSES.index(theClass))
    
    # print(testDict)
    trainDataset = np.array(trainDataset)
    testDataset = np.array(testDataset)
    trainClassLabels = np.array(trainClassLabels)
    testClassLabels = np.array(testClassLabels)




def printData():
    print("Train Dataset instances w/o class label:")
    print(trainDataset)
    print("Train Class Labels")
    print(trainClassLabels)
    print()
    print("Test Dataset instances w/o class label:")
    print(testDataset)
    print("Test Class Labels")
    print(testClassLabels)
    print()

getData()
# printData()
naivebayes = GaussianNB()
naivebayes.fit(trainDataset, trainClassLabels)
test_pred = naivebayes.predict(testDataset)
test_acc = np.mean(test_pred == testClassLabels)
confusionMatrixDict = dict()
for i in range(len(ALLCLASSES)):
    confusionMatrixDict[i] = []
# print(confusionMatrixDict)
# print(testClassLabels)
for i in range(len(test_pred)):
    confusionMatrixDict[testClassLabels[i]].append(test_pred[i])

#print confusion matrix
for i in range(len(ALLCLASSES)):
    print(ALLCLASSES[i] + "(" + str(i) + ")",  end = "  ")
print()
print("     ", "0  1  2")
for i in confusionMatrixDict:
    temp = confusionMatrixDict[i]
    print(i, "   ", temp.count(0), temp.count(1), temp.count(2))
print(f'Testing accuracy {test_acc*100:.2f}%')


