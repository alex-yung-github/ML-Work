# split imbalanced dataset into train and test sets with stratification
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import csv

# create dataset
with open('mushrooms.csv' ,'r') as f:
    count = 0
    X = []
    y = []
    for line in f:
        if(count == 0):
            count+=1
            continue
        temp = line.strip().split(",")
        tempX = temp[1:]
        tempY = temp[0]
        X.append(tempX)
        y.append(tempY)
    X = np.array(X)
    y = np.array(y)
print(X.shape, y.shape)

# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)
# print(Counter(y_train))
# print(Counter(y_test))
print(X_train[1])
print(y_train[1])

with open('mushroom-py-train.csv' ,'w', newline='') as f:
    csvwriter = csv.writer(f) 
    csvwriter.writerow(["cap-shape","cap-surface","cap-color","bruises","odor","gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","veil-type","veil-color","ring-number","ring-type","spore-print-color","population","habitat","class"])
    for i in range(len(X_train)):
        temp = X_train[i]
        toReturn = np.append(temp, y_train[i])
        csvwriter.writerow(toReturn)

with open('mushroom-py-test.csv' ,'w', newline='') as f:
    csvwriter = csv.writer(f) 
    csvwriter.writerow(["cap-shape","cap-surface","cap-color","bruises","odor","gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","veil-type","veil-color","ring-number","ring-type","spore-print-color","population","habitat","class"])
    for i in range(len(X_test)):
        temp = X_train[i]
        toReturn = np.append(temp, y_test[i])
        csvwriter.writerow(toReturn)


