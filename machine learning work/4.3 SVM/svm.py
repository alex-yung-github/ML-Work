from sklearn import svm
from sklearn.datasets import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
import numpy as np

blobsX, blobsY = make_blobs(n_samples=100, centers=2, n_features=2)


# scatter plot, dots colored by class value
df = DataFrame(dict(x=blobsX[:,0], y=blobsX[:,1], label=blobsY))
colors = {0:'red', 1:'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()

clf = svm.SVC()
clf.fit(blobsX, blobsY)
correct = 0
for i in range(len(blobsX)):
    tempList = np.array([blobsX[i]])
    tempVal = clf.predict(tempList)
    if(tempVal == blobsY[i]):
        correct += 1
    print("Point: ", blobsX[i], " | SVM Classified as: ", tempVal, " | ", "Actual Class: ", blobsY[i])
print("Accuracy: ", str(correct) + "%")
