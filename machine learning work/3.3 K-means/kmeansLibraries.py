from sklearn.cluster import KMeans
import numpy as np
import math


values = []
labels = []
K=6


def getData(file):
    toReturn = []
    count = 0
    with open(file, "r") as r:
        for line in r:
            if(count > 0):
                temp = line.strip()
                thing = tuple(temp.split(","))
                new = []
                for i in range(3):
                    val = math.log(float(thing[i]))
                    new.append(val)
                new.append(float(thing[3]))
                new.append(int(thing[4]))
                values.append(tuple(new))
            else:
                count += 1

getData("star_data.csv")
npvalues = np.array(values)
kmeans = KMeans(n_clusters=K, random_state=0).fit(npvalues)
thing = kmeans.predict(npvalues)
print(thing)