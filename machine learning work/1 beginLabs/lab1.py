import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
data = pd.read_csv("Iris.csv")

print(data.columns.values)
print(data.index.values)
print(data.iloc[0]) #row
b1 = data["sepallength"]
print(b1)

b2 = data["class"]
# plt.bar(b2, b1, color = "maroon", width = .4)
plt.plot(b2, b1, '--c')
plt.ylabel("Sepal Lengths")
plt.xlabel("Class")
plt.title("Sepal Length by Class")
plt.show()
# print (data.head(10))
