from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_circles(n_samples=1_000, factor=0.8, noise=0.05, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0, train_size = .5)
# plt.grid()
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
# plt.show()

with open('train.csv', 'w') as f:
    for point in range(len(X_train)):
        temp1 = str(X_train[point][0])
        temp2 = str(X_train[point][1])
        f.write(str(y_train[point]) + "," + temp1 + "," + temp2 + "\n")
    f.close()

with open('test.csv', 'w') as f:
    for point in range(len(X_test)):
        temp1 = str(X_test[point][0])
        temp2 = str(X_test[point][1])
        f.write(str(y_test[point]) + "," + temp1 + "," + temp2  + "\n")
    f.close()


