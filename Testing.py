import matplotlib.pyplot as plt
from sklearn.datasets from load_iris

iris = load_iris()

x = iris.data
labels = iris.target_names

gMarkers = ["+","_","x"]
gColours = ["Blue","Magenta","cyan"]
gIndices = [0,1,2]

f1 = 0
f2 = 1
for mark, col, i, iris.target_names in zip(gMarkers, gColours, gIndices, labels):
    plt.scatter(x = x[iris.target == i,f1],
    y = x[iris.target == i,f2],
    marker = mark, c=col,label=iris.target_names)
    plt.xlabel(iris.feature_names[f1])
    plt.ylabel(iris.feature_names[f2])
    plt.legend(loc='upper right')
    plt.show()
