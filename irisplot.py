
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
labels = iris.target_names
#Symbols to represent the points for the three classes on the graph.
gMarkers = ["+", "_", "x"]
#Colours to represent the points for the three classes on the graph
gColours = ["blue", "magenta", "cyan"]
#The index of the class in target_names
gIndices = [0, 1, 2]
#Column indices for the two features you want to plot against each other:

PlotSize = 4

for j in range(PlotSize):
   for k in range(PlotSize):
      plt.subplot(PlotSize, PlotSize, j+1+k*PlotSize)
      for mark, col, i, iris.target_name in zip(gMarkers, gColours, gIndices, labels):
         plt.scatter(x = X[iris.target == i, j], y = X[iris.target == i, k],
         marker = mark, c = col, label=iris.target_name)
plt.xlabel(iris.feature_names[j])
plt.ylabel(iris.feature_names[k])
plt.tight_layout()
plt.show()

