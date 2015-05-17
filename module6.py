import csv
import numpy
import scipy
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import neighbors
import knnplots
from sklearn.naive_bayes import GaussianNB

from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV


#Code common to all modeles from module 3 onwards
##NB. The X and yTransformed variables come from the preprocessing in the previous module.
fileName = "wdbc.csv"
fileOpen = open(fileName, "rU")
csvData = csv.reader(fileOpen)
dataList = list(csvData)
dataArray =  numpy.array(dataList)
X = dataArray[:,2:32].astype(float)
y = dataArray[:, 1]
le = preprocessing.LabelEncoder()
le.fit(y)
yTransformed = le.transform(y)
XTrain, XTest, yTrain, yTest = train_test_split(X, yTransformed)

knnK3 = neighbors.KNeighborsClassifier(n_neighbors = 3)
knnK15 = neighbors.KNeighborsClassifier(n_neighbors = 15)
nbmodel = GaussianNB()

knn3scores = cross_validation.cross_val_score(knnK3,XTrain, yTrain, cv = 5)
print knn3scores
print "Mean of scores KNN3", knn3scores.mean()
print "SD of scores knn3", knn3scores.std()

knn15scores = cross_validation.cross_val_score(knnK15, XTrain, yTrain, cv = 5)
print knn15scores
print "Mean of scores knn15", knn15scores.mean()
print "SD of Scores knn15", knn15scores.std()

nbscores = cross_validation.cross_val_score(nbmodel,XTrain,yTrain, cv = 5)
print nbscores
print "mean of scores nb", nbscores.mean()
print "sd of scores nb", nbscores.std()

meansKNNK3 = []
sdsknnk3 = []
meansknnk15 = []
sdsknnk15 = []
meansnb = []
sdsnb = []

ks = range(2,21)

for k in ks:
    knn15scores = cross_validation.cross_val_score(knnK3, XTrain,
                                                   yTrain, cv=k)
    knn15scores = cross_validation.cross_val_score(knnK15,XTrain,
                                                   yTrain,cv=k)
    nbscores = cross_validation.cross_val_score(nbmodel, XTrain,
                                                yTrain, cv=k)
    meansKNNK3.append(knn3scores.mean())
    sdsknnk3.append(knn3scores.std())
    meansknnk15.append(knn15scores.mean())
    sdsknnk15.append(knn15scores.std())
    meansnb.append(nbscores.mean())
    sdsnb.append(nbscores.std())

# plt.plot(ks, meansKNNK3, label="KNN 3 mean accuracy", color = "purple")
# plt.plot(ks, meansknnk15, label="KNN 15 mean accuracy",color="yellow")
# plt.plot(ks, meansnb,label="NB mean accuracy", color = "blue")
# plt.legend(loc=3)
# plt.ylim(0.9,1)
# plt.title("Accuracy means with Increasing K")
# plt.show()
#
# plt.plot(ks, sdsknnk3, label="KNN 3 sd accuracy",color="purple")
# plt.plot(ks,sdsknnk15, label="KNN 15 sd accuracy", color="yellow")
# plt.plot(ks, sdsnb, label="NB sd accuracy", color="blue")
# plt.legend(loc=3)
# plt.ylim(0,0.1)
# plt.title("Accuracy standard deviations with increasing k")
# plt.show()

parameters = [{'n_neighbors':[1,3,5,10,50,100],
              'weights':['uniform','distance']}]
clf = GridSearchCV(neighbors.KNeighborsClassifier(), parameters, cv=10, scoring="f1")
clf.fit(XTrain,yTrain)

print "best parameter set found: "
print clf.best_estimator_

# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric = 'minkowski',
#                      metric_params=None,n_neighbors=3,p=2,weights='uniform')

print "grid scores:"

for params, mean_score, scores in clf.grid_scores_:
    print "%0.5f (+/-%0.03f) for %r" % (mean_score,scores.std()/2, params)




