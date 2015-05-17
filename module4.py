import csv
import numpy
import scipy
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
#Imports for Module 4
from sklearn import neighbors
import knnplots


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

nbrs = neighbors.NearestNeighbors(n_neighbors=5, algorithm="ball_tree").fit(X)
distances, indices = nbrs.kneighbors(X)

# print indices[:5]
# print '----'
# print distances[:5]


KnnK3 = neighbors.KNeighborsClassifier(n_neighbors=3)
KnnK3 = KnnK3.fit(X, yTransformed)
predictedk3 = KnnK3.predict(X)

KnnK15 = neighbors.KNeighborsClassifier(n_neighbors=15)
KnnK15 = KnnK15.fit(X, yTransformed)
predictedk15 = KnnK15.predict(X)
#
# print numpy.sum(predictedk3 == predictedk15)

nonAgreement = predictedk3[predictedk3 != predictedk15]
# print 'Number of discrepancies', len(nonAgreement)

nonAgreementPredictedK3 = predictedk3[predictedk3 != yTransformed]
# print 'Number of discrepancies', len(nonAgreementPredictedK3)

nonAgreementPredictedK15 = predictedk15[predictedk15 != yTransformed]
# print 'Number of discrepancies', len(nonAgreementPredictedK15)
#
# print 'accuracy 3 nearest neighbours: ', metrics.accuracy_score(yTransformed, predictedk3)
# print 'accuracy 15 nearest neighbours: ', metrics.accuracy_score(yTransformed, predictedk15)

knnWD = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
knnWD = knnWD.fit(X, yTransformed)
predictedWD = knnWD.predict(X)
#
# print numpy.sum(predictedWD != yTransformed)

XTrain, XTest, YTrain, YTest = train_test_split(X,yTransformed)

print XTrain.shape
print YTrain.shape

knnWD = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
knnWD = knnWD.fit(XTrain, YTrain)
predictedWD = knnWD.predict(XTest)

print metrics.classification_report(YTest, predictedWD)
print 'accuracy: ', metrics.accuracy_score(YTest, predictedWD)

knnplots.plotaccuracy(XTrain,YTrain,XTest,YTest,310)

knnplots.decisionplot(XTrain,YTrain, n_neighbors=3,weights='uniform')

knnplots.decisionplot(XTrain,YTrain, n_neighbors=15,weights='uniform')

