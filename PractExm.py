__author__ = 'Alejandro'

filename = "wdbc.csv"

fileopen = open(filename,"rU")
csvdata = csv.reader(fileopen)
datalist = list(csvdata)

dataArray =  numpy.array(dataList)

print(dataArray)