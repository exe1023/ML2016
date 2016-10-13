import csv
import numpy as np
import random
from bestconstants import hours, costsum, loopnum, usefeature
from numpy.linalg import inv
import math


def closedform(feature):
	tmp = [cnt*10 for cnt in range(len(feature[0])/10 )]
	featuretmp = [[] for i in tmp]
	for i in range(len(feature[0])/10 ):
		for j in usefeature:
			featuretmp[i] += feature[j][i*10:i*10+hours]
	x = np.array(featuretmp, dtype=np.float64)
	y = np.array([ [feature[9][cnt+hours]] for cnt in tmp ], dtype=np.float64)

	w = (inv( (np.transpose(x)).dot(x) + 1000 ).dot(np.transpose(x))).dot(y)
	return w


feature = [[] for j in range(18)]

w = np.full((len(usefeature),hours),1,dtype=np.float64)
b = 1.

# read train data
f = open('train3.csv', 'rb')
count = 0
for line in csv.reader(f):
	feature[count % 18].extend(map(float,line[3:]))
	count += 1

w = closedform(feature)

#test
out = open('kaggle_best.csv', 'w')
print >> out, "id,value"
test = open('test2.csv', 'rb')
count = 0
feature = [[] for j in range(18)]
for line in csv.reader(test):
	feature[count % 18].extend(map(float,line[2:]))
	count += 1

#feature = normalize(feature)

for i in range(240):
	tmp = []
	for j in usefeature:
		tmp += feature[j][i*9 + 9 - hours:i*9 + 9]
	output = np.array(tmp, dtype=np.float64).dot(w)
	print >> out, "id_%d,%f" %(i, output)



f.close()