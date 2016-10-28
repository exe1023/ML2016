import numpy as np
import sys
import csv
from math import e

def sigmoid(z):
	return 1/(1+e**(-z))
def xentropy(y,yhat):
	if y == 1:
		return 0
	return -1*(yhat*np.log(y) + (1-yhat)*np.log(1-y))
def validation(w,b,file):
	with open(file,'rb') as f:
		loss = 0.
		for line in csv.reader(f):
			x = np.array(map(float,line[1:-1]), dtype=np.float64)
			y = sigmoid(np.sum(w * x) + b)
			loss += xentropy(y, float(line[-1]))
		print 'file:%s loss:%f'%(file,loss)


def normalize(x):
	return ( (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-20) )



modelname = sys.argv[1] + '.npz'
data = np.load(modelname)
w = data['w']
b = data['b']

tf = open(sys.argv[2], 'rb')
out = open(sys.argv[3], 'w')
print >>out, 'id,label'

x = []
for line in csv.reader(tf):
	sample = map(float,line[1:])
	x.append(sample)

x = np.array(x,dtype=np.float64)
#x = normalize(x)
y = sigmoid(np.dot(x,w) + b)
row,column = y.shape
for i in range(row):
	if y[i,0] > 0.5:
		print >>out, '%s,1'%(i+1)
	else:
		print >>out, '%s,0'%(i+1)