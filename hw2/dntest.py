import numpy as np
import sys
import csv
from math import e

def sigmoid(z):
	return 1/(1+e**(-z))

def relu(z,deriv=False):
	if deriv == True:
		return 1. * (z > 0)
	return z * (z > 0)

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
	#return ( (x - x.min(axis=0)) / (x.ptp(axis=0) + 1e-20) )
	return ( (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-20) )

modelname = sys.argv[1] + '.npz'
data = np.load(modelname)
w,w0,w1,w2 = data['w'],data['w0'],data['w1'],data['w2']
b,b0,b1,b2 = data['b'],data['b0'],data['b1'],data['b2']


tf = open(sys.argv[2], 'rb')
out = open(sys.argv[3], 'w')
print >>out, 'id,label'

x = []
for line in csv.reader(tf):
	sample = map(float,line[1:])
	x.append(sample)

x = np.array(x,dtype=np.float64)
x = normalize(x)
l0 = sigmoid(np.dot(x,w0) + b0)
l1 = sigmoid(np.dot(l0,w1) + b1)
l2 = sigmoid(np.dot(l1,w2) + b2)
y = sigmoid(np.dot(l2,w) + b)
row,column = y.shape
for i in range(row):
	if y[i,0] > 0.5:
		print >>out, '%s,1'%(i+1)
	else:
		print >>out, '%s,0'%(i+1)