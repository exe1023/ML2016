import numpy as np
import sys
import csv
from math import e

def sigmoid(z,deriv=False):
	if deriv == True:
		return z * (1-z)
	return 1/(1+np.exp(-z))

def xentropy(y,yhat):
	if y == 1:
		return 0
	return -1*(yhat*np.log(y) + (1-yhat)*np.log(1-y))

def normalize(x):
	return ( (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-20) )


def accuracy(y,yhat):
	row,column=y.shape
	cnt = 0.
	for i in range(row):
		if y[i,0] < 0.5 and yhat[i,0] < 0.5:
			cnt += 1
		elif y[i,0] > 0.5 and yhat[i,0] > 0.5:
			cnt += 1
	return float(cnt/row)


w = np.random.randn(57,1)
b = 1.
g = np.full((57,1),1e-20,dtype=np.float64)
gb = 1e-20


validation = False
fold = -1
if len(sys.argv) >= 4:
	validation = True
	fold = int(sys.argv[3])

#read data
x_train,y_train = [],[]
x_valid,y_valid = [],[]

tf = open(sys.argv[1], 'rb')
for line in csv.reader(tf):
	sample = map(float,line[1:])
	x_train.append(sample[:-1])
	y_train.append([sample[-1]])

validlen = (int)(4001/5)

if validation == False:
	x_valid = x_train
	y_valid = y_train
else:
	print "start validation"
	x_valid = x_train[validlen * fold: validlen * (fold+1)]
	y_valid = y_train[validlen * fold: validlen * (fold+1)]
	x_train = x_train[:validlen * fold] + x_train[validlen * (fold+1): ]
	y_train = y_train[:validlen * fold] + y_train[validlen * (fold+1): ]

x_train = np.array(x_train,dtype=np.float64)
y_train = np.array(y_train,dtype=np.float64)
x_valid = np.array(x_valid,dtype=np.float64)
y_valid = np.array(y_valid,dtype=np.float64)

for i in range(20000):
	loss = 0
	dw = np.zeros(w.shape,dtype=np.float64) #weight gradient
	db = 0. #bias gradient

	y = sigmoid(np.dot(x_train,w) + b)
	y_err = y_train - y

	dw += x_train.T.dot(y_err) #compute gradient
	db += y_err.sum() #compute gradient

	_y = sigmoid(np.dot(x_valid,w) + b)
	validaccr = accuracy(_y,y_valid)

	#regularization: 0.1 * 20000 : 0.923~0.924
	#dw += 0.1*w
	dw += 0.1 * (np.abs(w)) #regularization
	#adagrad
	g = g + (dw ** 2)
	gb = gb + (db ** 2)
	deltaw = (0.1/np.sqrt(g))*dw
	deltab = (0.1/np.sqrt(gb))*db
	w = w + deltaw
	b = b + deltab

	print "\rloop:%d validaccr:%f"%(i,validaccr),

if validation == False:
	np.savez(sys.argv[2],w=w,b=b)
else:
	print "nn.py, fold:%d, validaccr:%f"%(fold, accuracy(_y,y_valid))
