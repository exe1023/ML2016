import numpy as np
import sys
import csv
import matplotlib.pyplot as plt

def sigmoid(z,deriv=False):
	if deriv == True:
		return z * (1-z)
	return 1/(1+np.exp(-z))

def normalize(x):
	#return ( (x - x.min(axis=0)) / (x.ptp(axis=0) + 1e-20) )
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

validlen = (int)(4001/10)

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
x_train = normalize(x_train)
y_train = np.array(y_train,dtype=np.float64)
x_valid = np.array(x_valid,dtype=np.float64)
x_valid = normalize(x_valid)
y_valid = np.array(y_valid,dtype=np.float64)

#Neural Network

#neuron numbers = 31 or 39
alpha = 6
#neuralnum = int((4001/ (alpha * (57+1)) ))
#neuralnum = 45 #2/3 * size of input + size of output
neuralnum = 30 #(input+output)/2 + 1~10 7, now best

w0 = np.random.randn(57,neuralnum) * np.sqrt(2.0/(57 * neuralnum))
w1 = np.random.randn(neuralnum,neuralnum) * np.sqrt(2.0/(neuralnum ** 2))
w2 = np.random.randn(neuralnum,5) * np.sqrt(2.0/(neuralnum * 5))
w = np.random.randn(5,1) * np.sqrt(2.0/(5))

b0 = np.zeros((1,neuralnum))
b1 = np.zeros((1,neuralnum))
b2 = np.zeros((1,5))
b = np.zeros(1)

#adagrad initilize
g0 = np.full(w0.shape,1e-6,dtype=np.float64)
g1 = np.full(w1.shape,1e-6,dtype=np.float64)
g2 = np.full(w2.shape,1e-6,dtype=np.float64)
g = np.full(w.shape,1e-6,dtype=np.float64)

gb0 = np.full(b0.shape,1e-6)
gb1 = np.full(b1.shape,1e-6)
gb2 = np.full(b2.shape,1e-6)
gb = np.full(b.shape,1e-6)


learnrate = 0.1
valid_accr = []
validaccr = 0

for loopnum in range(750):
	x = x_train

	l0 = sigmoid(np.dot(x,w0) + b0)
	l1 = sigmoid(np.dot(l0,w1) + b1)
	l2 = sigmoid(np.dot(l1,w2) + b2)
	y = sigmoid(np.dot(l2,w) + b)

	#back propagation
	#sigmoid(deriv=True) means it output the derivative of sigmoid function
	y_error = y_train-y
	y_delta = y_error * sigmoid(y,deriv=True)

	l2_error = y_delta.dot(w.T)
	l2_delta = l2_error * sigmoid(l2,deriv=True)

	l1_error = l2_delta.dot(w2.T)
	l1_delta = l1_error * sigmoid(l1,deriv=True)

	l0_error = l1_delta.dot(w1.T)
	l0_delta = l0_error * sigmoid(l0,deriv=True)

	dw0 = x.T.dot(l0_delta)
	dw1 = l0.T.dot(l1_delta)
	dw2 = l1.T.dot(l2_delta)
	dw = l2.T.dot(y_delta)

	#adagrad
	g0 = g0 + (dw0 ** 2)
	g1 = g1 + (dw1 ** 2)
	g2 = g2 + (dw2 ** 2)
	g = g + (dw ** 2)

	deltaw0 = (learnrate/np.sqrt(g0))*dw0
	deltaw1 = (learnrate/np.sqrt(g1))*dw1
	deltaw2 = (learnrate/np.sqrt(g2))*dw2
	deltaw = (learnrate/np.sqrt(g))*dw
	w0 = w0 + deltaw0
	w1 = w1 + deltaw1
	w2 = w2 + deltaw2
	w = w + deltaw

	gb0 += np.mean(l0_delta, axis=0) ** 2
	gb1 += np.mean(l1_delta, axis=0) ** 2
	gb2 += np.mean(l2_delta, axis=0) ** 2
	gb += np.mean(y_delta, axis=0) ** 2
	b0 += (learnrate/np.sqrt(gb0)) * np.mean(l0_delta, axis=0)
	b1 += (learnrate/np.sqrt(gb1)) * np.mean(l1_delta, axis=0)
	b2 += (learnrate/np.sqrt(gb2)) * np.mean(l2_delta, axis=0)
	b += (learnrate/np.sqrt(gb)) * np.mean(y_delta, axis=0)

	#valid
	_l0 = sigmoid(np.dot(x_valid,w0) + b0)
	_l1 = sigmoid(np.dot(_l0,w1) + b1)
	_l2 = sigmoid(np.dot(_l1,w2) + b2)
	_y = sigmoid(np.dot(_l2,w) + b)
	validaccr = accuracy(_y,y_valid)

	print "\rloop:%d, validaccr:%f"%(loopnum,validaccr),
	valid_accr.append(validaccr)


np.savez(sys.argv[2], w=w, w0=w0, w1=w1, w2=w2, b=b, b0=b0, b1=b1, b2=b2)
