import csv
import numpy as np
import random
from constants import hours, costsum, loopnum, usefeature
from numpy.linalg import inv

def adadelta(feature, w, b, g, gb, D, Db, cnt):
	dw = np.zeros((len(usefeature),hours), dtype=np.float64)
	db = 0

	#cnt = random.randint(0,5000-costsum)
	epoch = cnt + costsum

	loss = 0

	while(cnt != epoch):
		yhat = feature[9][cnt+hours]
		x = np.array([feature[i][cnt:cnt+hours] for i in usefeature], dtype=np.float64)
		y = np.sum(x * w) + b
		loss += (y-yhat)**2
		for i in range(len(usefeature)):
			for j in range(hours):
				dw[i,j] += 2*(y-yhat)*x[i,j]
		db += 2*(y-yhat)
		cnt += 10

	dw = dw + 2*100*w

	g = 0.95*g + 0.05*(dw ** 2)
	gb = 0.95*gb + 0.05*(db ** 2)

	eplison = 0.000001
	deltaw = ((np.sqrt(D+eplison)/np.sqrt(g+eplison))*dw)*-1
	deltab = ((np.sqrt(Db+eplison)/np.sqrt(gb+eplison))*db)*-1
	D = 0.95*D + 0.05*(deltaw**2)
	Db = 0.95*Db + 0.05*(deltab**2)
	w = w + deltaw
	b = b + deltab

	#return (w-((0.1/np.sqrt(g)) * dw) , b-((0.1/np.sqrt(gb)) * db), g, gb )
	return (w, b, g, gb, D, Db, loss)

def adagrad(feature, w, b, g, gb, cnt):
	dw = np.zeros((len(usefeature),hours), dtype=np.float64)
	db = 0

	loss = 0 # for early stop and observation
	maxcnt = cnt + costsum

	while(cnt != maxcnt ):
		yhat = feature[9][cnt+hours] #real pm2.5
		x = np.array([feature[i][cnt:cnt+hours] for i in usefeature], dtype=np.float64) #feature of pm2.5
		y = np.sum(x * w) + b #count estimated value
		loss += (y-yhat)**2
		for i in range(len(usefeature)):
			for j in range(hours):
				dw[i,j] += 2*(y-yhat)*x[i,j] #count (dL/dW)
		db += 2*(y-yhat) #count (dL/db)
		cnt += 10

	dw = dw + 2*100*w #regularization

	#adagrad part
	g = g + (dw ** 2)
	gb = gb + (db ** 2)

	deltaw = (1/np.sqrt(g))*dw*-1
	deltab = (1/np.sqrt(gb))*db*-1
	w = w + deltaw
	b = b + deltab

	return (w , b, g, gb, loss )

def sgd(feature, w, b, cnt):
	dw = np.zeros((len(usefeature),hours), dtype=np.float64)
	db = 0

	#cnt = random.randint(0,5000-costsum)
	epoch = cnt + costsum

	loss = 0

	while(cnt != epoch):
		yhat = feature[9][cnt+hours]
		x = np.array([feature[i][cnt:cnt+hours] for i in usefeature], dtype=np.float64)
		y = np.sum(x * w) + b
		loss += (y-yhat)**2
		for i in range(len(usefeature)):
			for j in range(hours):
				dw[i,j] = 2*(y-yhat)*x[i,j]
		db += 2*(y-yhat)
		cnt += 10
	#dw = dw + 2*100*w
		w = w - 0.000001*dw
		b = b - 0.000001*db

	return (w , b, loss )

def adam(feature, w, b, g, gb, m, mb, cnt, update):
	dw = np.zeros((len(usefeature),hours), dtype=np.float64)
	db = 0

	#cnt = random.randint(0,5000-costsum)
	epoch = cnt + costsum

	loss = 0

	while(cnt != epoch):
		yhat = feature[9][cnt+hours]
		x = np.array([feature[i][cnt:cnt+hours] for i in usefeature], dtype=np.float64)
		#for i in range(len(usefeature)):
		#	mean = np.sum(x[i]) / hours
		#	var = np.sum((x[i] - mean)**2)/hours
		#	x[i] = (x[i] - mean)/(var ** 0.5 + 1)
		y = np.sum(x * w) + b
		loss += (y-yhat)**2
		for i in range(len(usefeature)):
			for j in range(hours):
				dw[i,j] += 2*(y-yhat)*x[i,j]
		db += 2*(y-yhat)
		cnt += 10

	dw = dw + 2*100*w
	beta1 = 0.9
	beta2 = 0.999
	eplison = 0.00000001

	m = beta1*m + (1-beta1)*dw
	mb = beta1*mb + (1-beta1)*db
	g = beta2*g + (1-beta2)*(dw ** 2)
	gb = beta2*gb + (1-beta2)*(db ** 2)
	m_hat = m/(1-(beta1**update) )
	mb_hat = mb/(1-(beta1**update))
	g_hat = g/(1-(beta2**update))
	gb_hat = gb/(1-(beta2**update))
	deltaw = (0.001/(np.sqrt(g_hat)+eplison))*m*-1
	deltab = (0.001/np.sqrt(gb_hat)+eplison)*mb*-1
	w = w + deltaw
	b = b + deltab

	return (w , b, g, gb, m, mb, loss )

def normalize(feature):
	for i in usefeature:
		mean = sum(feature[i])/len(feature[i])
		var = sum([(x-mean)**2 for x in feature[i]])/len(feature[i])
		if var != 0:
			feature[i] = [(x-mean)/(var)**0.5 for x in feature[i]]
		else:
			feature[i] = [0 for x in feature[i]]
	return feature

def closedform(feature):
	tmp = [cnt*10 for cnt in range(len(feature[0])/10 )]
	featuretmp = [[] for i in tmp]
	for i in range(len(feature[0])/10 ):
		for j in usefeature:
			featuretmp[i] += feature[j][i*10:i*10+hours]
	#x = np.array([feature[9][cnt:cnt+hours] for cnt in tmp] )
	x = np.array(featuretmp, dtype=np.float64)
	y = np.array([ [feature[9][cnt+hours]] for cnt in tmp ], dtype=np.float64)

	w = (inv( (np.transpose(x)).dot(x) + 1000 ).dot(np.transpose(x))).dot(y)
	return w


feature = [[] for j in range(18)]

w = np.full((len(usefeature),hours),1,dtype=np.float64)
b = 1.
#adagrad
g = np.full((len(usefeature),hours),0,dtype=np.float64)
gb = 0.
#adadelta
D = np.full((len(usefeature),hours),0,dtype=np.float64)
Db = 0.
#adam
m = np.full((len(usefeature),hours),0,dtype=np.float64)
mb = 0.

# read train data
f = open('train3.csv', 'rb')
count = 0
for line in csv.reader(f):
	feature[count % 18].extend(map(float,line[3:]))
	count += 1

#train
loop = 0
loss = 0
update = 1
errormean = 0.
bestmean = 100000
while loop != loopnum:
	for cnt in range(len(feature[0])-costsum):
		w,b,g,gb,loss=adagrad(feature, w, b, g, gb, cnt)
		#w,b,g,gb,D,Db,loss=adadelta(feature, w, b, g, gb, D, Db, cnt)
		#w,b,loss=sgd(feature, w, b, cnt)
		#w,b,g,gb,m,mb,loss=adam(feature, w, b, g, gb, m, mb, cnt, update)
		update += 1
		loss = (loss/240)**0.5 #rms
		#print 'epoch:%d loss:%f\r'%(loop,loss),
		errormean += loss

	errormean /= (len(feature[0]) - costsum)
	#print 'epoch:%d errmean:%f'%(loop,errormean)
	if errormean < bestmean:
		bestmean = errormean
	else:
		break
	errormean = 0
	loop += 1


#test
out = open('linear_regression.csv', 'w')
print >> out, "id,value"
test = open('test2.csv', 'rb')
count = 0
feature = [[] for j in range(18)]
for line in csv.reader(test):
	feature[count % 18].extend(map(float,line[2:]))
	count += 1

#feature = normalize(feature)

for i in range(240):
	output = np.sum(np.array([feature[j][i*9 + 9 - hours:i*9 + 9] for j in usefeature], dtype=np.float64) * w) + b
	print >> out, "id_%d,%f" %(i, output)



f.close()