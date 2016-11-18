import sys
import pickle
import numpy as np
from keras.models import Sequential,Model
from keras.layers import *
from keras.optimizers import *
from keras.utils import np_utils
from keras.constraints import maxnorm
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import *
import keras
from keras import backend as K

def preprocess_data(X):
	scaler = StandardScaler()
	scaler.fit(X)
	X = scaler.transform(X)
	return X


def encoder(ae,fnum=(96,144,192)):
	K.set_image_dim_ordering('th')
	model = Sequential()

	model.add(Convolution2D(fnum[0],3,3, input_shape=(3,32,32),
		activation='relu',border_mode='same',weights=ae.layers[0].get_weights()))
	model.add(MaxPooling2D((2,2)))
	model.add(Convolution2D(fnum[1],3,3,
		activation='relu',border_mode='same',weights=ae.layers[2].get_weights()))
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(0.2))
	model.add(Convolution2D(fnum[2],3,3,
		activation='relu',border_mode='same',weights=ae.layers[4].get_weights()))
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(0.2))

	for layer in model.layers[0:1]:
		layer.trainable = False

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	#model.add(Dense(256, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(Activation('softmax'))
	model.summary()
	model.compile(optimizer = 'adam', loss='categorical_crossentropy',metrics=['accuracy'])
	#model.compile(optimizer=SGD(lr=1e-4,momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
	return model

def autoencoder(fnum=(96,144,192)):
	K.set_image_dim_ordering('th')
	model = Sequential()

	model.add(Convolution2D(fnum[0],3,3,
		activation='relu',border_mode='same',input_shape=(3,32,32)))
	model.add(MaxPooling2D((2,2)))
	model.add(Convolution2D(fnum[1],3,3,
		activation='relu',border_mode='same'))
	model.add(MaxPooling2D((2,2)))
	model.add(Convolution2D(fnum[2],3,3,
		activation='relu',border_mode='same'))
	model.add(MaxPooling2D((2,2)))

	model.add(Convolution2D(fnum[2],3,3,
		activation='relu',border_mode='same'))
	model.add(UpSampling2D((2,2)))
	model.add(Convolution2D(fnum[1],3,3,
		activation='relu',border_mode='same'))
	model.add(UpSampling2D((2,2)))
	model.add(Convolution2D(fnum[0],3,3,
		activation='relu',border_mode='same'))
	model.add(UpSampling2D((2,2)))
	model.add(Convolution2D(3,3,3,
		activation='relu',border_mode='same'))

	model.compile(optimizer = 'adam', loss='mse')
	return model

def allCNN(dprate=(0.2,0.5)):
	K.set_image_dim_ordering('th')
	model = Sequential()

	model.add(Dropout(dprate[0], input_shape=(3,32,32)))

	model.add(Convolution2D(96,3,3,
		activation='relu', border_mode='same'))
	model.add(Convolution2D(96,3,3,
		activation='relu', border_mode='same'))

	#model.add(Convolution2D(96,3,3,
		#activation='relu', border_mode='same', subsample=(2,2)))
	model.add(MaxPooling2D((3,3),strides=(2,2)))
	model.add(Dropout(dprate[1]))

	model.add(Convolution2D(192,3,3,
		activation='relu', border_mode='same'))
	model.add(Convolution2D(192,3,3,
		activation='relu', border_mode='same'))

	#model.add(Convolution2D(192,3,3,
		#activation='relu', border_mode='same', subsample=(2,2)))
	model.add(MaxPooling2D((3,3),strides=(2,2)))
	model.add(Dropout(dprate[1]))

	model.add(Convolution2D(192,3,3,
		activation='relu', border_mode='same'))
	model.add(Convolution2D(192,1,1,
		activation='relu', border_mode='same'))
	model.add(Convolution2D(10,1,1,
		activation='relu', border_mode='same'))
	model.add(GlobalAveragePooling2D())

	model.add(Dense(10))
	model.add(Activation('softmax'))

	model.summary()

	model.compile(loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['accuracy'])

	return model

def DCNN(convolution_num=3, fnum=(64,96,128), fsize=3, psize=2, dprate=0.2):
	K.set_image_dim_ordering('th')
	model = Sequential()

	model.add(Convolution2D(fnum[0], fsize, fsize,
		input_shape=(3,32,32), activation='relu', border_mode='same' ))
	model.add(Dropout(dprate))
	model.add(Convolution2D(fnum[0], fsize, fsize,
		activation='relu', border_mode='same'))
	model.add(MaxPooling2D((psize,psize)))

	for i in range(1,convolution_num):
		model.add(Convolution2D(fnum[i], fsize, fsize,
			activation='relu', border_mode='same'))
		model.add(Dropout(dprate))
		model.add(Convolution2D(fnum[i], fsize, fsize,
			activation='relu', border_mode='same'))
		model.add(MaxPooling2D((psize,psize)))

	model.add(Flatten())
	# 256 --> 0.67
	model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(dprate))
	model.add(Dense(10))
	model.add(Activation('softmax'))

	#model.summary()

	model.compile(loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['accuracy'])

	return model

def CNN(convolution_num=3, fnum=(32,64,128), fsize=3, psize=2, dprate=0.2):
	K.set_image_dim_ordering('th')
	model = Sequential()

	model.add(Convolution2D(fnum[0], fsize, fsize,
		input_shape=(3,32,32), activation='relu', border_mode='same' ))
	model.add(Dropout(dprate))
	model.add(Convolution2D(fnum[0], fsize, fsize,
		activation='relu', border_mode='same'))
	model.add(MaxPooling2D((psize,psize)))

	for i in range(1,convolution_num):
		model.add(Convolution2D(fnum[i], fsize, fsize,
			activation='relu', border_mode='same'))
		model.add(Dropout(dprate))
		model.add(Convolution2D(fnum[i], fsize, fsize,
			activation='relu', border_mode='same'))
		model.add(MaxPooling2D((psize,psize)))

	model.add(Flatten())
	# 256 --> 0.67
	model.add(Dense(256, activation='relu', W_constraint=maxnorm(3)))
	#model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(dprate))
	model.add(Dense(10))
	model.add(Activation('softmax'))

	#model.summary()

	model.compile(loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['accuracy'])

	return model

def imagegenerator():

	datagen = ImageDataGenerator(
    		rotation_range=20, #20
    		width_shift_range=0.2,
    		height_shift_range=0.2,
    		zoom_range=0.2,
    		shear_range=0.2,
    		horizontal_flip=True)
	return datagen

def load_labeldata(datapath, preprocess=True):
	print("start loading labeled data")
	path = datapath + 'all_label.p'
	all_label = pickle.load(open(path,'rb'))

	x_label,y_label = [],[]
	for i in range(len(all_label)):
		for j in range(len(all_label[i])):
			x_label.append(all_label[i][j])
			y_label.append(i)
	x_label = np.array(x_label, dtype='float64')
	if preprocess == True:
		x_label = preprocess_data(x_label).reshape((5000,3,32,32))
	else:
		x_label = x_label.reshape((5000,3,32,32))

	y_label = np.array(y_label)
	y_label = np_utils.to_categorical(y_label, 10)

	print("loading end")

	return x_label,y_label

def load_unlabeldata(datapath, loadnum, preprocess=True):
	print("start loading unlabeled data, loadnum = %d"%loadnum)
	path = datapath + 'all_unlabel.p'
	all_unlabel = pickle.load(open(path,'rb'))

	x_unlabel = np.array(all_unlabel[:loadnum], dtype='float64')
	if preprocess == True:
		x_unlabel = preprocess_data(x_unlabel).reshape((loadnum,3,32,32))
	else:
		x_unlabel = x_unlabel.reshape((loadnum,3,32,32))

	print("loading end")

	return x_unlabel

def load_testdata(datapath, preprocess=True):
	print("start loading testing data")
	path = datapath + 'test.p'
	testdata = pickle.load(open(path,'rb'))

	x_test,x_id = [],[]
	for i in range(10000):
		x_test.append(testdata['data'][i])
		x_id.append(testdata['ID'][i])
	x_test = np.array(x_test, dtype='float64')
	if preprocess == True:
		x_test = preprocess_data(x_test).reshape((10000,3,32,32))
	else:
		x_test = x_test.reshape((10000,3,32,32))

	print("loading end")

	return x_test,x_id

def output_result(result,id,datapath):
	with open(datapath,'w') as f:
		sys.stdout = f
		print("ID,class")
		for i in range(10000):
			print(id[i],",",np.argmax(result[i]))
		sys.stdout = sys.__stdout__
