import sys
import pickle
import numpy as np
from keras.models import Sequential, load_model
from sklearn.preprocessing import StandardScaler
from common import *
from keras.utils import np_utils
from keras.preprocessing.image import *
from keras import backend as K
#import tensorflow as tf
#tf.python.control_flow_ops = tf

def pseudolabel(x,result,thr=0.9):

	x_label,y_label = [],[]
	labeled_index = []
	labeled = 0

	print("label start. use the data with confidence >= %f"%thr)
	for i in range(result.shape[0]):
		if np.amax(result[i]) >= thr:
			index = np.argmax(result[i])
			x_label.append(x[i])
			y_label.append(index)
			labeled_index.append(i)
			labeled += 1

	usenum = -1
	x = np.delete(x,labeled_index,0)
	x_label = np.array(x_label[:usenum])
	y_label = np.array(y_label[:usenum])
	y_label = np_utils.to_categorical(y_label, 10)

	print("label end, %d labeled, use %d data"%(labeled, usenum) )

	return x,x_label,y_label

def main():
	K.set_image_dim_ordering('th')
	x_label,y_label = load_labeldata(sys.argv[1])

	model = CNN()
	datagen = imagegenerator()
	datagen.fit(x_label)
	model.fit_generator(datagen.flow(x_label, y_label, batch_size=32),
                   samples_per_epoch=len(x_label), nb_epoch=130)
	#model.save('cnn.h5')

	x_unlabel = load_unlabeldata(sys.argv[1], 45000)

	result = model.predict(x_unlabel)
	x_unlabel,tmpx,tmpy = pseudolabel(x_unlabel,result)
	x_label = np.concatenate((x_label,tmpx),axis=0)
	y_label = np.concatenate((y_label,tmpy),axis=0)

	model = CNN()
	datagen = imagegenerator()
	datagen.fit(x_label)
	model.fit_generator(datagen.flow(x_label, y_label, batch_size=32),
                   samples_per_epoch=len(x_label), nb_epoch=30)

	model.save(sys.argv[2])
	#x_test,x_id = load_testdata(sys.argv[1])

	#result = model.predict(x_test)
	#output_result(result, x_id, sys.argv[2])



if __name__ == '__main__':
	main()