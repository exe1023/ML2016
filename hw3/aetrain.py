import sys
import pickle
import numpy as np
from keras.models import Sequential, load_model, Model
from sklearn.preprocessing import StandardScaler
from common import *
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras import backend as K
#import tensorflow as tf
#tf.python.control_flow_ops = tf


def main():
	K.set_image_dim_ordering('th')
	x_label,y_label = load_labeldata(sys.argv[1])
	x_test,x_id = load_testdata(sys.argv[1])

	ae = autoencoder()
	x_unlabel = load_unlabeldata(sys.argv[1], 45000)
	x_all = np.concatenate((x_label,x_unlabel),axis=0)
	#x_all = np.concatenate((x_all,x_test),axis=0)

	ae.fit(x_all,x_all,nb_epoch=30,batch_size=32)
	#ae.save('ae.h5')

	print("training of encoder end. use labeled data to tune model")

	model = encoder(ae)

	datagen = imagegenerator()
	datagen.fit(x_label)
	#model.fit(x_label,y_label,nb_epoch=40,batch_size=32)
	model.fit_generator(datagen.flow(x_label, y_label, batch_size=32),
                    samples_per_epoch=len(x_label), nb_epoch=100)
	model.save(sys.argv[2])
	#print(model.evaluate(x_label,y_label))
	#result = model.predict(x_test)
	#output_result(result, x_id, sys.argv[2])

if __name__ == '__main__':
	main()
