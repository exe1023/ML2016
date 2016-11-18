import sys
import pickle
import numpy as np
from keras.models import Sequential, load_model
from sklearn.preprocessing import StandardScaler
from common import *
#import tensorflow as tf
#tf.python.control_flow_ops = tf

def main():
	model = load_model(sys.argv[1])
	x_test,x_id = load_testdata(sys.argv[2])
	x_label,y_label = load_labeldata(sys.argv[2])

	print(model.evaluate(x_label,y_label))
	result = model.predict(x_test)
	output_result(result, x_id, sys.argv[3])


if __name__ == '__main__':
	main()