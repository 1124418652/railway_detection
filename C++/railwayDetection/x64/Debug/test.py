import os
import tensorflow as tf 
import h5py
import numpy as np 
import cv2


output_graph_def = tf.GraphDef()
pb_file_path = "models/constant_model.pb"

with h5py.File('../../../../python_algrithom/train/train.h5') as fr:
	data = (np.array(list(fr['data'])) / 255)[1000:1100]
	label = np.array(list(fr['label'])).reshape((-1, 1))[1000:1100]

print(data.shape)

with open(pb_file_path, 'rb') as f:
	output_graph_def.ParseFromString(f.read())
	tf.import_graph_def(output_graph_def, name = '')
	print('done')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	input_x = sess.graph.get_tensor_by_name('inputx:0')
	prediction = sess.graph.get_tensor_by_name('predict:0')
	keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
	A_fc3 = sess.graph.get_tensor_by_name('output:0')
	pre = sess.run(prediction, feed_dict = {input_x: data, keep_prob:1})
	print(pre.ravel().tolist())