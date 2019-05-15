# -*- coding: utf-8 -*-
"""
Create on 2019/5/8

@Author: xhj
"""

import os
import cv2
import h5py
import json
import numpy as np
import tensorflow as tf 


def get_parameters():
	with tf.Graph().as_default() as g:
		output_graph_def = tf.GraphDef()
		pb_file_path = 'models/constant_model.pb'
		
		with open(pb_file_path, 'rb') as f:
			output_graph_def.ParseFromString(f.read())
			tf.import_graph_def(output_graph_def, name = '')

		with tf.Session(graph = g) as sess:
			W_conv1 = sess.graph.get_tensor_by_name('W1:0')
			b_conv1 = sess.graph.get_tensor_by_name('b1:0')
			W_conv2 = sess.graph.get_tensor_by_name('W2:0')
			b_conv2 = sess.graph.get_tensor_by_name('b2:0')
			W_fc1 = sess.graph.get_tensor_by_name('Wfc1:0')
			b_fc1 = sess.graph.get_tensor_by_name('bfc1:0')
			W_fc3 = sess.graph.get_tensor_by_name('Wfc3:0')
			b_fc3 = sess.graph.get_tensor_by_name('bfc3:0')

			W1, b1, W2, b2, W3, b3, W4, b4 = sess.run([W_conv1, b_conv1,
				W_conv2, b_conv2, W_fc1, b_fc1, W_fc3, b_fc3])
			W1 = W1.tolist()
			b1 = b1.tolist()
			W2 = W2.tolist()
			b2 = b2.tolist()
			W3 = W3.tolist()
			b3 = b3.tolist()
			W4 = W4.tolist()
			b4 = b4.tolist()
	return [W1, b1, W2, b2, W3, b3, W4, b4]

def to_file(d, filename = 'models/parameters.txt'):
	with open(filename, 'w') as f:
		json.dump(d, f)

def conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME'):
	return tf.nn.conv2d(x, W, strides = strides, padding = padding)

def pool(x, pool_type = 'max', ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
	padding = 'SAME'):
	if 'max' == pool_type:
		return tf.nn.max_pool(x, ksize = ksize, strides = strides, padding = padding)
	elif 'average' == pool_type:
		return tf.nn.avg_pool(x, ksize = ksize, strides = strides, padding = padding)

def reconstruct_model(X, y, W1, b1, W2, b2, W3, b3, W4, b4):
	x = tf.placeholder(tf.float32, [None, 64, 64, 3], name = 'inputx')
	y = tf.placeholder(tf.float32, [None, 1], name = 'inputy')

	W_conv1 = tf.Variable(W1, name = 'W_conv1')
	b_conv1 = tf.Variable(b1, name = 'b_conv1')
	a_conv1 = tf.nn.relu()


if __name__ == '__main__':
	# [W1, b1, W2, b2, W3, b3, W4, b4] = get_parameters()
	# d = dict(W1 = W1, b1 = b1, W2 = W2, b2 = b2, W3 = W3, b3 = b3, W4 = W4, b4 = b4)
	# to_file(d)
	# with open('models/parameters.txt', 'rb') as f:
	# 	param = json.load(f)
	# 	print(param['W1'])
	a = cv2.imread('1903.jpg')
	b = np.random.randint(1, 10, (100, 100), dtype = np.uint8)
	print(a.dtype, b.dtype)
	f = np.ones((3, 3)) / 9
	cv2.filter2D(b, -1, f)