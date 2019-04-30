# -*- coding: utf-8 -*-

import os
import cv2
import h5py
import numpy as np 
import tensorflow as tf
from PIL import Image 
from data_prepare import *
from tensorflow.python.framework import graph_util


def weight_variable(shape, name = 'conv1'):
	initial = tf.truncated_normal(shape, stddev = 0.01)
	return tf.Variable(initial, name = name)

def bias_variable(shape, name = 'bias1'):
	initial = tf.constant(0., shape = shape)
	return tf.Variable(initial, name = name)

def conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME'):
	return tf.nn.conv2d(x, W, strides = strides, padding = padding)

def pool(x, pool_type = 'max', ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
	padding = 'SAME'):
	if 'max' == pool_type:
		return tf.nn.max_pool(x, ksize = ksize, strides = strides, padding = padding)
	elif 'average' == pool_type:
		return tf.nn.avg_pool(x, ksize = ksize, strides = strides, padding = padding)


x = tf.placeholder(tf.float32, [None, 64, 64, 3], name = 'inputx')
y = tf.placeholder(tf.float32, [None, 1], name = 'inputy')

# 网络的第一层，卷积层
W_conv1 = weight_variable([5, 5, 3, 24], name = 'W1')
b_conv1 = bias_variable([24], 'b1')
a_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
a_conv1_pool = pool(a_conv1)

# 网络的第二层，卷积层
W_conv2 = weight_variable([5, 5, 24, 32], 'W2')
b_conv2 = bias_variable([32], 'b2')
a_conv2 = tf.nn.relu(conv2d(a_conv1_pool, W_conv2) + b_conv2)
a_conv2_pool = pool(a_conv2)

# 网络的第三层，全连接层
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
shape_new = int(np.prod(a_conv2_pool.shape[1:]))
A_input = tf.reshape(a_conv2_pool, [-1, shape_new])
W_fc1 = weight_variable([shape_new, 128], 'Wfc1')
b_fc1 = bias_variable([128], 'bfc1')
Z_fc1 = tf.matmul(A_input, W_fc1) + b_fc1
A_fc1 = tf.nn.relu(Z_fc1)
A_fc1_prob = tf.nn.dropout(A_fc1, keep_prob)

# 网络的第四层，全连接层
# W_fc2 = weight_variable([256, 128], 'Wfc2')
# b_fc2 = bias_variable([128])
# Z_fc2 = tf.matmul(A_fc1_prob, W_fc2) + b_fc2 
# A_fc2 = tf.nn.relu(Z_fc2)
# A_fc2_prob = tf.nn.dropout(A_fc2, keep_prob)

# 网络的第五层，产生预测
W_fc3 = weight_variable([128, 1], 'Wfc3')
b_fc3 = bias_variable([1], 'bfc3')
Z_fc3 = tf.matmul(A_fc1_prob, W_fc3) + b_fc3
A_fc3 = tf.nn.sigmoid(Z_fc3, name = 'output')

cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = A_fc3))
train_step = tf.train.AdamOptimizer(0.00001).minimize(cross_entropy, name = 'train_step')
predict = tf.cast(A_fc3 > 0.5, tf.int32, name = 'predict')


"""
从文件中读入数据
"""
# with h5py.File('train/train_happy.h5') as f:
# 	print(list(f.keys()))
# 	data1 = np.array(list(f['train_set_x']))
# 	label = np.array(list(f['train_set_y'])).reshape([-1, 1])

# data = []
# for img in data1:
# 	img = Image.fromarray(img)
# 	img = img.resize((64, 64))
# 	data.append(np.array(img))
# data = np.array(data) / 255
# print(label.shape)

with h5py.File('train/train.h5') as f:
	data = np.array(list(f['data'])) / 255
	label = np.array(list(f['label'])).reshape((-1, 1))

"""
保存为 .ckpt 文件
"""
def run(model_file_path, num_epoches = 51):
	# saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		mini_batches = get_batches(data, label, batch_size = 100)
		for epoch in range(num_epoches):
			for X_data, y_data in mini_batches:
				sess.run(train_step, feed_dict = {x: X_data, y: y_data, keep_prob: 0.7})
				cost = sess.run(cross_entropy, feed_dict = {x: X_data, y: y_data, keep_prob: 0.7})
				prediction = sess.run(A_fc3, feed_dict = {x: X_data, y: y_data, keep_prob: 1})

				W = sess.run(Z_fc3, feed_dict = {x: X_data, y: y_data, keep_prob: 0.7})
				prediction = np.where(prediction > 0.5, 1, 0).reshape(-1)
				print(prediction)
				print(y_data.reshape(-1))
				print('accuracy:', np.sum(prediction == y_data.reshape(-1))/len(y_data))
				print('Iter ' + str(epoch) + ' cost: ' + str(cost))
		# saver.save(sess, 'models/my_models.ckpt')
		constant_graph = graph_util.convert_variables_to_constants(sess, 
			sess.graph_def, ['predict'])
		with tf.gfile.FastGFile(model_file_path, mode = 'wb') as f:
			f.write(constant_graph.SerializeToString())

"""
保存为 .pb 文件
"""
def run1(model_file_path, num_epoches = 21):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		mini_batches = get_batches(data, label, batch_size = 50)
		for epoch in range(num_epoches):

			for X_data, y_data in mini_batches:
			# 	sess.run(train_step, feed_dict = {x: X_data, y: y_data, keep_prob: 0.7})
			# 	cost = sess.run(cross_entropy, feed_dict = {x: X_data, y: y_data, keep_prob: 0.7})
			# 	prediction = sess.run(A_fc3, feed_dict = {x: X_data, y: y_data, keep_prob: 1})

			# # print(prediction)
			# # print(y_data.reshape(-1))
			# 	prediction = np.where(prediction > 0.5, 1, 0).reshape(-1)
			# 	acc = np.sum(prediction == y_data.reshape(-1))/len(y_data)
			# 	print('accuracy:', acc)
			# 	print('prediction:\n', prediction.flatten())
			# 	print('ydata\n', y_data.flatten())
			# 	print('Iter ' + str(epoch) + ' cost: ' + str(cost))

				sess.run(train_step, feed_dict = {x: X_data, y: y_data, keep_prob: 0.7})
				cost = sess.run(cross_entropy, feed_dict = {x: X_data, y: y_data, keep_prob: 0.7})
				prediction = sess.run(A_fc3, feed_dict = {x: X_data, y: y_data, keep_prob: 1})

				W = sess.run(Z_fc3, feed_dict = {x: X_data, y: y_data, keep_prob: 0.7})
				prediction = np.where(prediction > 0.5, 1, 0).reshape(-1)
				print(prediction)
				print(y_data.reshape(-1))
				print('accuracy:', np.sum(prediction == y_data.reshape(-1))/len(y_data))
				print('Iter ' + str(epoch) + ' cost: ' + str(cost))

			# max_accuracy = 0 
			# min_cost = 1
			# if cost < min_cost and acc > max_accuracy:
			# 	constant_graph = graph_util.convert_variables_to_constants(sess, 
			# 		sess.graph_def, ['predict'])
			# 	with tf.gfile.FastGFile(model_file_path, mode = 'wb') as f:
			# 		f.write(constant_graph.SerializeToString())


run('models/constant_model.pb', 121)
# run('models/constant_model.pb')