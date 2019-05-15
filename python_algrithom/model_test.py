# -*- coding: utf-8 -*-

import os
import cv2
import h5py
import numpy as np
import tensorflow as tf 


def test_model():
	with h5py.File('train/train.h5') as f:
		data = np.array(list(f['data'])) / 255
		label = np.array(list(f['label'])).reshape((-1, 1))


	with tf.Session() as sess:
		saver = tf.train.import_meta_graph('models/my_models.ckpt.meta')       # 取出模型的计算图
		saver.restore(sess, 'models/my_models.ckpt')

		x = sess.graph.get_tensor_by_name('inputx:0')
		y = sess.graph.get_tensor_by_name('inputy:0')
		output = sess.graph.get_tensor_by_name('output:0')
		train_step = sess.graph.get_operation_by_name('train_step')

		res = sess.run(output, feed_dict = {x: data, y: label, 'keep_prob:0': 1})

		prediction = np.where(res > 0.5, 1, 0)
		acc = np.sum(prediction == label) / len(label)

		print(acc)


# with tf.Graph().as_default() as g:
# 	output_graph_def = tf.GraphDef()
# 	pb_file_path = 'models/constant_model.pb'
# 	with open(pb_file_path, 'rb') as f:
# 		output_graph_def.ParseFromString(f.read())
# 		# 将计算图从 output_graph_def 中导入到当前的默认图中
# 		tf.import_graph_def(output_graph_def, name='')   

# 		"""
# 		提取参数的内容
# 		"""
# 	with tf.Session(graph = g) as sess:
# 		W_conv1 = sess.graph.get_tensor_by_name('W1:0')
# 		b_conv1 = sess.graph.get_tensor_by_name('b1:0')
# 		W_conv2 = sess.graph.get_tensor_by_name('W2:0')
# 		b_conv2 = sess.graph.get_tensor_by_name('b2:0')
# 		W_fc1 = sess.graph.get_tensor_by_name('Wfc1:0')
# 		b_fc1 = sess.graph.get_tensor_by_name('bfc1:0')
# 		W_fc3 = sess.graph.get_tensor_by_name('Wfc3:0')
# 		b_fc3 = sess.graph.get_tensor_by_name('bfc3:0')

# 		W1, b1, W2, b2, W3, b3, W4, b4 = sess.run([W_conv1, b_conv1,
# 			W_conv2, b_conv2, W_fc1, b_fc1, W_fc3, b_fc3])


def predict(img, width, height, channel):
	"""
	通过调用tensorflow下训练好的pb模型来预测输入图片的结果

	Args:
		img: Python的list类型数据（一维数据），保存了图片数据
		width: int类型数据，表示图片的宽度
		height: int类型数据，表示图片的高度
		channel: int类型数据，表示图片的通道数
	"""
	
	if len(img) == 0:
		raise ValueError("The image is empty!")

	if len(img) != width * height * channel:
		raise ValueError("Data Length Error")

	img = np.array(img).reshape((height, width, channel))
	img = cv2.resize(img, (64, 64)) / 255.0

	with tf.Session(graph = g) as sess:
		sess.run(tf.global_variables_initializer())
		input_x = sess.graph.get_tensor_by_name('inputx:0')
		prediction = sess.graph.get_tensor_by_name('predict:0')
		keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')

		pre = sess.run(prediction, feed_dict = {input_x: [img], keep_prob:1})
		return pre.ravel()[0]

if __name__ == '__main__':
	with h5py.File('train/train.h5') as fr:
		data = (np.array(list(fr['data'])) / 255)[0:100]
		label = np.array(list(fr['label'])).reshape((-1, 1))[0:100]
	print(predict(data))