# -*- coding: utf-8 -*-

import os
import cv2
import h5py
import numpy as np
import tensorflow as tf 

# flag = 0
# with tf.Graph().as_default() as g:
# 	flag += 1
sess = tf.Session(graph = tf.get_default_graph())

def load_model():
	output_graph_def = tf.GraphDef()
	pb_file_path = "F:/小弦科技实习/铁轨检测/工作资料/railway_detection/C++/railwayDetection/x64/Debug/models/constant_model.pb"
	with open(pb_file_path, 'rb') as f:
		output_graph_def.ParseFromString(f.read())
		# 将计算图从 output_graph_def 中导入到当前的默认图中
		tf.import_graph_def(output_graph_def, name='')   
	print("done")

def predict(img, width, height, channel):
	"""
	通过调用tensorflow下训练好的pb模型来预测输入图片的结果

	Args:
		img: Python的list类型数据（一维数据），保存了图片数据
		width: int类型数据，表示图片的宽度
		height: int类型数据，表示图片的高度
		channel: int类型数据，表示图片的通道数
	"""
	num = len(img)
	if num == 0:
		raise ValueError("The image is empty!")

	# if len(img) != width * height * channel:
	# 	raise ValueError("Data Length Error")
	# print("call function")
	img = np.uint8(img)
	img = np.array(img).reshape((-1, height, width, channel))
	modify_img = []
	# print("to numpy")
	for tmp in img:
		modify_img.append(cv2.resize(tmp, (64, 64)) / 255.0)
	modify_img = np.array(modify_img)
	
	sess.run(tf.global_variables_initializer())
	input_x = sess.graph.get_tensor_by_name('inputx:0')
	prediction = sess.graph.get_tensor_by_name('predict:0')
	keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
	# print("keep")
	pre = sess.run(prediction, feed_dict = {input_x: modify_img, keep_prob:1})
	return pre.ravel().tolist()

# if __name__ == '__main__':
# 	with h5py.File('../../../../python_algrithom/train/train.h5') as fr:
# 		data = (np.array(list(fr['data'])) / 255)[0:10]
# 		label = np.array(list(fr['label'])).reshape((-1, 1))[0:100]
# 	load_model()
# 	print(predict(data, 64, 64, 3))