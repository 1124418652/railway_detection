# -*- coding: utf-8 -*-
"""
Create on 2019/5/8

Author: xhj
"""

import os
import cv2
import time
import json
import h5py
import numpy as np 


def load_parameters(filename):
	print("done2")
	if not os.path.exists(filename):
		print("donebbb")
		raise ValueError("The file is not exist!")
	print("done3")
	with open(filename, 'r') as fr:
		param = json.load(fr)
		W_conv1 = np.array(param['W1'])
		b_conv1 = np.array(param['b1'])
		W_conv2 = np.array(param['W2'])
		b_conv2 = np.array(param['b2'])
		Wfc1 = np.array(param['W3'])
		bfc1 = np.array(param['b3'])
		Wfc2 = np.array(param['W4'])
		bfc2 = np.array(param['b4'])
	return [W_conv1, b_conv1, W_conv2, b_conv2, Wfc1, bfc1, Wfc2, bfc2]


class CNN_model(object):
	def __init__(self, W_conv1, b_conv1, W_conv2, b_conv2, Wfc1, bfc1, Wfc2, bfc2):
		self.W_conv1 = W_conv1 
		self.b_conv1 = b_conv1
		self.W_conv2 = W_conv2 
		self.b_conv2 = b_conv2 
		self.Wfc1 = Wfc1
		self.bfc1 = bfc1 
		self.Wfc2 = Wfc2 
		self.bfc2 = bfc2

	def zero_pad(self, X, pad, value = 0):
		X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
			'constant', constant_values = value)
		return X_pad

	def conv_single_step(self, a_slice_prev, W, b):
		s = a_slice_prev * W + b
		Z = np.sum(s)
		return Z

	def conv_forward(self, A_prev, W, b, stride, padding):
		if not A_prev.ndim == W.ndim == 4:
			raise ValueError("Dimension of A_prev and W must be 4")
		assert(A_prev.shape[3] == W.shape[2])
		(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
		(f, f, _, n_C) = W.shape
		n_H = (n_H_prev + 2 * padding - f) // stride + 1
		n_W = (n_W_prev + 2 * padding - f) // stride + 1

		Z = np.zeros((m, n_H, n_W, n_C))
		A_prev_pad = self.zero_pad(A_prev, padding)

		for h in range(n_H):
			for w in range(n_W):
				for c in range(n_C):
					vert_start = h * stride 
					vert_end = vert_start + f 
					horiz_start = w * stride 
					horiz_end = horiz_start + f
					A_slice_prev = A_prev_pad[:, vert_start: vert_end, horiz_start: horiz_end, :]
					Z[:, h, w, c] = (A_slice_prev * W[:, :, :, c] + b[c]).sum(axis = (1, 2, 3)).flatten()

		A = np.where(Z > 0, Z, 0)
		# print('time used of conv_forward: ', time.time() - begin)
		return A

	def conv_forward_opencv(self, A_prev, W, b, stride, padding):
		# begin = time.time()
		if not A_prev.ndim == W.ndim == 4:
			raise ValueError("Dimension of A_prev and W must be 4")
		assert(A_prev.shape[3] == W.shape[2])

		(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
		(f, f, _, n_C) = W.shape
		n_H = (n_H_prev + 2 * padding - f) // stride + 1
		n_W = (n_W_prev + 2 * padding - f) // stride + 1

		Z = np.zeros((m, n_H, n_W, n_C))
		A_prev_pad = self.zero_pad(A_prev, padding)

		for i in range(m):
			for c in range(n_C):
				for c_prev in range(n_C_prev):
					W_inv = W[:, :, c_prev, c]
					# print(W_inv)
					# W_inv = cv2.flip(W_inv, -1)
					# print(W_inv)
					# row, col = W_inv.shape
					# for r in range(row // 2):
					# 	W_inv[r], W_inv[row - 1 - r] = W_inv[row - 1 - r], W_inv[r]
					# for m in W_inv:
					# 	for j in range(col // 2):
					# 		m[j], m[col - 1 - j] = m[col - 1 - j], m[j]
					# print(W_inv)
					Z[i, :, :, c] += cv2.filter2D(A_prev[i, :, :, c_prev], -1, 
						W_inv)
		A = np.where(Z > 0, Z, 0)
		# print("time used of conv_forward_opencv: ", time.time() - begin)
		return A

	def pool_forward(self, A_prev, f, stride, mode = 'max'):
		(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
		n_H = (n_H_prev - f) // stride + 1 
		n_W = (n_W_prev - f) // stride + 1
		n_C = n_C_prev

		A = np.zeros((m, n_H, n_W, n_C))

		for h in range(n_H):
			for w in range(n_W):
				vert_start = h * stride
				vert_end = vert_start + f 
				horiz_start = w * stride
				horiz_end = horiz_start + f
				A_slice_prev = A_prev[:, vert_start: vert_end, horiz_start: horiz_end, :]
				A[:, h, w, :] = np.max(A_slice_prev, axis = (1, 2))
		
		return A

	def dense_forward(self, A_prev, W, b, activate_func = 'relu'):
		assert(A_prev.shape[1] == W.shape[0])
		Z = np.matmul(A_prev, W) + b.T
		if 'relu' == activate_func:
			A = np.where(Z > 0, Z, 0)
		elif 'sigmoid' == activate_func:
			A = 1 / (1 + np.exp(-Z))
		return A

	def model(self, X):
		A1 = self.conv_forward_opencv(X, self.W_conv1, self.b_conv1, 1, 2)
		A1_pool = self.pool_forward(A1, 2, 2)
		A2 = self.conv_forward_opencv(A1_pool, self.W_conv2, self.b_conv2, 1, 2)
		A2_pool = self.pool_forward(A2, 2, 2)
		A2_pool = A2_pool.reshape((A2_pool.shape[0], -1))
		A3 = self.dense_forward(A2_pool, self.Wfc1, self.bfc1)
		A4 = self.dense_forward(A3, self.Wfc2, self.bfc2, 'sigmoid')
		return A4

	def _predict(self, X):
		A = self.model(X)
		prediction = np.where(A > 0.5, 1, 0)
		return prediction.ravel().tolist()

print("loading model's parameters...")
W_conv1, b_conv1, W_conv2, b_conv2, Wfc1, bfc1, Wfc2, bfc2 = load_parameters('F:/小弦科技实习/铁轨检测/工作资料/railway_detection/C++/railwayDetection/x64/Debug/models/parameters.txt')
print("finish loading model's parameters.")
print("create model...")
cnn_model = CNN_model(W_conv1, b_conv1, W_conv2, b_conv2, Wfc1, bfc1, Wfc2, bfc2)
print("finish creating model.")

def load_model():
	print("done11111111")
	# W_conv1, b_conv1, W_conv2, b_conv2, Wfc1, bfc1, Wfc2, bfc2 = load_parameters('F:/小弦科技实习/铁轨检测/工作资料/railway_detection/C++/railwayDetection/x64/Debug/models/parameters.txt')
	# print("done2")
	# cnn_model = CNN_model(W_conv1, b_conv1, W_conv2, b_conv2, Wfc1, bfc1, Wfc2, bfc2)
	# print("done2")

def predict(img, width, height, channel):
	num = len(img)
	if num == 0:
		raise ValueError("The image is empty!")
	img = np.uint8(img)
	img = np.array(img).reshape((-1, height, width, channel))
	modify_img = []

	for tmp in img:
		modify_img.append(cv2.resize(tmp, (64, 64)) / 255.0)
	modify_img = np.array(modify_img)
	return cnn_model._predict(modify_img)


if __name__ == '__main__':
	W_conv1, b_conv1, W_conv2, b_conv2, Wfc1, bfc1, Wfc2, bfc2 = load_parameters('models/parameters.txt')
	cnn_model = CNN_model(W_conv1, b_conv1, W_conv2, b_conv2, Wfc1, bfc1, Wfc2, bfc2)
	X = np.random.randint(1, 10, (3, 64, 64, 3))
	with h5py.File('train/train.h5') as fr:
		data = (np.array(list(fr['data'])) / 255)[0:100]
		label = np.array(list(fr['label'])).reshape((-1, 1))[1500:1600]
	# print(W_conv2.shape)
	# print(data.shape)
	data = data.flatten()
	begin = time.time()
	# print(cnn_model._predict(data))
	# print(label.flatten())

# 	load_model()
# 	predict(data, 64, 64, 3)

# 	print('time used: ', time.time() - begin)
	# cnn_model.conv_forward_opencv(data, W_conv1, b_conv1, 1, 2)
	# cnn_model.conv_forward(data, W_conv1, b_conv1, 1, 2)