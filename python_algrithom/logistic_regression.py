# -*- coding: utf-8 -*-
import os
import cv2
import sys
import h5py
import numpy as np


class LRModel():

	def forward_prop(self, W, b, X):
		assert(W.shape[1] == X.shape[0])
		Z = np.matmul(W, X) + b
		A = 1 / (1 + np.exp(-Z))
		return A

	def _get_mini_batch(self, X, y, batch_size = 100, shuffle = True, seed = 0):
		m = len(X[0])
		if True == shuffle:
			np.random.seed(seed = 0)
			permutation = np.random.permutation(m).tolist()
			X_shuffled = X[:, permutation]
			y_shuffled = y[permutation]
		else:
			X_shuffled = X 
			y_shuffled = y

		num_batches = m // batch_size
		mini_batches = []
		for k in range(num_batches):
			X_batch = X_shuffled[:, k * batch_size: (k+1) * batch_size]
			y_batch = y_shuffled[k * batch_size: (k+1) * batch_size]
			mini_batches.append((X_batch, y_batch))
		if m % batch_size != 0:
			X_batch = X_shuffled[:, num_batches * batch_size:]
			y_batch = y_shuffled[num_batches * batch_size:]
			mini_batches.append((X_batch, y_batch))
		return mini_batches

	def fit(self, X, y, epochs = 100, lr = 0.001, batch_size = 256, shuffle = True):
		"""
		执行模型训练的过程，使用梯度下降法对模型进行拟合

		Args:
			X: 输入数据，维数为:(feature_num, sample_num)
			y: 输入数据对应的标签，0 表示负类，1 表示正类，维数为 (sample_num)
		"""

		feature_num = len(X)
		self.W = np.random.randn(1, feature_num)
		self.b = 0

		for epoch in range(epochs):
			for (X_batch, y_batch) in self._get_mini_batch(X, y, batch_size, shuffle):
				A = self.forward_prop(self.W, self.b, X_batch)
				loss = -np.mean(y_batch * np.log(A) + (1 - y_batch) * np.log(1 - A))
				dZ = (A - y_batch) / batch_size
				dW = np.matmul(dZ, X_batch.T)
				db = np.sum(dZ)
				self.W -= lr * dW 
				self.b -= lr * db
			print("cost of epoch %d: %s" %(epoch, loss))
			A_predict = np.where(self.forward_prop(self.W, self.b, X) > 0.5, 1, 0)
			precision = np.equal(A_predict, y).sum() / len(y)
			print("precision: ", precision)
			print("\n")
		return self.W, self.b

	def _predict(self, X, W, b):
		A = self.forward_prop(W, b, X)
		return np.where(A > 0.5, 1, 0).ravel().tolist()


def save_model(W, b, file_path = 'models/logistic_model.txt'):
	print("begin saving to file: ", file_path)
	if os.path.exists(file_path):
		os.remove(file_path)
	with h5py.File(file_path) as fw:
		fw['W'] = W
		fw['b'] = b
	print("finish saving.")


# W = None 
# b = None
# with h5py.File('F:/小弦科技实习/铁轨检测/工作资料/railway_detection/C++/railwayDetection/x64/Debug/models/logistic_model.txt') as fr:
# 	W = np.array(list(fr['W']))
# 	b = np.array(fr['b'])

# def load_model():
# 	print("loading model...")

# def predict(img, width, height, channel):
# 	num = len(img)
# 	if num == 0:
# 		raise ValueError("The image is empty!")
# 	img = np.uint8(img)
# 	img = np.array(img).reshape((-1, height, width, channel))
# 	modify_img = []

# 	for tmp in img:
# 		modify_img.append(cv2.resize(tmp, (64, 64)) / 255.0)
# 	modify_img = np.array(modify_img).reshape((-1, 64 * 64 * 3)).T
# 	model = LRModel()
# 	return model._predict(modify_img, W, b)


if __name__ == '__main__':
	file_path = 'train/train.h5'
	datafile = h5py.File(file_path, 'r')
	print(list(datafile.keys()))
	data = np.array(list(datafile['data'])).reshape(-1, 64*64*3).T / 255.0
	label = np.array(list(datafile['label']))
	model = LRModel()
	W, b = model.fit(data, label, epochs = 1500)
	save_model(W, b)


	# print(predict(data[1000:1100], 64, 64, 3))