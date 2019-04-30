# -*- coding: utf-8 -*-
"""
Create on 22 April

Author: xhj
"""

import os
import cv2
import h5py
import numpy as np 


def image_crop(img, x, y, width, height):

	img_height, img_width = img.shape[:2]

	if not isinstance(img, np.ndarray) or y+height >= img_height or x+width >= img_width:
		pass

	else:
		return img[y: y+height, x: x+width]


def write_img():
	dir_path1 = '../C++/railwayDetection/railwayDetection/img/data_set/positive/'
	dir_path2 = '../C++/railwayDetection/railwayDetection/img/data_set/negtive/'
	data1 = []
	label1 = []
	for filename in os.listdir(dir_path1):
		filename = os.path.join(dir_path1, filename)
		img = cv2.imread(filename)
		img = cv2.resize(img, (64, 64))
		data1.append(img)
		label1.append(1)

	for filename in os.listdir(dir_path2):
		filename = os.path.join(dir_path2, filename)
		img = cv2.imread(filename)
		img = cv2.resize(img, (64, 64))
		data1.append(img)
		label1.append(0)

	if not os.path.exists('train/'):
		os.mkdir('train')

	with h5py.File('train/train.h5') as f:
		f['data'] = data1
		f['label'] = label1


def get_batches(X, y, batch_size, axis = 0, seed = 0):
    
    assert(X.shape[axis] == y.shape[axis])
    np.random.seed(seed)
    m = X.shape[axis]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    num_complete_minibatches = m // batch_size
    
    if 0 == axis:
        shuffled_X = X[permutation, :, :]
        shuffled_y = y[permutation, :]
        for k in range(num_complete_minibatches):
            mini_batch_X = shuffled_X[k * batch_size: (k + 1) * batch_size, :, :]
            mini_batch_y = shuffled_y[k * batch_size: (k + 1) * batch_size, :]
            mini_batches.append((mini_batch_X, mini_batch_y))
        if m % batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches * batch_size, :, :]
            mini_batch_y = shuffled_y[num_complete_minibatches * batch_size, :]
            mini_batches.append((mini_batch_X.reshape([-1, 64, 64, 3]), mini_batch_y.reshape(-1, 1)))
        return mini_batches
        
    elif 1 == axis:
        shuffled_X = X[:, permutation]
        shuffled_y = y[:, permutation]
        for k in range(num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * batch_size: (k + 1) * batch_size]
            mini_batch_y = shuffled_y[:, k * batch_size: (k + 1) * batch_size]
            mini_batches.append((mini_batch_X, mini_batch_y))
        if m % batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * batch_size]
            mini_batch_y = shuffled_y[:, num_complete_minibatches * batch_size]
            mini_batches.append((mini_batch_X, mini_batch_y))
        return mini_batches


# write_img()