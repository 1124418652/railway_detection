# -*- coding:utf-8 -*-
"""
Create on 25 March

Author: xhj
"""

import os
import cv2
import sys
import time
import numpy as np
from utils import *
from collections import defaultdict


def demo1():

	img_resource = cv2.imread('../v20190227/souce_images/12.png', 1)
	height, width = img_resource.shape[:2]
	img = img_resource[int(height / 2):, :, :]


	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	dst = cv2.GaussianBlur(gray_img, ksize = (7, 7), sigmaX = 3, sigmaY = 3)
	dst = cv2.bilateralFilter(dst, d = 10, sigmaColor = 100, sigmaSpace = 15)
	
	cv2.namedWindow('source', 2)
	# cv2.namedWindow('dst', 2)


	edge_image = cv2.Canny(dst, 50, 150)
	detection_with_houghlines(edge_image, threshold = 150, source_img = img, draw_lines = True)

	cv2.imshow("edge_image",edge_image)

	cv2.imshow('source', img)
	# cv2.imshow('dst', dst)

	cv2.waitKey(0)

def demo():

	import time

	begin = time.time()

	# line_detection_from_video('../视频1~5/3.mp4')
	img_resource = cv2.imread('../../v20190227/souce_images/00.png')


	gauss_img, roi_rect = pre_procession(img_resource, ksize = (13, 13), sigmaY = 3, sigmaX = 2, filter_type = 'bilateralFilter')
	edge_image = cv2.Canny(gauss_img, 50, 150)
	line_points = detection_with_houghlinesP(edge_image, threshold = 80, minLineLength = 60, maxLineGap = 15)
	if 0 == len(line_points):
		draw_lines = False

	print("time: ", time.time() - begin)
	
	for line in line_points:
		x1 = line[0] + roi_rect[0]
		y1 = line[1] + roi_rect[1]
		x2 = line[2] + roi_rect[0]
		y2 = line[3] + roi_rect[1]

		cv2.line(img_resource, (x1, y1), (x2, y2), (0, 0, 255), 2)
	# line_points, img_tmp, edge_image = single_frame_line_detection(img_resource)
	cv2.imshow('edge', edge_image)
	cv2.imshow('line', img_resource)
	cv2.waitKey(0)


def demo_with_mean():
	
	save_dir_path = 'test_image/resource'
	read_dir_path = '../../v20190227/souce_images/'

	if not os.path.exists(save_dir_path):
		os.makedirs(save_dir_path)

	print(os.listdir(read_dir_path))
	for file in os.listdir(read_dir_path):
		if not os.path.splitext(file)[-1] in ['.jpg', '.png']:
			print(file)
			continue

		start = time.time()

		img_resource = cv2.imread(os.path.join(read_dir_path, file))
		tmp = img_resource.copy()
		height, width = img_resource.shape[:2]
		gauss_img, roi_rect, gray_img = pre_procession(img_resource, ksize = (13,13), 
			sigmaX = 3, sigmaY = 3, filter_type = 'gaussian')
		edge_image = cv2.Canny(gauss_img, 50, 150)

		line_points, rho_theta = detection_with_houghlines(edge_image, threshold = 150)

		# print(line_points)
		points = defaultdict(list)
		for key, (cos_t, sin_t, x0, y0) in enumerate(rho_theta):
			r = (x0 + roi_rect[0]) * cos_t + (y0 + roi_rect[1]) * sin_t
			x = (r - height * sin_t) / cos_t
		
			if x <= width / 5:
				points[1].append((x, key))
			elif x <= width / 5 * 2:
				points[2].append((x, key))
			elif x <= width * 3 / 5:
				points[3].append((x, key))
			elif x <= width * 4 / 5:
				points[4].append((x, key))
			else:
				points[5].append((x, key))

			img_resource = cv2.circle(img_resource, (int(x), height), 15, (0, 0, 255), 8)
	

		for i in range(1,6):
			if 0 == len(points[i]):
				pass
			else:
				a = np.array(points[i])
				x_new = sum(a[:, 0]) / len(points[i])
			
				h = np.argmin(np.abs(a[:, 0] - x_new)) # 每个区域中离均值最近的点在该区域中的索引
				# print(h)
				tmp = cv2.circle(tmp, (int(points[i][h][0]), height), 15, (0, 0, 255), 8)
				x1, y1, x2, y2 = line_points[points[i][h][1]]
				x1 += roi_rect[0]
				y1 += roi_rect[1]
				x2 += roi_rect[0]
				y2 += roi_rect[1]
				cv2.line(tmp, (x1, y1), (x2, y2), (0, 0, 255), 4)

		print('time used:', time.time() - begin, 's')

		filename = os.path.splitext(file)[0]
		new_dir = os.path.join(save_dir_path, filename)
		if not os.path.exists(new_dir):
			os.makedirs(new_dir)
		cv2.imwrite(os.path.join(new_dir, 'gray_img.jpg'), gray_img)
		cv2.imwrite(os.path.join(new_dir, 'gauss_img.jpg'), gauss_img)
		cv2.imwrite(os.path.join(new_dir, 'edge_image.jpg'), edge_image)
		cv2.imwrite(os.path.join(new_dir, 'res_img.jpg'), tmp)


def demo_with_hist():

	save_dir_path = 'test_image/change_0.78V_channel'
	read_dir_path = '../../v20190227/souce_images/'

	if not os.path.exists(save_dir_path):
		os.makedirs(save_dir_path)

	for file in os.listdir(read_dir_path):
		if not os.path.splitext(file)[-1] in ['.jpg', '.png']:
			continue

		begin = time.time()
		print(file)

		img_resource = cv2.imread(os.path.join(read_dir_path, file))
		# img_resource = cv2.resize(img_resource, (int(img_resource.shape[1]/2),
		# 	int(img_resource.shape[0]/2)))
		print(img_resource.shape)
		tmp = cv2.cvtColor(img_resource, cv2.COLOR_BGR2HSV)
		tmp = cv2.merge((tmp[:,:,0], tmp[:,:,1], (tmp[:,:,2] * 0.6).astype(np.uint8)))
		tmp = cv2.cvtColor(tmp, cv2.COLOR_HSV2BGR)

		height, width = img_resource.shape[:2]
		gauss_img, roi_rect, gray_img = pre_procession(tmp, ksize = (13,13), 
			sigmaX = 3, sigmaY = 7, filter_type = 'gaussian', 
			roi_rect = [int(width/6), int(height*2/3), width-2*int(width/6)-1, 
						height-int(height*2/3)-1])
		edge_image = cv2.Canny(gauss_img, 50, 150)

		line_points, rho_theta = detection_with_houghlines(edge_image, threshold = 120)

		# print(line_points)
		points = defaultdict(list)
		for key, (cos_t, sin_t, x0, y0) in enumerate(rho_theta):
			r = (x0 + roi_rect[0]) * cos_t + (y0 + roi_rect[1]) * sin_t
			x = (r - height * sin_t) / cos_t
		
			if x <= width / 5:
				points[1].append((x, key))
			elif x <= width / 5 * 2:
				points[2].append((x, key))
			elif x <= width * 3 / 5:
				points[3].append((x, key))
			elif x <= width * 4 / 5:
				points[4].append((x, key))
			else:
				points[5].append((x, key))

			img_resource = cv2.circle(img_resource, (int(x), height), 15, (0, 0, 255), 8)
	

		for i in range(1,6):
			if 0 == len(points[i]):
				pass
			else:
				a = np.array(points[i])
				x_new = sum(a[:, 0]) / len(points[i])
			
				h = np.argmin(np.abs(a[:, 0] - x_new)) # 每个区域中离均值最近的点在该区域中的索引
				# print(h)
				tmp = cv2.circle(tmp, (int(points[i][h][0]), height), 15, (0, 0, 255), 8)
				x1, y1, x2, y2 = line_points[points[i][h][1]]
				x1 += roi_rect[0]
				y1 += roi_rect[1]
				x2 += roi_rect[0]
				y2 += roi_rect[1]
				cv2.line(tmp, (x1, y1), (x2, y2), (0, 0, 255), 4)

		print('time used: ', time.time() - begin, 's')

		filename = os.path.splitext(file)[0]
		new_dir = os.path.join(save_dir_path, filename)
		if not os.path.exists(new_dir):
			os.makedirs(new_dir)
		cv2.imwrite(os.path.join(new_dir, 'gray_img.jpg'), gray_img)
		cv2.imwrite(os.path.join(new_dir, 'gauss_img.jpg'), gauss_img)
		cv2.imwrite(os.path.join(new_dir, 'edge_image.jpg'), edge_image)
		cv2.imwrite(os.path.join(new_dir, 'res_img.jpg'), tmp)

	
if __name__ == '__main__':

	line_detection_from_video('../../sp1/轨道有异物1.mp4')