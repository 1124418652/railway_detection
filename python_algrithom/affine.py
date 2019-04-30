# -*- coding: utf-8 -*-
"""
Create on 4 April

Author: xhj
"""

import os
import cv2
import time
import argparse
import numpy as np 
from utils import *
# from data_prepare import *


parser = argparse.ArgumentParser(description = 'affine transformation')
parser.add_argument('-p','--image_path', type = str, 
	default = '../../v20190227/souce_images/m2.jpg',
	help = 'The file path of the image you want to used.')
args = parser.parse_args()
IMAGE_PATH = args.image_path

if not os.path.exists(IMAGE_PATH):
	raise ValueError("Can't find the image in %s" % (IMAGE_PATH))


def image_crop(img, x, y, width, height):

	img_height, img_width = img.shape[:2]

	if not isinstance(img, np.ndarray) or y+height >= img_height or x+width >= img_width:
		pass

	else:
		return img[y: y+height, x: x+width]


if __name__ == '__main__':
	
	image_source = cv2.imread(IMAGE_PATH)
	if not isinstance(image_source, np.ndarray):
		raise ValueError("Can't open the image.")

	line_points, tmp, edge_image = single_frame_line_detection(image_source)

	if 2 == len(line_points):
		x1, y1, x2, y2 = line_points[0]
		x3, y3, x4, y4 = line_points[1]

		src = np.float32([[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]])
		dst = np.float32([[(x1, y1), (x1, 0), (x3, y3), (x3, 0)]])

		line1 = [(x1, y1), (x1, 0)]
		line2 = [(x3, y3), (x3, 0)]

		M = cv2.getPerspectiveTransform(src, dst)       # 获取透视变换矩阵
		Minv = cv2.getPerspectiveTransform(dst, src)    # 获取透视变换的逆矩阵

		warped_img = cv2.warpPerspective(image_source, M, image_source.shape[1::-1], 
										 flags=cv2.INTER_LINEAR)

		i = 0
		dir_path = '../../v20190227/souce_images/data/'
		for y in range(0, y1 - 60, 5):
			for x in range(x1 - 100, x1 + 90, 7):
				data = image_crop(warped_img, x, y, 100, 100)
				cv2.imwrite(os.path.join(dir_path, str(i)+'.jpg'), data)
				print(i)
				i += 1
		
		# 选取直线两侧的区域作为铁轨表面的候选区域
		roi_img1 = warped_img[:, line1[0][0] - 40: line1[0][0], :]
		roi_img2 = warped_img[:, line1[0][0]: line1[0][0] + 40, :]
		roi_img3 = warped_img[:, line2[0][0] - 40: line2[0][0], :]
		roi_img4 = warped_img[:, line2[0][0]: line2[0][0] + 40, :]

		# 将铁轨表面的候选ROI区域转成灰度图
		warped_gray_img1 = cv2.cvtColor(roi_img1, cv2.COLOR_BGR2GRAY)
		warped_gray_img2 = cv2.cvtColor(roi_img2, cv2.COLOR_BGR2GRAY)
		warped_gray_img3 = cv2.cvtColor(roi_img3, cv2.COLOR_BGR2GRAY)
		warped_gray_img4 = cv2.cvtColor(roi_img4, cv2.COLOR_BGR2GRAY)

		gauss_img1 = cv2.bilateralFilter(warped_gray_img1, d = 10, sigmaColor = 30,
			sigmaSpace = 15)
		gauss_img2 = cv2.bilateralFilter(warped_gray_img2, d = 10, sigmaColor = 30,
			sigmaSpace = 15)
		gauss_img3 = cv2.bilateralFilter(warped_gray_img3, d = 10, sigmaColor = 30,
			sigmaSpace = 15)
		gauss_img4 = cv2.bilateralFilter(warped_gray_img4, d = 10, sigmaColor = 30,
			sigmaSpace = 15)

		grad_img1_y = cv2.Sobel(gauss_img1, cv2.CV_8UC1, 0, 1, 3)
		grad_img2_y = cv2.Sobel(gauss_img2, cv2.CV_8UC1, 0, 1, 3)
		grad_img3_y = cv2.Sobel(gauss_img3, cv2.CV_8UC1, 0, 1, 3)
		grad_img4_y = cv2.Sobel(gauss_img4, cv2.CV_8UC1, 0, 1, 3)

		y_hist1 = []
		y_hist2 = []
		y_hist3 = []
		y_hist4 = []

		object_points = []

		if np.mean(grad_img1_y) < 10 and np.mean(grad_img1_y) > 2:
			grad_img1_y = np.where(grad_img1_y > np.max(grad_img1_y) / 2, grad_img1_y, 0)
			grad_img1_y = cv2.erode(grad_img1_y, None)
			for val in grad_img1_y:
				y_hist1.append(val.sum())
			i = 0
			while i < len(y_hist1) - 200:
				if y_hist1[i] != 0:
					j = i + 200
					if np.sum(y_hist1[i:j]) > 10 * 200:
						object_points.append((line1[0][0] - 40, i))
						i += 200
					else:
						pass
				i += 1

		if np.mean(grad_img2_y) < 10 and np.mean(grad_img2_y) > 2:
			grad_img2_y = np.where(grad_img2_y > np.max(grad_img2_y) / 2, grad_img2_y, 0)
			grad_img2_y = cv2.erode(grad_img2_y, None)
			for val in grad_img2_y:
				y_hist2.append(val.sum())
			i = 0
			while i < len(y_hist2) - 200:
				if y_hist2[i] != 0:
					j = i + 200
					if np.sum(y_hist2[i:j]) > 10 * 200:
						object_points.append((line1[0][0] - 40, i))
						i += 200
					else:
						pass
				i += 1
		if np.mean(grad_img3_y) < 10 and np.mean(grad_img3_y) > 2:
			grad_img3_y = np.where(grad_img3_y > np.max(grad_img3_y) / 2, grad_img3_y, 0)
			grad_img3_y = cv2.erode(grad_img3_y, None)
			for val in grad_img3_y:
				y_hist3.append(val.sum())
			i = 0
			while i < len(y_hist3) - 200:
				if y_hist3[i] != 0:
					j = i + 200
					if np.sum(y_hist3[i:j]) > 10 * 200:
						object_points.append((line2[0][0] - 40, i))
						i += 200
					else:
						pass
				i += 1 

		if np.mean(grad_img4_y) < 10 and np.mean(grad_img4_y) > 2:
			grad_img4_y = np.where(grad_img4_y > np.max(grad_img4_y) / 2, grad_img4_y, 0)
			grad_img4_y = cv2.erode(grad_img4_y, None)
			for val in grad_img4_y:
				y_hist4.append(val.sum())
			i = 0
			while i < len(y_hist4) - 200:
				if y_hist4[i] != 0:
					j = i + 200
					if np.sum(y_hist4[i:j]) > 10 * 200:
						object_points.append((line2[0][0] - 40, i))
						i += 200
					else:
						pass
				i += 1 


		# print(len(y_hist1), grad_img1_y.shape)
		# print(object_points)

		# if len(object_points) > 0:
		# 	point = np.dot(Minv, np.array([object_points[0][0], object_points[0][1], 1]))
		# 	print(point.shape)
		# 	cv2.rectangle(warped_img, (object_points[0][0], object_points[0][1] - 100), 
		# 		(object_points[0][0] + 40, object_points[0][1] + 100), (0, 0, 255), 2)

		# 	image_source = cv2.warpPerspective(warped_img, Minv, image_source.shape[1::-1], 
		# 								 flags=cv2.INTER_LINEAR)
		# 	# cv2.circle(image_source, (int(point[0] / point[-1]), int(point[1]/ point[-1])), 10, (0, 0, 255), 2)
		# 	point = cv2.convertPointsFromHomogeneous(np.array([point]))
		# 	cv2.circle(image_source, (int(point[0]), int(point[1])), 10, (0, 0, 255), 2)

		cv2.namedWindow('image_source', 2)
		cv2.imshow('image_source', image_source)

		# canny_x = cv2.Canny(gauss_img1, 50, 150)
		# cv2.imshow('gauss_img1', gauss_img1)
		# cv2.imshow('gauss_img2', gauss_img2)


		# cv2.namedWindow('grad_img1_y', 2)

		# cv2.imshow('grad_img1_y', grad_img1_y)
		# cv2.namedWindow('grad_img2_y', 2)
		# cv2.imshow('grad_img2_y', grad_img2_y)
		# cv2.namedWindow('grad_img3_y', 2)
		# cv2.imshow('grad_img3_y', grad_img3_y)
		# cv2.namedWindow('grad_img4_y', 2)
		# cv2.imshow('grad_img4_y', grad_img4_y)


		cv2.namedWindow('w', 2)
		cv2.imshow('w', warped_img)



		# cv2.imshow('roi_img', roi_img1)
		# cv2.namedWindow('affined', 2)
		# cv2.imshow('affined', binary_img)


		# cv2.namedWindow('source_img', 2)
		# cv2.imshow('source_img', warped_img)
		# cv2.namedWindow('roi_img1', 2)
		# cv2.imshow('roi_img1', roi_img1)
		# cv2.namedWindow('roi_img2', 2)
		# cv2.imshow('roi_img2', roi_img2)
		# cv2.namedWindow('roi_img3', 2)
		# cv2.imshow('roi_img3', roi_img3)
		# cv2.namedWindow('roi_img4', 2)
		# cv2.imshow('roi_img4', roi_img4)



		
	# cv2.namedWindow('tmp', 2)
	# cv2.imshow('tmp', tmp)
		cv2.waitKey(0)