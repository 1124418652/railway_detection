# -*- coding:utf-8 -*-
"""
Create on 25 March

Author: xhj
"""

import os
import cv2
import numpy as np
from collections import defaultdict


# __all__ = ['pre_procession', 'detection_with_houghlines', 'video_to_avi',
# 		   'single_frame_line_detection', 'line_detection_from_video',
# 		   'detection_with_houghlinesP']


def pre_procession(img, roi_rect = [], ksize = (13, 13), sigmaX = 3, 
				   sigmaY = 5, filter_type = 'gaussian'):
	"""
	单帧图片预处理：提取ROI->转灰度->高斯模糊

	Parameters:
	img: 输入图片
	roi_rect: list 类型，包含（left_top_x, left_top_y, width, height），如果
			  是None，默认截取下方1/2作为ROI
	ksize: 高斯模糊的核尺寸
	sigmaX, sigmaY: 高斯模糊的尺度
	filter_type: str类型参数 {'gaussian', 'bilateralFilter'}

	Return:
	gauss_img: 经过高斯模糊之后的roi图像
	roi_rect: roi的坐标即尺寸信息，list 类型，包含（left_top_x, left_top_y, width, height）,
			  为了后续将roi中识别的线条放回图片中
	"""

	height, width = img.shape[:2]
	if 0 == len(roi_rect):
		half_height = int(height / 2)
		roi = img[half_height:, :]
		roi_rect = [0, half_height, width, half_height]
	else:
		[lx, ly, rw, rh] = list(map(int, roi_rect))
		roi_rect = [lx, ly, rw, rh]
		assert(lx >= 0 and lx < width and lx + rw < width)
		assert(ly >= 0 and ly < height and ly + rh < height)
		roi = img[ly: ly + rh, lx: lx + rw]

	gray_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

	# gray_img = cv2.equalizeHist(gray_img)
	def half(k):
		return (int(k[0] / 2) + 1, k[1])

	if 'gaussian' == filter_type:
		gauss_img = np.zeros_like(gray_img)
		rw, rh = roi_rect[2:]
		gauss_img[:int(rh/4), :] = cv2.GaussianBlur(gray_img[:int(rh/4), :], 
												   ksize, sigmaX/2 , sigmaY/4)
		gauss_img[int(rh/4):int(rh/2),:] = cv2.GaussianBlur(gray_img[int(rh/4):int(rh/2),:],
															ksize, sigmaX*2/3 , sigmaY/3)
		gauss_img[int(rh/2):, :] = cv2.GaussianBlur(gray_img[int(rh/2):, :], 
												 ksize, sigmaX, sigmaY)
	elif 'bilateralFilter' == filter_type:
		gauss_img = cv2.bilateralFilter(gray_img, d = 10, sigmaColor = 50, 
			sigmaSpace = 15)

	return gauss_img, roi_rect, gray_img


def detection_with_houghlines(edge_img, rho = 1, theta = 0.01, 
							  threshold = 250, max_theta = np.pi / 3,
							  source_img = None, draw_lines = False):
	"""
	使用houghline对二值图像中的直线进行拼接

	Parameters:
	edge_img: ndarray 类型的二值图像
	rho: hough 空间中 rho 的递增值
	theta: hough 空间中 theta 的递增值
	threshold: HoughLines 算法的阈值，即hough空间中正弦曲线交点的累加值
	max_theta: 对直线进行筛选的条件，只保留theta小于max_theta的直线
	source_img: 如果需要在图片中绘制直线，该参数需要传入与edge_img相同大小的原图
	draw_lines: bool 型值，判断是否需要绘制直线

	Returns:
	line_points: list 类型，包含每条直线的两个点: (x1, y1, x2, y1)
	"""
	
	if edge_img.ndim != 2:
		raise ValueError("image input error")

	lines = cv2.HoughLines(edge_img, rho, theta, threshold)
	line_points = []
	rho_theta = []

	if not isinstance(lines, np.ndarray):
		return line_points, rho_theta

	for line in lines:
		r, t = line[0]
		if t < max_theta or t > np.pi - max_theta:
			a = np.cos(t)
			b = np.sin(t)
			x0 = r * a 
			y0 = r * b 

			x1 = int(x0 + 10000 * (-b))
			y1 = int(y0 + 10000 * a)
			x2 = int(x0 - 10000 * (-b))
			y2 = int(y0 - 10000 * a)

			line_points.append((x1, y1, x2, y2))
			rho_theta.append((a, b, x0, y0))

			if True == draw_lines:
				if not isinstance(source_img, np.ndarray):
					raise ValueError("the source image is empty")
				if source_img.shape[:2] != edge_img.shape[:2]:
					raise ValueError("""the shape of source image and 
										edge image should be same""")
				cv2.line(source_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

	return line_points, rho_theta


def get_bottom_x(rho_theta, roi_rect):

	x_tant = []
	for (cos_t, sin_t, x0, y0) in rho_theta:
		r = (x0 + roi_rect[0]) * cos_t + (y0 + roi_rect[1]) * sin_t
		x = (r - height * sin_t) / cos_t
		tan_t = sin_t / cos_t

		x_tant.append((x, tan_t))

	return x_tant


def detection_with_houghlinesP(edge_img, rho = 1, theta = 0.01, 
							   threshold = 120, minLineLength = 150,
							   maxLineGap = 20, min_tan = 0.8,
							   source_img = None, draw_lines = False):
	"""
	使用 cv2.HoughLinesP 对二值图像中的直线进行拼接

	Parameters:
	edge_img: ndarray 类型的二值图像
	rho: hough 空间中 rho 的递增值
	theta: hough 空间中 theta 的递增值
	threshold: HoughLines 算法的阈值，即hough空间中正弦曲线交点的累加值
	minLineLength: int型参数，判定为直线的最小长度
	maxLineGap: int型参数，直线中间可以出现的最大断裂
	min_tan: 直线与x轴夹角的最小正切值
	source_img: None or np.ndarray 如果需要在图片中绘制直线，该参数需要
				传入与edge_img相同大小的原图
	draw_lines: bool 型值，判断是否需要绘制直线

	Returns:
	line_points: list 类型，包含每条直线的两个点: (x1, y1, x2, y1)
	"""

	if edge_img.ndim != 2:
		raise ValueError("image input error")

	lines = cv2.HoughLinesP(edge_img, rho, theta, threshold, 
							minLineLength = minLineLength, 
							maxLineGap = maxLineGap)

	line_points = []
	if not isinstance(lines, np.ndarray):
		return line_points

	# 使用min_tan作为限制条件对直线进行筛选
	line_points = lines[(np.abs((lines[:, :, 3] - lines[:, :, 1]) \
		  				/(lines[:, :, 2] - lines[:, :, 0])) > min_tan).flatten()]
	line_points = line_points.reshape((-1, 4))

	if True == draw_lines:
		if not isinstance(source_img, np.ndarray):
			raise ValueError("the source image is empty")
		if source_img.shape[:2] != edge_img.shape[:2]:
			raise ValueError("""the shape of source image and 
								edge image should be same""")
		for line in line_points:
			x1, y1, x2, y2  = line[:]
			cv2.line(source_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

	return line_points


def video_to_avi(file_path):
	"""
	将输入路径中的视频文件转为 .avi 格式

	Parameters:
	file_path: 视频文件的完整文件路径
	"""

	if not os.path.exists(file_path):
		raise ValueError("The file is not exist")

	try:
		dir_path, file_name = os.path.split(file_path)
		file_name = os.path.splitext(file_name)[0]
		file_name += '.avi'
		new_file_path = os.path.join(dir_path, file_name)
	except:
		raise ValueError("Can't change the filename")

	video = cv2.VideoCapture(file_path)               # 打开原视频文件
	fps = int(video.get(cv2.CAP_PROP_FPS))     	      # 获取原视频的帧速率
	size = (video.get(cv2.CAP_PROP_FRAME_WIDTH),\
			video.get(cv2.CAP_PROP_FRAME_HEIGHT))     # 获取原视频文件的单帧尺寸
	size = tuple(map(int, size))

	# 用于写视频文件的对象
	writer = cv2.VideoWriter(new_file_path,\
		cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),\
		fps, size)

	flag, frame = video.read()
	while flag:       # loop until there are no more frames
		writer.write(frame)
		flag, frame = video.read()


def single_frame_line_detection(img_resource, draw_lines = False):
	"""
	对从视频中抽取出来的单帧图像进行铁轨检测

	Parameters:
	img_source: 从视频中抽取出来的单帧图像，该函数会修改img_source,
				在img_source中标注出检测到的直线
	Returns:
	line_points: list 类型，包含每条直线的两个点: (x1, y1, x2, y1)
	"""

	# img_tmp = img_source.copy()
	# gauss_img, roi_rect = pre_procession(img_tmp)
	# edge_image = cv2.Canny(gauss_img, 50, 150)
	# line_points = detection_with_houghlines(edge_image, threshold = 150)

	# if 0 == len(line_points):
	# 	draw_lines = False

	# if draw_lines:
	# 	for line in line_points:
	# 		x1 = line[0] + roi_rect[0]
	# 		y1 = line[1] + roi_rect[1]
	# 		x2 = line[2] + roi_rect[0]
	# 		y2 = line[3] + roi_rect[1]

	# 		cv2.line(img_tmp, (x1, y1), (x2, y2), (0, 0, 255), 2)

	# return line_points, img_tmp, edge_image

	tmp = cv2.cvtColor(img_resource, cv2.COLOR_BGR2HSV)
	tmp = cv2.merge((tmp[:,:,0], tmp[:,:,1], (tmp[:,:,2] * 0.6).astype(np.uint8)))
	tmp = cv2.cvtColor(tmp, cv2.COLOR_HSV2BGR)

	height, width = img_resource.shape[:2]
	gauss_img, roi_rect, gray_img = pre_procession(tmp, ksize = (13,13), 
		sigmaX = 3, sigmaY = 7, filter_type = 'gaussian', 
		roi_rect = [int(width/6), int(height*2/3), width-2*int(width/6)-1, 
					height-int(height*2/3)-1])
	edge_image = cv2.Canny(gauss_img, 50, 150)

	line_points, rho_theta = detection_with_houghlines(edge_image, threshold = 100)

	points = defaultdict(list)
	for key, (cos_t, sin_t, x0, y0) in enumerate(rho_theta):
		r = (x0 + roi_rect[0]) * cos_t + (y0 + roi_rect[1]) * sin_t   # 将ROI坐标系转换成原图像坐标系
		x1 = (r - height * sin_t) / cos_t                             # 求直线与图片底部的交点
		x2 = (r - height * 2 / 3 * sin_t) / cos_t

		if x1 <= width / 5:
			points[1].append((x1, x2, key))
		elif x1 <= width / 5 * 2:
			points[2].append((x1, x2, key))
		elif x1 <= width * 3 / 5:
			points[3].append((x1, x2, key))
		elif x1 <= width * 4 / 5:
			points[4].append((x1, x2, key))
		else:
			points[5].append((x1, x2, key))

		# img_resource = cv2.circle(img_resource, (int(x1), height), 15, (0, 0, 255), 8)
	new_line_points = []
	for i in range(1,6):
		if 0 == len(points[i]):
			pass
		else:
			a = np.array(points[i])
			x_new = sum(a[:, 0]) / len(points[i])
			
			h = np.argmin(np.abs(a[:, 0] - x_new)) # 每个区域中离均值最近的点在该区域中的索引
			new_line_points.append([int(points[i][h][0]), height, 
									int(points[i][h][1]), int(height * 2 / 3)])
			tmp = cv2.circle(tmp, (int(points[i][h][0]), height), 15, (0, 0, 255), 8)

			# 以下注释掉的步骤是在原先未处理的直线对中找出筛选出来的直线
			# x1, y1, x2, y2 = line_points[points[i][h][2]]  # points[i][h][2] 保存的是该点在所有直线中的索引值
			# x1 += roi_rect[0]
			# y1 += roi_rect[1]
			# x2 += roi_rect[0]
			# y2 += roi_rect[1]
			cv2.line(tmp, (int(points[i][h][0]), height), 
						  (int(points[i][h][1]), int(height * 2 / 3)), (0, 0, 255), 4)

	return new_line_points, tmp, edge_image



def line_detection_from_video(file_path, new_file_name = None):
	"""
	对视频文件进行铁轨检测
	"""

	if not os.path.exists(file_path):
		raise ValueError("The file is not exist!")

	dir_path, file_name = os.path.split(file_path)
	if not new_file_name:
		new_file_name = os.path.join(dir_path,\
			os.path.splitext(file_name)[0] + 'detected' + '.avi')
	else:
		new_file_name = os.path.join(dir_path, new_file_name)

	video = cv2.VideoCapture(file_path)
	fps = int(video.get(cv2.CAP_PROP_FPS))     	      # 获取原视频的帧速率
	size = (video.get(cv2.CAP_PROP_FRAME_WIDTH),\
			video.get(cv2.CAP_PROP_FRAME_HEIGHT))     # 获取原视频文件的单帧尺寸
	size = tuple(map(int, size))                      # 转 int 类型

	# 用于写视频文件的对象
	writer = cv2.VideoWriter(new_file_name,\
		cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),\
		fps, size)

	flag, frame = video.read()
	# cv2.namedWindow('img', 1)
	while flag:
		# cv2.imshow('img', frame)
		line_points, _img, _ = single_frame_line_detection(frame, True)
		writer.write(_img)
		flag, frame = video.read()


def obstacle_detection(file_path):

	if not os.path.exits(file_path):
		raise ValueError("The file is not exits")

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
			while i < len(y_hist1) - 100:
				if y_hist1[i] != 0:
					j = i + 100
					if np.sum(np.where(y_hist1[i:j] > 0, 1, 0)) > 10:
						object_points.append((line1[0][0] - 40, i))
						i += 100
					else:
						pass
				i += 1

		if np.mean(grad_img2_y) < 10 and np.mean(grad_img2_y) > 2:
			grad_img2_y = np.where(grad_img2_y > np.max(grad_img2_y) / 2, grad_img2_y, 0)
			grad_img2_y = cv2.erode(grad_img2_y, None)
			for val in grad_img2_y:
				y_hist2.append(val.sum())
			i = 0
			while i < len(y_hist2) - 100:
				if y_hist2[i] != 0:
					j = i + 100
					if np.sum(np.where(y_hist2[i:j]) > 0, 1, 0) > 10:
						object_points.append((line1[0][0] - 40, i))
						i += 100
					else:
						pass
				i += 1

		if np.mean(grad_img3_y) < 10 and np.mean(grad_img3_y) > 2:
			grad_img3_y = np.where(grad_img3_y > np.max(grad_img3_y) / 2, grad_img3_y, 0)
			grad_img3_y = cv2.erode(grad_img3_y, None)
			for val in grad_img3_y:
				y_hist3.append(val.sum())
			i = 0
			while i < len(y_hist3) - 100:
				if y_hist3[i] != 0:
					j = i + 100
					if np.sum(np.where(y_hist3[i:j]) > 0, 1, 0) > 10:
						object_points.append((line2[0][0] - 40, i))
						i += 100
					else:
						pass
				i += 1 

		if np.mean(grad_img4_y) < 10 and np.mean(grad_img4_y) > 2:
			grad_img4_y = np.where(grad_img4_y > np.max(grad_img4_y) / 2, grad_img4_y, 0)
			grad_img4_y = cv2.erode(grad_img4_y, None)
			for val in grad_img4_y:
				y_hist4.append(val.sum())
			i = 0
			while i < len(y_hist4) - 100:
				if y_hist4[i] != 0:
					j = i + 100
					if np.sum(np.where(y_hist4[i:j]) > 0, 1, 0) > 10:
						object_points.append((line2[0][0] - 40, i))
						i += 100
					else:
						pass
				i += 1 

		if len(object_points) > 0:
			for point in object_points:
				cv2.rectangle(warped_img, (point[0], point[1]), (point[0] + 40, 
					point[1]+100), (0,0,255), 2)
