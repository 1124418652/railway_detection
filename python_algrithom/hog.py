#-*- coding: utf-8 -*-
"""
@project: extract histogram of oriented gradient
@author: xhj
"""

import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt 
# from math import sqrt


class Hog_discriptor(object):
	"""
	HOG 描述符的实现
	"""

	def __init__(self, img, cell_size = 8, bin_size = 9, cells_per_block = 2, stride = 1):
		"""
		construct function:
			一个block由2x2个cell组成，步长为1个cell大小
		args:
			img: 输入图像（更准确的说是检测窗口），这里要求为灰度图像，对于行人检测图像
				 大小一般为 128x64，即是图像上的一小块裁切区域
			cell_size: 细胞单元的大小，如8，表示8x8个像素
			bin_size: 直方图的bin个数
			cell_per_block: 每个block的大小，用cell的个数度量，如2，表示一个block中有2x2个cell
			stride: 每个block在cell组成的矩阵中的滑动步长
		"""

		self.img = np.sqrt(img * 1.0 / float(np.max(img))) * 255     # 先将输入图像进行尺度变换
		self.cell_size = cell_size
		self.bin_size = bin_size
		self.angle_unit = 180 / self.bin_size     # 这里采用180°
		self.block_size = cells_per_block
		self.stride = stride
		assert type(self.bin_size) == int, "bin_size should be integer"
		assert type(self.cell_size) == int, "cell_size should be integer"
		assert(180 % self.bin_size == 0)

	def _global_gradient(self):
		"""
		计算输入图像的梯度幅度以及梯度的方向（角度）

		return:
			magnitude: magnitude = sqrt(dx^2 + dy^2)
			angle: angle = atan2(dy, dx)[*180/PI]
		"""

		gradient_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize = 3)
		gradient_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize = 3)
		gradient_magnitude, gradient_angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees = True)   # 角度需要用角度制表示（默认为弧度）
		return gradient_magnitude, gradient_angle / 2

	def _cell_gradient(self, cell_magnitude, cell_angle):
		"""
		为每个细胞单元构建梯度方向直方图，在决定梯度方向为cell_angle时应该增加哪一个bin时，采用了
		双线性插值的方法。
		例如：当我们把180°划分为9个bin的时候，分别对应0，20，40，...，160 这些角度，当角度是10，幅值是
			  4，因为角度10介于0~20°的中间（正好一半），所以把幅值一分为二地放到0和20两个bin中。

		args: 
			cell_magnitude: cell 中每个像素点的梯度幅值, dims 为 cell_size x cell_size
			cell_angle: cell 中每个像素点的梯度方向，dims 与 cell_magnitude 一致
		return:
			orientation_bin: 返回该cell对应的梯度直方图，长度为bin_size
		"""

		orientation_bin = np.zeros(self.bin_size)
		cell_rows, cell_cols = cell_magnitude.shape 
		for row in range(cell_rows):
			for col in range(cell_cols):
				# 根据angle判断当前的magnitude应该加到哪个bin中
				magnitude = cell_magnitude[row][col]
				angle = cell_angle[row][col]
				bin_before, bin_after = int(angle // self.angle_unit), int(angle // self.angle_unit) + 1 
				if bin_after > self.bin_size - 1:        # 当角度>160°时，需要分配到160和0两个bin中
					bin_after -= self.bin_size
				if bin_before > self.bin_size - 1:
					bin_before -= self.bin_size
				weight = angle % self.angle_unit / self.angle_unit     # bin_after 占有该magnitude的比例
				orientation_bin[bin_before] += (1 - weight) * magnitude
				orientation_bin[bin_after] += weight * magnitude
		return orientation_bin

	def render_gradient(self):
		"""
		将得到的每个cell的梯度方向直方图绘出，得到特征图
		
		args:
			image: 画布，和输入图像一样大 [h,w]
			cell_gradient: 输入图像的每个cell单元的梯度直方图，形状为 [h/cell_size, w/cell_size, bin_size]
		return:
			image: 特征图
		"""

		rows, cols = self.img.shape
		mag_array, angle_array = self._global_gradient()
		max_magnitude = np.max(mag_array)


	def extract(self, mode = "R-HOG", feature_image = True):
		"""
		从图像中提取出HOG特征，HOG特征以矩阵形式保存
		args:
			mode: 表示block的形状，包含三个取值{"R-ROG", "C—ROG", "SC-ROG"}
			feature_image: 表示是否要显示梯度特征图
		"""
		
		src_rows, src_cols = self.img.shape
		cell_rows = src_rows // self.cell_size
		cell_cols = src_cols // self.cell_size
		gradient_magnitude, gradient_angle = self._global_gradient()

		# 从cell的尺度计算图像的梯度方向直方图
		# cell_bin_array 的第一维和第二维是该cell在 cell 矩阵中的行列索引，第三位保存的是
		# cell(x,y) 的梯度方向直方图
		cell_bin_array = np.zeros((cell_rows, cell_cols, self.bin_size))  # 保存每个cell的方向梯度直方图
		for row in range(cell_rows):       # 先计算每个cell的梯度方向直方图
			for col in range(cell_cols):
				cell_magnitude = gradient_magnitude[row * self.cell_size : (row + 1) * self.cell_size,
							 					    col * self.cell_size : (col + 1) * self.cell_size]
				cell_angle = gradient_angle[row * self.cell_size : (row + 1) * self.cell_size, 
											col * self.cell_size : (col + 1) * self.cell_size]
				cell_bin_array[row, col, :] = self._cell_gradient(cell_magnitude, cell_angle)
		print(cell_bin_array.shape)
		print(self.img.shape)

		if feature_image:
			feat_img_array = np.zeros((src_rows, src_cols))
			max_cell_bin = cell_bin_array.max()
			half_cell_width = 0.5 * self.cell_size
			assert(max_cell_bin > 0)
			for row in range(cell_rows):
				for col in range(cell_cols):
					cell_bin = cell_bin_array[row][col] / max_cell_bin     # 归一化
					angle = 0        # 对于每个cell，分别统计angle从0到180度的magnitude
					angle_gap = self.angle_unit
					for magnitude in cell_bin:
						angle_radian = math.radians(angle)
						x1 = int(row * self.cell_size + half_cell_width 
								+ magnitude * half_cell_width * math.cos(angle_radian))
						y1 = int(col * self.cell_size + half_cell_width 
								+ magnitude * half_cell_width * math.sin(angle_radian))
						x2 = int(row * self.cell_size + half_cell_width 
								- magnitude * half_cell_width * math.cos(angle_radian))
						y2 = int(col * self.cell_size + half_cell_width 
								- magnitude * half_cell_width * math.sin(angle_radian))
						cv2.line(feat_img_array, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
						angle += angle_gap
			plt.imshow(feat_img_array)
			plt.show()


		if "R-HOG" == mode.upper():
			block_rows = (cell_rows - self.block_size) // self.stride + 1
			block_cols = (cell_cols - self.block_size) // self.stride + 1 

			# block_bin_array 和 cell_bin_array 类似，是一个三维的数组，第三维保存的是该位置的
			# block 的梯度方向直方图
			block_bin_array = np.zeros((block_rows, block_cols, 
										self.bin_size * self.block_size * self.block_size))
			for row in range(block_rows):
				for col in range(block_cols):
					# tmp_block_bin 为每个block中的梯度直方图矩阵
					tmp_block_bin = cell_bin_array[row * self.stride : row * self.stride + self.block_size,
												   col * self.stride : col * self.stride + self.block_size, :]
					tmp_block_bin = tmp_block_bin.flatten()
					tmp_block_bin = tmp_block_bin / math.sqrt(tmp_block_bin.dot(tmp_block_bin) + 10e-8)   # L2归一化
					block_bin_array[row, col, :] = tmp_block_bin
			return block_bin_array


if __name__ == '__main__':
	image_path = "../../v20190227/souce_images/warped_img.jpg"
	src_image = cv2.imread(image_path, 0)
	# src_image = cv2.resize(src_image, (64, 128))
	print(src_image.shape)
	hog = Hog_discriptor(src_image)
	print(hog.extract().shape)