#-*- coding: utf-8 -*-
import os
import cv2
import time
import numpy as np 

video_path = '../../sp1/轨道有异物1.mp4'
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
		cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = tuple(map(int, size))
writer = cv2.VideoWriter('../../视频1~5/轨道有异物1extract.avi', 
						 cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
						 fps, size)

num = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 7000)
flag, frame = cap.read()
while True:
	num += 1
	flag, frame = cap.read()
	print(num)
	if num > 5000:
		break
	if flag:
		writer.write(frame)
