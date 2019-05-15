#pragma once

#ifndef __SFR_DETECTION_H
#define __SFR_DETECTION_H

#include <iostream>
#include <vector>
#include <string>
#include <deque>
#include <ctime>
#include "lineDetection.h"
#include "obstacleDetection.h"

/**
 定义单帧图片中直线检测和障碍物检测的函数
 */
void sfrDetect(const cv::Mat &frame, std::vector<LinePoints> &lines,
	std::vector<ObstacleInfo> &obstacleList, 
	std::deque<std::vector<LinePoints>> &lineDeque, int queueSize,
	LineDetector &detector, ObstacleDetector &obs, bool *initRoi);

/**
 该函数仅用于测试单帧检测函数，后续需要删除
 */
void videoTest(std::string filepath);

#endif // __SFR_DETECTION_H