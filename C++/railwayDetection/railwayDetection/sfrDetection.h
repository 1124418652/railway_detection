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
 ���嵥֡ͼƬ��ֱ�߼����ϰ�����ĺ���
 */
void sfrDetect(const cv::Mat &frame, std::vector<LinePoints> &lines,
	std::vector<ObstacleInfo> &obstacleList, 
	std::deque<std::vector<LinePoints>> &lineDeque, int queueSize,
	LineDetector &detector, ObstacleDetector &obs, bool *initRoi);

/**
 �ú��������ڲ��Ե�֡��⺯����������Ҫɾ��
 */
void videoTest(std::string filepath);

#endif // __SFR_DETECTION_H