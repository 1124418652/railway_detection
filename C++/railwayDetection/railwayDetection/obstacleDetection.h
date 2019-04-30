#pragma once

#ifndef __OBSTACLE_DETECTION_H
#define __OBSTACLE_DETECTION_H

#include <iostream>
#include <vector>
#include <deque>
#include <stack>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "lineDetection.h"
#include "loadTensorflowModel.h"


#define OBSTACLE_DETECTE_BY_GRAD 0
#define OBSTACLE_DETECTE_BY_BLOB 1

typedef struct ObstacleInfo {
	cv::Point center;
	int width;
	int height;
	ObstacleInfo(cv::Point center, int width, int height) :center(center),
		width(width), height(height) {};
	~ObstacleInfo() {};
}ObstacleInfo;

/*
 定义障碍物检测的类，使用该类进行障碍物检测有两个前提条件：
 1、前期已经正确检测到两条表示铁轨的直线，每条直线用一个 LinePoints 结构表示，
	输入检测的直线形式为 vector<LinePoints>，并且两条直线的 y 坐标必须是对齐的
 2、目前只能针对直线轨道进行检测
*/
class ObstacleDetector
{
public:
	ObstacleDetector()
	{
		module = NULL;
		pFunc = NULL;

		if (!loadPyModule(&module, &pDict, &pFunc)) 
		{
			std::cout << "不能正确导入Python模块" << std::endl;
			//exit(-1);
		}
	}

	/* 
	 @brief 获取透视变换以及逆透视变换矩阵的函数
	 @params srcPoints 原图片中的至少四个点坐标
	 @params destPoints 透视变换得到的图片中与原图片对应的四个点坐标
	 @params M 透视变换的变换矩阵
	 @params Minv 逆透视变换的变换矩阵
	*/
	void getPerspectiveMatrix(const std::vector<cv::Point> &srcPoints, 
		const std::vector<cv::Point> &destPoints, cv::Mat &M, cv::Mat &Minv);

	/*
	 @brief 通过该函数可以对障碍物检测中需要用到的参数进行手动调整
	*/
	void modifyParameters(int leftROIWidth = 40, int rightROIWidth = 40, 
		int d = 10, int sigmaColor = 30, int sigmaSpace = 15,
		int gradMeanThresh = 10, int regionHeight = 200);

	/*
	 @brief 执行障碍物检测的函数，该函数可以通过参数来确定使用什么方式进行障碍物检测。在函数中
			通过统计每条铁轨ROI的平均梯度来判断是否为可能有障碍物的铁轨（平均梯度小于maxThresh,
			大于minThresh则判断为可能有障碍物的铁轨，需要进行检测）
	 @params img 输入进行检测图片
	 @params lines 在图片中已经检测到的直线信息
	 @params obstacleList 用于保存找到的障碍物的位置信息（每个元素为ObstacleInfo类型）
	 @params maxThresh 判断为轨道面中有障碍物的最大阈值
	 @params minThresh 判断为轨道面中有障碍物的最小阈值
	 @params minGap 障碍物之间的最小间隔，如果同一ROI中两个检测出的障碍物的y坐标之间的差值小于
					该值，则判定为非障碍物
	 @returns 返回一个int型数据，表示检测到的障碍物的数目
	*/
	int obstacleDetection(const cv::Mat &img, const std::vector<LinePoints> &lines, 
		std::vector<ObstacleInfo> &obstacleList, 
		int maxThresh = 10, int minThresh = 1.0, int minGap = 25, int binaryThresh = 100,
		int type = OBSTACLE_DETECTE_BY_GRAD);

private:
	cv::Mat img;                    // 输入进行检测的原图片，由函数调用者提供
	std::vector<LinePoints> lines;  // 在原图片中已经得到的铁轨直线，由函数调用者提供
	cv::Mat warpedImg;              // 经过透视变换之后的铁轨图片

	/* 用于调整障碍物检测的ROI的参数 */
	int leftROIWidth = 40;          // 检测到的铁轨直线左边选取的ROI宽度
	int rightROIWidth = 40;         // 检测到的铁轨直线右边选取的ROI宽度

	/* 双边滤波算子中使用的参数 */
	int d = 10;
	int sigmaColor = 30;
	int sigmaSpace = 15;

	/* 障碍物区域判断的阈值参数 */
	int gradMeanThresh = 10;
	int regionHeight = 100;

	/* 表示Python中模块和函数对象的参数 */
	PyObject *module;    // python模块对象
	PyObject *pFunc;     // python函数对象
	PyObject *pDict;     // python中的
};

#endif // !__OBSTACLE_DETECTION_H