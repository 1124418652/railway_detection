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
 �����ϰ�������࣬ʹ�ø�������ϰ�����������ǰ��������
 1��ǰ���Ѿ���ȷ��⵽������ʾ�����ֱ�ߣ�ÿ��ֱ����һ�� LinePoints �ṹ��ʾ��
	�������ֱ����ʽΪ vector<LinePoints>����������ֱ�ߵ� y ��������Ƕ����
 2��Ŀǰֻ�����ֱ�߹�����м��
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
			std::cout << "������ȷ����Pythonģ��" << std::endl;
			//exit(-1);
		}
	}

	/* 
	 @brief ��ȡ͸�ӱ任�Լ���͸�ӱ任����ĺ���
	 @params srcPoints ԭͼƬ�е������ĸ�������
	 @params destPoints ͸�ӱ任�õ���ͼƬ����ԭͼƬ��Ӧ���ĸ�������
	 @params M ͸�ӱ任�ı任����
	 @params Minv ��͸�ӱ任�ı任����
	*/
	void getPerspectiveMatrix(const std::vector<cv::Point> &srcPoints, 
		const std::vector<cv::Point> &destPoints, cv::Mat &M, cv::Mat &Minv);

	/*
	 @brief ͨ���ú������Զ��ϰ���������Ҫ�õ��Ĳ��������ֶ�����
	*/
	void modifyParameters(int leftROIWidth = 40, int rightROIWidth = 40, 
		int d = 10, int sigmaColor = 30, int sigmaSpace = 15,
		int gradMeanThresh = 10, int regionHeight = 200);

	/*
	 @brief ִ���ϰ�����ĺ������ú�������ͨ��������ȷ��ʹ��ʲô��ʽ�����ϰ����⡣�ں�����
			ͨ��ͳ��ÿ������ROI��ƽ���ݶ����ж��Ƿ�Ϊ�������ϰ�������죨ƽ���ݶ�С��maxThresh,
			����minThresh���ж�Ϊ�������ϰ�������죬��Ҫ���м�⣩
	 @params img ������м��ͼƬ
	 @params lines ��ͼƬ���Ѿ���⵽��ֱ����Ϣ
	 @params obstacleList ���ڱ����ҵ����ϰ����λ����Ϣ��ÿ��Ԫ��ΪObstacleInfo���ͣ�
	 @params maxThresh �ж�Ϊ����������ϰ���������ֵ
	 @params minThresh �ж�Ϊ����������ϰ������С��ֵ
	 @params minGap �ϰ���֮�����С��������ͬһROI�������������ϰ����y����֮��Ĳ�ֵС��
					��ֵ�����ж�Ϊ���ϰ���
	 @returns ����һ��int�����ݣ���ʾ��⵽���ϰ������Ŀ
	*/
	int obstacleDetection(const cv::Mat &img, const std::vector<LinePoints> &lines, 
		std::vector<ObstacleInfo> &obstacleList, 
		int maxThresh = 10, int minThresh = 1.0, int minGap = 25, int binaryThresh = 100,
		int type = OBSTACLE_DETECTE_BY_GRAD);

private:
	cv::Mat img;                    // ������м���ԭͼƬ���ɺ����������ṩ
	std::vector<LinePoints> lines;  // ��ԭͼƬ���Ѿ��õ�������ֱ�ߣ��ɺ����������ṩ
	cv::Mat warpedImg;              // ����͸�ӱ任֮�������ͼƬ

	/* ���ڵ����ϰ������ROI�Ĳ��� */
	int leftROIWidth = 40;          // ��⵽������ֱ�����ѡȡ��ROI���
	int rightROIWidth = 40;         // ��⵽������ֱ���ұ�ѡȡ��ROI���

	/* ˫���˲�������ʹ�õĲ��� */
	int d = 10;
	int sigmaColor = 30;
	int sigmaSpace = 15;

	/* �ϰ��������жϵ���ֵ���� */
	int gradMeanThresh = 10;
	int regionHeight = 100;

	/* ��ʾPython��ģ��ͺ�������Ĳ��� */
	PyObject *module;    // pythonģ�����
	PyObject *pFunc;     // python��������
	PyObject *pDict;     // python�е�
};

#endif // !__OBSTACLE_DETECTION_H