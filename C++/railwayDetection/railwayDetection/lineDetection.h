#pragma once

#ifndef __LINE_DETECTION_H
#define __LINE_DETECTION_H

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const double PI = 3.1416;

typedef struct LinePoints {
	int x1;   // ֱ�߶εײ�����
	int y1;
	int x2;   // ֱ�߶ζ�������
	int y2;
	LinePoints(int x1, int y1, int x2, int y2) :x1(x1), y1(y1), x2(x2), y2(y2) {};
	LinePoints() :x1(0), y1(0), x2(0), y2(0){};
	~LinePoints() {};
}LinePoints;


/*
���������߼����࣬�ڸ����з�װ�ˣ�
1������houghLines����Լ�ͼƬԤ�������Ѳ���.
2����������ɸѡ�� houghLines ����ʵ��.
3������ houghLines �ĵ�֡�����⺯����ʵ��
4����֡ͼƬԤ��������ʵ��
5����ȡֱ����ͼƬ�ײ�����ĺ���ʵ��
*/
class LineDetector
{
public:
	LineDetector() = default;

	/**
	 @brief ͨ���������º������Զ� houghLines �Ĳ���������������,
	*/
	void setThreshold(int threshold) { this->threshold = threshold; }
	void setTheta(double theta) { this->theta = theta; }
	void setRho(double rho) { this->rho = rho; }
	void setMaxTheta(double maxTheta) { this->maxTheta = maxTheta; }
	
	/**
	 @brief ͨ���������º������Զ�ͼƬԤ��������У�GaussianBlar �Ĳ���
			������������
	*/
	void setKsize(const cv::Size &ksize) { this->ksize = ksize; }
	void setSigmaX(double sigmaX) { this->sigmaX = sigmaX; }
	void setSigmaY(double sigmaY) { this->sigmaY = sigmaY; }

	/**
	 @brief ���º������ڷ��ص�ǰʹ�õĸ�˹�˲��Ĳ���
	*/
	cv::Size getKsize() const { return this->ksize; }
	double getSigmaX() const { return this->sigmaX; }
	double getSigmaY() const { return this->sigmaY; }

	/*
	 @brief �������ԭͼ�����Ԥ������������� ROI ѡȡ���ҶȻ������Լ�
			������ĸ�˹�˲�
	 @param img ��Ҫ����Ԥ�����Դͼ��
	 @param roiRect Vec4i ���ͣ��ֱ𱣴� roi ���(left_top_x, left_top_y, width, height)
	*/
	void preProcession(const cv::Mat &img, const cv::Vec4i &roiRect);
	
	/*
	 @brief ʹ��Ĭ�ϵĲ�������houghֱ�߼�⣬�ú���ʹ�õ�ȫ�����Ѿ�����������
			����������û���ṩ���Ե������β�
	 @param rhoTheta ��Vec4f��ɵ�vector�����ڱ������֮��õ���
					��cos(theta), sin(theta), rho*cos(theta), rho*sin(theta)��
	*/
	void detectionWithHoughLines(std::vector<cv::Vec4f> &rhoTheta);

	/**
	 @brief ��֡ͼƬ�������߼�⺯��
	 @param img ���뵽�����н��м���ԭͼ��Size(2560, 1440), CV_8UC3
	 @param linePoints �� LinePoints ��ɵ�vector�����ڼ�¼��֡���Ľ��ֱ��
	 @param roiRect ��ԭͼ�н�ȡ�� ROI ����(left_top_x, left_top_y, width, height)�����û��
					��ʽ�����ֵ����ʹ��Ĭ��ֵ��ΪROI����
	*/
	void sfrDetection(const cv::Mat &img, std::vector<LinePoints> &lines,
		int y2 = -1, cv::Vec4i &roiRect = cv::Vec4i(0, 0, 0, 0));

	/**
	 @brief ��������
	*/
	virtual ~LineDetector() {};

private:
	cv::Mat gaussImg, grayImg;        // Ԥ����֮������ͼ��
	cv::Mat cannyImg;                 // canny ��Եͼ��
   
	/* houghLines ��Ҫ�õ��Ĳ��� */
	double rho = 1.0;
	double theta = 0.01;
	int threshold = 120;
	double maxTheta = PI / 3;         // ���ڶԼ�⵽��ֱ�߽���ɸѡ�����Ƕ�

	/* ͼƬԤ���������Ҫ�õ��Ĳ��� */
	cv::Size ksize = cv::Size(13, 13);
	double sigmaX = 3;
	double sigmaY = 7;

	/* Ԥ��������н�ȡ�� roi ��Ϣ */
	cv::Vec4i roiRect;

	/* HoughLines ��Ⲣ��ͨ��ɸѡ֮���ֱ����Ϣ
	   (cos(theta), sin(theta), rho*cos(theta), rho*sin(theta)) */
	std::vector<cv::Vec4f> rhoTheta;
};


#endif // !__LINE_DETECTION_H
