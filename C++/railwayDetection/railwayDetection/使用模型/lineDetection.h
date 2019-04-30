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
	int x1;   // 直线段底部坐标
	int y1;
	int x2;   // 直线段顶部坐标
	int y2;
	LinePoints(int x1, int y1, int x2, int y2) :x1(x1), y1(y1), x2(x2), y2(y2) {};
	LinePoints() :x1(0), y1(0), x2(0), y2(0){};
	~LinePoints() {};
}LinePoints;


/*
定义铁轨线检测的类，在该类中封装了：
1、用于houghLines检测以及图片预处理的最佳参数.
2、包含条件筛选的 houghLines 函数实现.
3、基于 houghLines 的单帧铁轨检测函数的实现
4、单帧图片预处理函数的实现
5、获取直线与图片底部交点的函数实现
*/
class LineDetector
{
public:
	LineDetector() = default;

	/**
	 @brief 通过调用以下函数可以对 houghLines 的参数进行重新设置,
	*/
	void setThreshold(int threshold) { this->threshold = threshold; }
	void setTheta(double theta) { this->theta = theta; }
	void setRho(double rho) { this->rho = rho; }
	void setMaxTheta(double maxTheta) { this->maxTheta = maxTheta; }
	
	/**
	 @brief 通过调用以下函数可以对图片预处理过程中，GaussianBlar 的参数
			进行重新设置
	*/
	void setKsize(const cv::Size &ksize) { this->ksize = ksize; }
	void setSigmaX(double sigmaX) { this->sigmaX = sigmaX; }
	void setSigmaY(double sigmaY) { this->sigmaY = sigmaY; }

	/**
	 @brief 以下函数用于返回当前使用的高斯滤波的参数
	*/
	cv::Size getKsize() const { return this->ksize; }
	double getSigmaX() const { return this->sigmaX; }
	double getSigmaY() const { return this->sigmaY; }

	/*
	 @brief 对输入的原图像进行预处理操作，包括 ROI 选取，灰度化处理以及
			分区域的高斯滤波
	 @param img 需要进行预处理的源图像
	 @param roiRect Vec4i 类型，分别保存 roi 框的(left_top_x, left_top_y, width, height)
	*/
	void preProcession(const cv::Mat &img, const cv::Vec4i &roiRect);
	
	/*
	 @brief 使用默认的参数进行hough直线检测，该函数使用的全都是已经经过调整的
			参数，所以没有提供可以调整的形参
	 @param rhoTheta 由Vec4f组成的vector，用于保存计算之后得到的
					（cos(theta), sin(theta), rho*cos(theta), rho*sin(theta)）
	*/
	void detectionWithHoughLines(std::vector<cv::Vec4f> &rhoTheta);

	/**
	 @brief 单帧图片的铁轨线检测函数
	 @param img 输入到函数中进行检测的原图，Size(2560, 1440), CV_8UC3
	 @param linePoints 由 LinePoints 组成的vector，用于记录单帧检测的结果直线
	 @param roiRect 在原图中截取的 ROI 区域，(left_top_x, left_top_y, width, height)，如果没有
					显式传入该值，则使用默认值作为ROI区域。
	*/
	void sfrDetection(const cv::Mat &img, std::vector<LinePoints> &lines,
		int y2 = -1, cv::Vec4i &roiRect = cv::Vec4i(0, 0, 0, 0));

	/**
	 @brief 析构函数
	*/
	virtual ~LineDetector() {};

private:
	cv::Mat gaussImg, grayImg;        // 预处理之后的输出图像
	cv::Mat cannyImg;                 // canny 边缘图像
   
	/* houghLines 需要用到的参数 */
	double rho = 1.0;
	double theta = 0.01;
	int threshold = 120;
	double maxTheta = PI / 3;         // 用于对检测到的直线进行筛选的最大角度

	/* 图片预处理过程需要用到的参数 */
	cv::Size ksize = cv::Size(13, 13);
	double sigmaX = 3;
	double sigmaY = 7;

	/* 预处理过程中截取的 roi 信息 */
	cv::Vec4i roiRect;

	/* HoughLines 检测并且通过筛选之后的直线信息
	   (cos(theta), sin(theta), rho*cos(theta), rho*sin(theta)) */
	std::vector<cv::Vec4f> rhoTheta;
};


#endif // !__LINE_DETECTION_H
