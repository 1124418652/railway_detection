// railwayDetection.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include "lineDetection.h"
#include "obstacleDetection.h"
#include "videoDetection.h"
#include "loadTensorflowModel.h"

using namespace std;
using namespace cv;

int main()
{
	/*
	Mat img = imread("img/multiObstacle.jpg");
	Mat tmp = img.clone();
	if (!img.data)
		return -1;
	int height = img.rows;
	int width = img.cols;

	///* 直线检测部分api调用格式
	LineDetector detector;      // 创建直线检测类的对象
	vector<LinePoints> lines;   // 记录检测到的直线
	detector.setThreshold(100);
	Vec4i roiRect(width / 6,    // 图片中用于做检测的 ROI 区域定义
		          height * 2 / 3, 
				  width - width / 6 - 2, 
		          height - 1);
	detector.sfrDetection(img, lines, height * 3 / 5);     // 执行直线检测

	cout << lines.size() << endl;

	for (int i = 0; i < lines.size(); ++i)          // 绘制直线
	{
		line(img, Point(lines[i].x1, lines[i].y1),
			Point(lines[i].x2, lines[i].y2), Scalar(0, 0, 255), 2);
	}


	///* 障碍物检测部分的api调用格式
	if (lines.size() == 2)
	{
		ObstacleDetector obs;
		vector<ObstacleInfo> obstacleList;      // 用于记录检测到的障碍物信息，包括中心坐标和宽度及高度(这个是给定值50)
		obs.obstacleDetection(tmp, lines, obstacleList);   // 执行障碍物检测
		cout << obstacleList.size();
		if (obstacleList.size() > 0)            // 如果检测到了障碍物，则在原图中进行绘制
		{
			for (int i = 0; i < obstacleList.size(); ++i)
			{
				cv::rectangle(tmp, cv::Rect(obstacleList[i].center.x - obstacleList[i].width / 2,
					obstacleList[i].center.y - obstacleList[i].height / 2,
					obstacleList[i].width, obstacleList[i].height), cv::Scalar(0, 0, 255), 4);
			}
		}
	}
		
	
	cv::namedWindow("Image with lines", 2);
	cv::imshow("Image with lines", img);

	cv::namedWindow("Image with obstacles", 2);
	imshow("Image with obstacles", tmp);
	waitKey(0);

	system("pause");
	return 0;
	*/

	
	videoDetection(5, FROM_FILE, "img/轨道有异物1extract.avi");
	
	/*
	PyObject *module = NULL;
	PyObject *pDict = NULL;
	PyObject *pFunc = NULL;
	if (loadPyModule(&module, &pDict, &pFunc))
	{
		if (module == NULL)
			cout << "no" << endl;
		Mat img = imread("img/multiObstacle.jpg");
		cout << callPythonFunc(img, pFunc) << endl;
	}
	*/
	system("pause");

	return 0;
}
