// railwayDetection.cpp : �������̨Ӧ�ó������ڵ㡣
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

	///* ֱ�߼�ⲿ��api���ø�ʽ
	LineDetector detector;      // ����ֱ�߼����Ķ���
	vector<LinePoints> lines;   // ��¼��⵽��ֱ��
	detector.setThreshold(100);
	Vec4i roiRect(width / 6,    // ͼƬ������������ ROI ������
		          height * 2 / 3, 
				  width - width / 6 - 2, 
		          height - 1);
	detector.sfrDetection(img, lines, height * 3 / 5);     // ִ��ֱ�߼��

	cout << lines.size() << endl;

	for (int i = 0; i < lines.size(); ++i)          // ����ֱ��
	{
		line(img, Point(lines[i].x1, lines[i].y1),
			Point(lines[i].x2, lines[i].y2), Scalar(0, 0, 255), 2);
	}


	///* �ϰ����ⲿ�ֵ�api���ø�ʽ
	if (lines.size() == 2)
	{
		ObstacleDetector obs;
		vector<ObstacleInfo> obstacleList;      // ���ڼ�¼��⵽���ϰ�����Ϣ��������������Ϳ�ȼ��߶�(����Ǹ���ֵ50)
		obs.obstacleDetection(tmp, lines, obstacleList);   // ִ���ϰ�����
		cout << obstacleList.size();
		if (obstacleList.size() > 0)            // �����⵽���ϰ������ԭͼ�н��л���
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

	
	videoDetection(5, FROM_FILE, "img/���������1extract.avi");
	
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
