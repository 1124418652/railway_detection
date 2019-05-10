#include "stdafx.h"
#include "obstacleDetection.h"


void ObstacleDetector::getPerspectiveMatrix(const std::vector<cv::Point> &srcPoints,
	const std::vector<cv::Point> &destPoints, cv::Mat &M, cv::Mat &Minv)
{
	assert(4 == srcPoints.size());
	assert(4 == destPoints.size());
	cv::Point2f src[4];
	cv::Point2f dest[4];

	for (int i = 0; i < 4; ++i)
	{
		src[i] = cv::Point2f(srcPoints[i]);
		dest[i] = cv::Point2f(destPoints[i]);
	}

	M = cv::getPerspectiveTransform(src, dest);
	Minv = cv::getPerspectiveTransform(dest, src);
}

void ObstacleDetector::modifyParameters(int leftROIWidth, int rightROIWidth,
	int d, int sigmaColor, int sigmaSpace,
	int gradMeanThresh, int regionHeight)
{
	this->leftROIWidth = leftROIWidth;
	this->rightROIWidth = rightROIWidth;
	this->d = d;
	this->sigmaColor = sigmaColor;
	this->sigmaSpace = sigmaSpace;
	this->gradMeanThresh = gradMeanThresh;
	this->regionHeight = regionHeight;
}

int ObstacleDetector::obstacleDetection(const cv::Mat &img, 
	const std::vector<LinePoints> &lines, std::vector<ObstacleInfo> &obstacleList, 
	int maxThresh, int minThresh, int binaryThresh, int minGap, int type)
{
	/*
	以下内容是为了提取数据新加的，后续需要删除
	*/

	std::string filePath = "img/data_set/";
	int num = 1200;

	/*
	到这为止
	*/

	assert(2 == lines.size());
	if (!obstacleList.empty())             // 保证用于保存结果的obstacleList为空
		obstacleList.clear();
	int numObstacles = 0;                  // 记录检测到的障碍物的数目
	int warpedLine1x1 = img.cols / 4;
	int warpedLine2x1 = img.cols * 3 / 4;
	LinePoints line1_res(0, 0, 0, 0);      // 用于记录检测到的两条原始直线
	LinePoints line2_res(0, 0, 0, 0);

	if (lines[0].x1 <= lines[1].x1)
	{
		line1_res = lines[0];       // 检测到的第一条直线
		line2_res = lines[1];       // 检测到的第二条直线
	}
	else
	{
		line1_res = lines[1];
		line2_res = lines[0];
	}

	/* 透视变换 */
	cv::Mat M, Minv;     // 透视变换的转换矩阵
	cv::Mat warpedImg;
	std::vector<cv::Point> srcPoints = { cv::Point(line1_res.x1, line1_res.y1),
										 cv::Point(line1_res.x2, line1_res.y2),
										 cv::Point(line2_res.x1, line2_res.y1),
										 cv::Point(line2_res.x2, line2_res.y2) };
	std::vector<cv::Point> destPoints = { cv::Point(warpedLine1x1, img.rows),      // line1_res.y1
										  cv::Point(warpedLine1x1, 0),
										  cv::Point(warpedLine2x1, img.rows),
										  cv::Point(warpedLine2x1, 0) };
	getPerspectiveMatrix(srcPoints, destPoints, M, Minv);       // 获取透视变换的矩阵
	cv::warpPerspective(img, warpedImg, M, cv::Size(img.cols, img.rows));

	/* 从透视变换之后的图片中截取检测直线两侧指定长度的ROI作为待检测区域 */
	cv::Mat roiImg1 = warpedImg(cv::Range::all(), 
								cv::Range(warpedLine1x1 - leftROIWidth, warpedLine1x1 - 2));
	cv::Mat roiImg2 = warpedImg(cv::Range::all(),
								cv::Range(warpedLine1x1 + 2, warpedLine1x1 + leftROIWidth));
	cv::Mat roiImg3 = warpedImg(cv::Range::all(),
								cv::Range(warpedLine2x1 - rightROIWidth, warpedLine2x1 - 2));
	cv::Mat roiImg4 = warpedImg(cv::Range::all(),
								cv::Range(warpedLine2x1 + 2, warpedLine2x1 + rightROIWidth));
	cv::cvtColor(roiImg1, roiImg1, cv::COLOR_RGB2GRAY);
	cv::cvtColor(roiImg2, roiImg2, cv::COLOR_RGB2GRAY);
	cv::cvtColor(roiImg3, roiImg3, cv::COLOR_RGB2GRAY);
	cv::cvtColor(roiImg4, roiImg4, cv::COLOR_RGB2GRAY);
	cv::Mat gaussImg1, gaussImg2, gaussImg3, gaussImg4;         // 需要先对待测区域进行双边滤波
	cv::bilateralFilter(roiImg1, gaussImg1, d, sigmaColor, sigmaSpace);
	cv::bilateralFilter(roiImg2, gaussImg2, d, sigmaColor, sigmaSpace);
	cv::bilateralFilter(roiImg3, gaussImg3, d, sigmaColor, sigmaSpace);
	cv::bilateralFilter(roiImg4, gaussImg4, d, sigmaColor, sigmaSpace);

	/* 分别计算四个ROI y方向的梯度值 */
	cv::Mat gradyImg1, gradyImg2, gradyImg3, gradyImg4;         
	cv::Sobel(gaussImg1, gradyImg1, CV_8UC1, 0, 1, 3);
	cv::Sobel(gaussImg2, gradyImg2, CV_8UC1, 0, 1, 3);
	cv::Sobel(gaussImg3, gradyImg3, CV_8UC1, 0, 1, 3);
	cv::Sobel(gaussImg4, gradyImg4, CV_8UC1, 0, 1, 3);

	cv::Scalar meanGradyImg1 = cv::mean(gradyImg1);
	cv::Scalar meanGradyImg2 = cv::mean(gradyImg2);
	cv::Scalar meanGradyImg3 = cv::mean(gradyImg3);
	cv::Scalar meanGradyImg4 = cv::mean(gradyImg4);

	/* 用于接收y方向统计直方图的数组 */
	int *yHist1 = new int[gradyImg1.rows];
	int *yHist2 = new int[gradyImg2.rows];
	int *yHist3 = new int[gradyImg3.rows];
	int *yHist4 = new int[gradyImg4.rows];
	memset(yHist1, 0, gradyImg1.rows * sizeof(int));
	memset(yHist2, 0, gradyImg2.rows * sizeof(int));
	memset(yHist3, 0, gradyImg3.rows * sizeof(int));
	memset(yHist4, 0, gradyImg4.rows * sizeof(int));

	/* 对 ROI 区域1进行障碍物检测 */
	if (meanGradyImg1[0] < maxThresh && meanGradyImg1[0] > minThresh)
	{
		cv::Mat kernal = cv::Mat::ones(1, leftROIWidth / 5, CV_8UC1);      // 定义腐蚀变换的核
		cv::threshold(gradyImg1, gradyImg1, binaryThresh, 255, cv::THRESH_BINARY);
		cv::erode(gradyImg1, gradyImg1, kernal);                           // 对梯度的阈值图进行腐蚀运算
		for (int row = 0; row < gradyImg1.rows; ++row)
		{
			uchar *p = gradyImg1.ptr<uchar>(row);
			int rowSum = 0;
			for (int col = 0; col < gradyImg1.cols; ++col)       // 计算每一行的梯度和
				rowSum += (int)p[col];
			yHist1[row] += rowSum;                               // 记录下y方向的直方图信息
		}

		/* 进行障碍物搜索和定位的算法 */
		std::deque<ObstacleInfo> tmpObsDeque;
		for (int i = 50; i < gradyImg1.rows - 50; ++i)
		{
			if (yHist1[i] != 0)
			{
				/* 进行障碍物定位的程序，即在一维数组中搜索波峰的数目 */
				int sumGrad = 0;
				for (int j = i; j < i + 50; ++j)
					sumGrad += 1;
				if (sumGrad > leftROIWidth / 2)
				{
					numObstacles += 1;       // 障碍物的数目加1
					int xWraped = warpedLine1x1 - leftROIWidth / 2;
					int yWraped = i;

					/*
					以下内容是为了提取数据新加的，后续需要删除
					*/
	
					cv::imwrite(filePath + std::to_string(num) + ".jpg", 
						warpedImg(cv::Range(i - 50, i + 50), 
								  cv::Range(xWraped - 50, xWraped + 50)));
					num += 1;
					std::cout << num << std::endl;
					/*
					到这为止
					*/

					std::vector<cv::Point2f> points = { cv::Point2f(xWraped, yWraped) }, pointTrans;
					cv::perspectiveTransform(points, pointTrans, Minv);
					tmpObsDeque.push_back(ObstacleInfo(cv::Point(pointTrans[0].x,             // 候选障碍物框入队
														pointTrans[0].y), 50, 50));
					i += 50;
				}
			}
		}
	
		int head = 0, tail = 0;
		if (tmpObsDeque.size() > 8)
		{
			for (size_t i = 0; i < tmpObsDeque.size() - 1; ++i)
			{
				if (tmpObsDeque[i + 1].center.y - tmpObsDeque[i].center.y > minGap)
				{
					if (head == tail) {
						head++;
						tail++;
					}
					else {
						while (head < tail)
							head++;
					}
					obstacleList.push_back(tmpObsDeque[head]);
				}
				else
					tail++;
			}
		}
		else
		{
			for (size_t i = 0; i < tmpObsDeque.size(); ++i)
				obstacleList.push_back(tmpObsDeque[i]);
		}
	}

	if (meanGradyImg2[0] < maxThresh && meanGradyImg2[0] > minThresh)
	{
		cv::Mat kernal = cv::Mat::ones(1, leftROIWidth / 5, CV_8UC1);
		cv::threshold(gradyImg2, gradyImg2, binaryThresh, 255, cv::THRESH_BINARY);
		cv::erode(gradyImg2, gradyImg2, kernal);
		for (int row = 0; row < gradyImg2.rows; ++row)
		{
			uchar *p = gradyImg2.ptr<uchar>(row);
			int rowSum = 0;
			for (int col = 0; col < gradyImg2.cols; ++col)
				rowSum += (int)p[col];
			yHist2[row] += rowSum;
		}

		/* 进行障碍物搜索和定位的算法 */
		std::deque<ObstacleInfo> tmpObsDeque;
		for (int i = 50; i < gradyImg2.rows - 50; ++i)
		{
			if (yHist2[i] != 0)
			{
				/* 进行障碍物定位的程序，即在一维数组中搜索波峰的数目 */
				int sumGrad = 0;
				for (int j = i; j < i + 50; ++j)
					sumGrad += 1;
				if (sumGrad > leftROIWidth / 2)
				{
					numObstacles += 1;       // 障碍物的数目加1
					int xWraped = warpedLine1x1 + leftROIWidth / 2;
					int yWraped = i;

					/*
					以下内容是为了提取数据新加的，后续需要删除
					*/

					cv::imwrite(filePath + std::to_string(num) + ".jpg",
						warpedImg(cv::Range(i - 50, i + 50),
							cv::Range(xWraped - 50, xWraped + 50)));
					num += 1;
					std::cout << num << std::endl;		

					/*
					到这为止
					*/

					std::vector<cv::Point2f> points = { cv::Point2f(xWraped, yWraped) }, pointTrans;
					cv::perspectiveTransform(points, pointTrans, Minv);
					tmpObsDeque.push_back(ObstacleInfo(cv::Point(pointTrans[0].x,
						pointTrans[0].y), 50, 50));
					i += 50;
				}
			}
		}
		int head = 0, tail = 0;
		if (tmpObsDeque.size() > 8)
		{
			for (size_t i = 0; i < tmpObsDeque.size() - 1; ++i)
			{
				if (tmpObsDeque[i + 1].center.y - tmpObsDeque[i].center.y > minGap)
				{
					if (head == tail) {
						head++;
						tail++;
					}
					else {
						while (head < tail)
							head++;
					}
					obstacleList.push_back(tmpObsDeque[head]);
				}
				else
					tail++;
			}
		}
		else
		{
			for (size_t i = 0; i < tmpObsDeque.size(); ++i)
				obstacleList.push_back(tmpObsDeque[i]);
		}
	}

	if (meanGradyImg3[0] < maxThresh && meanGradyImg3[0] > minThresh)
	{
		cv::Mat kernal = cv::Mat::ones(1, rightROIWidth / 5, CV_8UC1);
		cv::threshold(gradyImg3, gradyImg3, binaryThresh, 255, cv::THRESH_BINARY);
		cv::erode(gradyImg3, gradyImg3, kernal);
		for (int row = 0; row < gradyImg3.rows; ++row)
		{
			uchar *p = gradyImg3.ptr<uchar>(row);
			int rowSum = 0;
			for (int col = 0; col < gradyImg3.cols; ++col)
				rowSum += (int)p[col];
			yHist3[row] += rowSum;
		}

		/* 进行障碍物搜索和定位的算法 */
		std::deque<ObstacleInfo> tmpObsDeque;
		for (int i = 50; i < gradyImg3.rows - 50; ++i)
		{
			if (yHist3[i] != 0)
			{
				/* 进行障碍物定位的程序，即在一维数组中搜索波峰的数目 */
				int sumGrad = 0;
				for (int j = i; j < i + 50; ++j)
					sumGrad += 1;
				if (sumGrad > rightROIWidth / 2)
				{
					int xWraped = warpedLine2x1 - rightROIWidth / 2;
					int yWraped = i;

					/*
					以下内容是为了提取数据新加的，后续需要删除
	*/

					cv::imwrite(filePath + std::to_string(num) + ".jpg",
						warpedImg(cv::Range(i - 50, i + 50),
							cv::Range(xWraped - 50, xWraped + 50)));
					num += 1;
					std::cout << num << std::endl;

					/*
					到这为止
					*/

					std::vector<cv::Point2f> points = { cv::Point2f(xWraped, yWraped) }, pointTrans;
					cv::perspectiveTransform(points, pointTrans, Minv);
					tmpObsDeque.push_back(ObstacleInfo(cv::Point(pointTrans[0].x,
						pointTrans[0].y), 50, 50));
					i += 50;
				}
			}
		}
		int head = 0, tail = 0;
		if (tmpObsDeque.size() > 5)
		{
			for (size_t i = 0; i < tmpObsDeque.size() - 1; ++i)
			{
				if (tmpObsDeque[i + 1].center.y - tmpObsDeque[i].center.y > minGap)
				{
					if (head == tail) {
						head++;
						tail++;
					}
					else {
						while (head < tail)
							head++;
					}
					obstacleList.push_back(tmpObsDeque[head]);
				}
				else
					tail++;
			}
		}
		else
		{
			for (size_t i = 0; i < tmpObsDeque.size(); ++i)
				obstacleList.push_back(tmpObsDeque[i]);
		}
	}

	if (meanGradyImg4[0] < maxThresh && meanGradyImg4[0] > minThresh)
	{
		cv::Mat kernal = cv::Mat::ones(1, rightROIWidth / 5, CV_8UC1);
		cv::threshold(gradyImg4, gradyImg4, binaryThresh, 255, cv::THRESH_BINARY);
		cv::erode(gradyImg4, gradyImg4, kernal);
		for (int row = 0; row < gradyImg4.rows; ++row)
		{
			uchar *p = gradyImg4.ptr<uchar>(row);
			int rowSum = 0;
			for (int col = 0; col < gradyImg4.cols; ++col)
				rowSum += (int)p[col];
			yHist4[row] += rowSum;
		}
		
		/* 进行障碍物识别的代码 */
		std::deque<ObstacleInfo> tmpObsDeque;
		for (int i = 50; i < gradyImg4.rows - 50; ++i)
		{
			if (yHist4[i] != 0)
			{
				/* 进行障碍物定位的程序，即在一维数组中搜索波峰的数目 */
				int sumGrad = 0;
				for (int j = i; j < i + 50; ++j)
					sumGrad += 1;
				if (sumGrad > rightROIWidth / 2)
				{
					int xWraped = warpedLine2x1 + rightROIWidth / 2;
					int yWraped = i;

					/*
					以下内容是为了提取数据新加的，后续需要删除
		*/

					cv::imwrite(filePath + std::to_string(num) + ".jpg",
						warpedImg(cv::Range(i - 50, i + 50),
							cv::Range(xWraped - 50, xWraped + 50)));
					num += 1;
					std::cout << num << std::endl;

					/*
					到这为止
					*/
					std::vector<cv::Point2f> points = { cv::Point2f(xWraped, yWraped) }, pointTrans;
					cv::perspectiveTransform(points, pointTrans, Minv);
					tmpObsDeque.push_back(ObstacleInfo(cv::Point(pointTrans[0].x,
						pointTrans[0].y), 50, 50));
					i += 50;		
				}
			}
		}
		int head = 0, tail = 0;
		if (tmpObsDeque.size() > 5)
		{
			for (size_t i = 0; i < tmpObsDeque.size() - 1; ++i)
			{
				if (tmpObsDeque[i + 1].center.y - tmpObsDeque[i].center.y > minGap)
				{
					if (head == tail) {
						head++;
						tail++;
					}
					else {
						while (head < tail)
							head++;
					}
					obstacleList.push_back(tmpObsDeque[head]);
				}
				else
					tail++;
			}
		}
		else
		{
			for (size_t i = 0; i < tmpObsDeque.size(); ++i)
				obstacleList.push_back(tmpObsDeque[i]);
		}
	}

	delete[]yHist1;
	delete[]yHist2;
	delete[]yHist3;
	delete[]yHist4;
	return numObstacles;
}

int ObstacleDetector::obstacleDetectionWithModel(const cv::Mat &img,
	const std::vector<LinePoints> &lines, std::vector<ObstacleInfo> &obstacleList,
	int maxThresh, int minThresh, int binaryThresh, int minGap)
{
	assert(2 == lines.size());
	if (!obstacleList.empty())             // 保证用于保存结果的obstacleList为空
		obstacleList.clear();
	int numObstacles = 0;                  // 记录检测到的障碍物的数目
	int warpedLine1x1 = img.cols / 4;
	int warpedLine2x1 = img.cols * 3 / 4;
	LinePoints line1_res(0, 0, 0, 0);      // 用于记录检测到的两条原始直线
	LinePoints line2_res(0, 0, 0, 0);

	if (lines[0].x1 <= lines[1].x1)
	{
		line1_res = lines[0];       // 检测到的第一条直线
		line2_res = lines[1];       // 检测到的第二条直线
	}
	else
	{
		line1_res = lines[1];
		line2_res = lines[0];
	}

	/* 透视变换 */
	cv::Mat M, Minv;     // 透视变换的转换矩阵
	cv::Mat warpedImg;
	std::vector<cv::Point> srcPoints = { cv::Point(line1_res.x1, line1_res.y1),
		cv::Point(line1_res.x2, line1_res.y2),
		cv::Point(line2_res.x1, line2_res.y1),
		cv::Point(line2_res.x2, line2_res.y2) };
	std::vector<cv::Point> destPoints = { cv::Point(warpedLine1x1, img.rows),      // line1_res.y1
		cv::Point(warpedLine1x1, 0),
		cv::Point(warpedLine2x1, img.rows),
		cv::Point(warpedLine2x1, 0) };
	getPerspectiveMatrix(srcPoints, destPoints, M, Minv);       // 获取透视变换的矩阵
	cv::warpPerspective(img, warpedImg, M, cv::Size(img.cols, img.rows));

	/* 从透视变换之后的图片中截取检测直线两侧指定长度的ROI作为待检测区域 */
	cv::Mat roiImg1 = warpedImg(cv::Range::all(),
		cv::Range(warpedLine1x1 - leftROIWidth, warpedLine1x1 - 2));
	cv::Mat roiImg2 = warpedImg(cv::Range::all(),
		cv::Range(warpedLine1x1 + 2, warpedLine1x1 + leftROIWidth));
	cv::Mat roiImg3 = warpedImg(cv::Range::all(),
		cv::Range(warpedLine2x1 - rightROIWidth, warpedLine2x1 - 2));
	cv::Mat roiImg4 = warpedImg(cv::Range::all(),
		cv::Range(warpedLine2x1 + 2, warpedLine2x1 + rightROIWidth));
	cv::cvtColor(roiImg1, roiImg1, cv::COLOR_RGB2GRAY);
	cv::cvtColor(roiImg2, roiImg2, cv::COLOR_RGB2GRAY);
	cv::cvtColor(roiImg3, roiImg3, cv::COLOR_RGB2GRAY);
	cv::cvtColor(roiImg4, roiImg4, cv::COLOR_RGB2GRAY);
	cv::Mat gaussImg1, gaussImg2, gaussImg3, gaussImg4;         // 需要先对待测区域进行双边滤波
	cv::bilateralFilter(roiImg1, gaussImg1, d, sigmaColor, sigmaSpace);
	cv::bilateralFilter(roiImg2, gaussImg2, d, sigmaColor, sigmaSpace);
	cv::bilateralFilter(roiImg3, gaussImg3, d, sigmaColor, sigmaSpace);
	cv::bilateralFilter(roiImg4, gaussImg4, d, sigmaColor, sigmaSpace);

	/* 分别计算四个ROI y方向的梯度值 */
	cv::Mat gradyImg1, gradyImg2, gradyImg3, gradyImg4;
	cv::Sobel(gaussImg1, gradyImg1, CV_8UC1, 0, 1, 3);
	cv::Sobel(gaussImg2, gradyImg2, CV_8UC1, 0, 1, 3);
	cv::Sobel(gaussImg3, gradyImg3, CV_8UC1, 0, 1, 3);
	cv::Sobel(gaussImg4, gradyImg4, CV_8UC1, 0, 1, 3);

	cv::Scalar meanGradyImg1 = cv::mean(gradyImg1);
	cv::Scalar meanGradyImg2 = cv::mean(gradyImg2);
	cv::Scalar meanGradyImg3 = cv::mean(gradyImg3);
	cv::Scalar meanGradyImg4 = cv::mean(gradyImg4);

	/* 用于接收y方向统计直方图的数组 */
	int *yHist1 = new int[gradyImg1.rows];
	int *yHist2 = new int[gradyImg2.rows];
	int *yHist3 = new int[gradyImg3.rows];
	int *yHist4 = new int[gradyImg4.rows];
	memset(yHist1, 0, gradyImg1.rows * sizeof(int));
	memset(yHist2, 0, gradyImg2.rows * sizeof(int));
	memset(yHist3, 0, gradyImg3.rows * sizeof(int));
	memset(yHist4, 0, gradyImg4.rows * sizeof(int));

	/* 对 ROI 区域1进行障碍物检测 */
	if (meanGradyImg1[0] < maxThresh && meanGradyImg1[0] > minThresh)
	{
		cv::Mat kernal = cv::Mat::ones(1, leftROIWidth / 5, CV_8UC1);      // 定义腐蚀变换的核
		cv::threshold(gradyImg1, gradyImg1, binaryThresh, 255, cv::THRESH_BINARY);
		cv::erode(gradyImg1, gradyImg1, kernal);                           // 对梯度的阈值图进行腐蚀运算
		for (int row = 0; row < gradyImg1.rows; ++row)
		{
			uchar *p = gradyImg1.ptr<uchar>(row);
			int rowSum = 0;
			for (int col = 0; col < gradyImg1.cols; ++col)       // 计算每一行的梯度和
				rowSum += (int)p[col];
			yHist1[row] += rowSum;                               // 记录下y方向的直方图信息
		}

		/* 进行障碍物搜索和定位的算法 */
		std::deque<ObstacleInfo> tmpObsDeque;
		std::vector<cv::Mat> obsTmpList;
		std::vector<int> predictRes;
		std::vector<cv::Vec2i> xyWraped;
		for (int i = 50; i < gradyImg1.rows - 50; ++i)
		{
			if (yHist1[i] != 0)
			{
				/* 进行障碍物定位的程序，即在一维数组中搜索波峰的数目 */
				int sumGrad = 0;
				for (int j = i; j < i + 50; ++j)
					sumGrad += 1;
				if (sumGrad > leftROIWidth)
				{
					numObstacles += 1;       // 障碍物的数目加1
					int xWraped = warpedLine1x1 - leftROIWidth / 2;
					int yWraped = i;
					xyWraped.push_back(cv::Vec2i(xWraped, yWraped));
					// 获取障碍物的候选区
					cv::Mat tmpImg = warpedImg(cv::Range(i - 50, i + 50),
						cv::Range(xWraped - 50, xWraped + 50));
					obsTmpList.push_back(tmpImg);
					i += 50;
				}
			}
		}

		if (-1 == callPythonFunc(obsTmpList, this->pFunc, predictRes))
		{
			return 0;
		}
		else
		{
			int xWraped, yWraped;
			for (size_t i = 0; i < predictRes.size(); ++i)
			{
				if (predictRes[i] == 1)
				{
					xWraped = xyWraped[i][0];
					yWraped = xyWraped[i][1];
					//std::vector<cv::Point2f> points = { cv::Point2f(xWraped, yWraped) }, pointTrans;
					/*
					cv::perspectiveTransform(points, pointTrans, Minv);
					tmpObsDeque.push_back(ObstacleInfo(cv::Point(pointTrans[0].x,
						pointTrans[0].y), 50, 50));
					*/
					tmpObsDeque.push_back(ObstacleInfo(cv::Point(xWraped, yWraped), 50, 50));
				}
			}
		}

		int head = 1, tail = 1;
		if (tmpObsDeque.size() > 5)
		{
			for (size_t i = 0; i < tmpObsDeque.size() - 1; ++i)
			{
				if (tmpObsDeque[i + 1].center.y - tmpObsDeque[i].center.y > minGap)
				{
					while (head < tail) {
						head++;
					}
					std::vector<cv::Point2f> points = { cv::Point2f(tmpObsDeque[head].center.x,
						tmpObsDeque[head].center.y) };
					std::vector<cv::Point2f> pointTrans;
					cv::perspectiveTransform(points, pointTrans, Minv);
					obstacleList.push_back(ObstacleInfo(cv::Point(pointTrans[0].x,
						pointTrans[0].y), 50, 50));
					head++;
					tail++;
				}
				else
					tail++;
			}
		}
		else
		{
			for (size_t i = 0; i < tmpObsDeque.size(); ++i)
			{
				std::vector<cv::Point2f> points = { cv::Point2f(tmpObsDeque[i].center.x,
					tmpObsDeque[i].center.y) };
				std::vector<cv::Point2f> pointTrans;
				cv::perspectiveTransform(points, pointTrans, Minv);
				obstacleList.push_back(ObstacleInfo(cv::Point(pointTrans[0].x,
					pointTrans[0].y), 50, 50));
			}
				//obstacleList.push_back(tmpObsDeque[i]);
		}
	}

	if (meanGradyImg2[0] < maxThresh && meanGradyImg2[0] > minThresh)
	{
		cv::Mat kernal = cv::Mat::ones(1, leftROIWidth / 5, CV_8UC1);
		cv::threshold(gradyImg2, gradyImg2, binaryThresh, 255, cv::THRESH_BINARY);
		cv::erode(gradyImg2, gradyImg2, kernal);
		for (int row = 0; row < gradyImg2.rows; ++row)
		{
			uchar *p = gradyImg2.ptr<uchar>(row);
			int rowSum = 0;
			for (int col = 0; col < gradyImg2.cols; ++col)
				rowSum += (int)p[col];
			yHist2[row] += rowSum;
		}

		/* 进行障碍物搜索和定位的算法 */
		std::deque<ObstacleInfo> tmpObsDeque;
		std::vector<cv::Mat> obsTmpList;
		std::vector<int> predictRes;
		std::vector<cv::Vec2i> xyWraped;
		for (int i = 50; i < gradyImg2.rows - 50; ++i)
		{
			if (yHist2[i] != 0)
			{
				/* 进行障碍物定位的程序，即在一维数组中搜索波峰的数目 */
				int sumGrad = 0;
				for (int j = i; j < i + 50; ++j)
					sumGrad += 1;
				if (sumGrad > leftROIWidth)
				{
					numObstacles += 1;       // 障碍物的数目加1
					int xWraped = warpedLine1x1 + leftROIWidth / 2;
					int yWraped = i;
					xyWraped.push_back(cv::Vec2i(xWraped, yWraped));
					// 获取障碍物的候选区
					cv::Mat tmpImg = warpedImg(cv::Range(i - 50, i + 50),
						cv::Range(xWraped - 50, xWraped + 50));
					obsTmpList.push_back(tmpImg);
					i += 50;
				}
			}
		}

		if (-1 == callPythonFunc(obsTmpList, this->pFunc, predictRes))
		{
			return 0;
		}
		else
		{
			int xWraped, yWraped;
			for (size_t i = 0; i < predictRes.size(); ++i)
			{
				if (predictRes[i] == 1)
				{
					xWraped = xyWraped[i][0];
					yWraped = xyWraped[i][1];
					/*
					std::vector<cv::Point2f> points = { cv::Point2f(xWraped, yWraped) }, pointTrans;
					cv::perspectiveTransform(points, pointTrans, Minv);
					tmpObsDeque.push_back(ObstacleInfo(cv::Point(pointTrans[0].x,
						pointTrans[0].y), 50, 50));
						*/
					tmpObsDeque.push_back(ObstacleInfo(cv::Point(xWraped, yWraped), 50, 50));
				}
			}
		}

		int head = 1, tail = 1;
		if (tmpObsDeque.size() > 5)
		{
			for (size_t i = 0; i < tmpObsDeque.size() - 1; ++i)
			{
				if (tmpObsDeque[i + 1].center.y - tmpObsDeque[i].center.y > minGap)
				{
					while (head < tail) {
						head++;
					}
					std::vector<cv::Point2f> points = { cv::Point2f(tmpObsDeque[head].center.x,
						tmpObsDeque[head].center.y) };
					std::vector<cv::Point2f> pointTrans;
					cv::perspectiveTransform(points, pointTrans, Minv);
					obstacleList.push_back(ObstacleInfo(cv::Point(pointTrans[0].x,
						pointTrans[0].y), 50, 50));
					head++;
					tail++;
				}
				else
					tail++;
			}
		}
		else
		{
			for (size_t i = 0; i < tmpObsDeque.size(); ++i) {
				std::vector<cv::Point2f> points = { cv::Point2f(tmpObsDeque[i].center.x,
					tmpObsDeque[i].center.y) };
				std::vector<cv::Point2f> pointTrans;
				cv::perspectiveTransform(points, pointTrans, Minv);
				obstacleList.push_back(ObstacleInfo(cv::Point(pointTrans[0].x,
					pointTrans[0].y), 50, 50));
			}
		}
	}

	if (meanGradyImg3[0] < maxThresh && meanGradyImg3[0] > minThresh)
	{
		cv::Mat kernal = cv::Mat::ones(1, rightROIWidth / 5, CV_8UC1);
		cv::threshold(gradyImg3, gradyImg3, binaryThresh, 255, cv::THRESH_BINARY);
		cv::erode(gradyImg3, gradyImg3, kernal);
		for (int row = 0; row < gradyImg3.rows; ++row)
		{
			uchar *p = gradyImg3.ptr<uchar>(row);
			int rowSum = 0;
			for (int col = 0; col < gradyImg3.cols; ++col)
				rowSum += (int)p[col];
			yHist3[row] += rowSum;
		}

		/* 进行障碍物搜索和定位的算法 */
		std::deque<ObstacleInfo> tmpObsDeque;
		std::vector<cv::Mat> obsTmpList;
		std::vector<int> predictRes;
		std::vector<cv::Vec2i> xyWraped;
		for (int i = 50; i < gradyImg3.rows - 50; ++i)
		{
			if (yHist3[i] != 0)
			{
				/* 进行障碍物定位的程序，即在一维数组中搜索波峰的数目 */
				int sumGrad = 0;
				for (int j = i; j < i + 50; ++j)
					sumGrad += 1;
				if (sumGrad > rightROIWidth)
				{
					int xWraped = warpedLine2x1 - rightROIWidth / 2;
					int yWraped = i;
					xyWraped.push_back(cv::Vec2i(xWraped, yWraped));
					// 获取障碍物的候选区
					cv::Mat tmpImg = warpedImg(cv::Range(i - 50, i + 50),
						cv::Range(xWraped - 50, xWraped + 50));
					obsTmpList.push_back(tmpImg);
					i += 50;
				}
			}
		}

		if (-1 == callPythonFunc(obsTmpList, this->pFunc, predictRes))
		{
			return 0;
		}
		else
		{
			int xWraped, yWraped;
			for (size_t i = 0; i < predictRes.size(); ++i)
			{
				if (predictRes[i] == 1)
				{
					xWraped = xyWraped[i][0];
					yWraped = xyWraped[i][1];
					/*
					std::vector<cv::Point2f> points = { cv::Point2f(xWraped, yWraped) }, pointTrans;
					cv::perspectiveTransform(points, pointTrans, Minv);
					tmpObsDeque.push_back(ObstacleInfo(cv::Point(pointTrans[0].x,
						pointTrans[0].y), 50, 50));
						*/
					tmpObsDeque.push_back(ObstacleInfo(cv::Point(xWraped, yWraped), 50, 50));
				}
			}
		}

		int head = 1, tail = 1;
		if (tmpObsDeque.size() > 5)
		{
			for (size_t i = 0; i < tmpObsDeque.size() - 1; ++i)
			{
				if (tmpObsDeque[i + 1].center.y - tmpObsDeque[i].center.y > minGap)
				{
					while (head < tail) {
						head++;
					}
					std::vector<cv::Point2f> points = { cv::Point2f(tmpObsDeque[head].center.x,
						tmpObsDeque[head].center.y) };
					std::vector<cv::Point2f> pointTrans;
					cv::perspectiveTransform(points, pointTrans, Minv);
					obstacleList.push_back(ObstacleInfo(cv::Point(pointTrans[0].x,
						pointTrans[0].y), 50, 50));
					head++;
					tail++;
				}
				else
					tail++;
			}
		}
		else
		{
			for (size_t i = 0; i < tmpObsDeque.size(); ++i) {
				std::vector<cv::Point2f> points = { cv::Point2f(tmpObsDeque[i].center.x,
					tmpObsDeque[i].center.y) };
				std::vector<cv::Point2f> pointTrans;
				cv::perspectiveTransform(points, pointTrans, Minv);
				obstacleList.push_back(ObstacleInfo(cv::Point(pointTrans[0].x,
					pointTrans[0].y), 50, 50));
			}
		}

	}

	if (meanGradyImg4[0] < maxThresh && meanGradyImg4[0] > minThresh)
	{
		cv::Mat kernal = cv::Mat::ones(1, rightROIWidth / 5, CV_8UC1);
		cv::threshold(gradyImg4, gradyImg4, binaryThresh, 255, cv::THRESH_BINARY);
		cv::erode(gradyImg4, gradyImg4, kernal);
		for (int row = 0; row < gradyImg4.rows; ++row)
		{
			uchar *p = gradyImg4.ptr<uchar>(row);
			int rowSum = 0;
			for (int col = 0; col < gradyImg4.cols; ++col)
				rowSum += (int)p[col];
			yHist4[row] += rowSum;
		}

		/* 进行障碍物识别的代码 */
		std::deque<ObstacleInfo> tmpObsDeque;
		std::vector<cv::Mat> obsTmpList;
		std::vector<int> predictRes;
		std::vector<cv::Vec2i> xyWraped;
		for (int i = 50; i < gradyImg4.rows - 50; ++i)
		{
			if (yHist4[i] != 0)
			{
				/* 进行障碍物定位的程序，即在一维数组中搜索波峰的数目 */
				int sumGrad = 0;
				for (int j = i; j < i + 50; ++j)
					sumGrad += 1;
				if (sumGrad > rightROIWidth)
				{
					int xWraped = warpedLine2x1 + rightROIWidth / 2;
					int yWraped = i;
					xyWraped.push_back(cv::Vec2i(xWraped, yWraped));
					// 获取障碍物的候选区
					cv::Mat tmpImg = warpedImg(cv::Range(i - 50, i + 50),
						cv::Range(xWraped - 50, xWraped + 50));
					obsTmpList.push_back(tmpImg);
					i += 50;
				}
			}
		}

		if (-1 == callPythonFunc(obsTmpList, this->pFunc, predictRes))
		{
			return 0;
		}
		else
		{
			int xWraped, yWraped;
			for (size_t i = 0; i < predictRes.size(); ++i)
			{
				if (predictRes[i] == 1)
				{
					xWraped = xyWraped[i][0];
					yWraped = xyWraped[i][1];
					/*
					std::vector<cv::Point2f> points = { cv::Point2f(xWraped, yWraped) }, pointTrans;
					cv::perspectiveTransform(points, pointTrans, Minv);
					tmpObsDeque.push_back(ObstacleInfo(cv::Point(pointTrans[0].x,
						pointTrans[0].y), 50, 50));
						*/
					tmpObsDeque.push_back(ObstacleInfo(cv::Point(xWraped, yWraped), 50, 50));
				}
			}
		}

		int head = 1, tail = 1;
		if (tmpObsDeque.size() > 5)
		{
			for (size_t i = 0; i < tmpObsDeque.size() - 1; ++i)
			{
				if (tmpObsDeque[i + 1].center.y - tmpObsDeque[i].center.y > minGap)
				{
					while (head < tail) {
						head++;
					}
					std::vector<cv::Point2f> points = { cv::Point2f(tmpObsDeque[head].center.x,
						tmpObsDeque[head].center.y) };
					std::vector<cv::Point2f> pointTrans;
					cv::perspectiveTransform(points, pointTrans, Minv);
					obstacleList.push_back(ObstacleInfo(cv::Point(pointTrans[0].x,
						pointTrans[0].y), 50, 50));
					head++;
					tail++;
				}
				else
					tail++;
			}
		}
		else
		{
			for (size_t i = 0; i < tmpObsDeque.size(); ++i) {
				std::vector<cv::Point2f> points = { cv::Point2f(tmpObsDeque[i].center.x,
					tmpObsDeque[i].center.y) };
				std::vector<cv::Point2f> pointTrans;
				cv::perspectiveTransform(points, pointTrans, Minv);
				obstacleList.push_back(ObstacleInfo(cv::Point(pointTrans[0].x,
					pointTrans[0].y), 50, 50));
			}
		}

	}

	delete[]yHist1;
	delete[]yHist2;
	delete[]yHist3;
	delete[]yHist4;
	return numObstacles;
}