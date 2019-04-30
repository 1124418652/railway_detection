#include "stdafx.h"
#include "lineDetection.h"


void LineDetector::preProcession(const cv::Mat &img, const cv::Vec4i &roiRect)
{
	int width = img.cols;      // 获取图片的宽
	int height = img.rows;     // 获取图片的高
	int lx = roiRect[0];       // 获取 ROI 的位置和尺寸信息
	int ly = roiRect[1];
	int rx = roiRect[2];
	int ry = roiRect[3];
	assert(lx >= 0 && lx < width && rx < width && lx < rx);
	assert(ly >= 0 && ly < height && ry < height && ly < ry);

	int rh = ry - ly;
	int rw = rx - lx;

	this->roiRect = roiRect;
	cv::Mat roiSource;         // 未做亮度调节前的 roi
	cv::Mat roi;               // 亮度调节之后的 roi
	std::vector<cv::Mat> channels;
	cv::Mat gauss(cv::Size(rw, rh), CV_8UC1);
	roiSource = img(cv::Range(ly, ry), cv::Range(lx, rx)).clone();

	/* 降低图片 ROI 的亮度 */
	cv::cvtColor(roiSource, roi, cv::COLOR_BGR2HSV);
	cv::split(roi, channels);
	channels[2] = channels[2] * 0.6;
	cv::merge(channels, roi);
	cv::cvtColor(roi, roi, cv::COLOR_HSV2BGR);

	cv::cvtColor(roi, this->grayImg, cv::COLOR_BGR2GRAY);
	
	/* 分区域高斯模糊 */
	cv::GaussianBlur(this->grayImg(cv::Range(0, rh / 4), cv::Range::all()),    // 对 roi 的上 1/4 进行滤波
		gauss(cv::Range(0, rh / 4), cv::Range::all()), 
		this->ksize, this->sigmaX / 2, this->sigmaY / 4);                      // 采用较小的尺度
	cv::GaussianBlur(this->grayImg(cv::Range(rh / 4, rh / 2), cv::Range::all()),   
		gauss(cv::Range(rh / 4, rh / 2), cv::Range::all()),
		this->ksize, this->sigmaX * 2 / 3, this->sigmaY / 3);
	cv::GaussianBlur(this->grayImg(cv::Range(rh / 2, rh), cv::Range::all()),  
		gauss(cv::Range(rh / 2, rh), cv::Range::all()),
		this->ksize, this->sigmaX, this->sigmaY);

	this->gaussImg = gauss.clone();
}

void LineDetector::detectionWithHoughLines(std::vector<cv::Vec4f> &rhoTheta)
{
	// assert(this->gaussImg.data);
	cv::Mat cannyEdge;
	std::vector<cv::Vec2f> lines;     // 传入HoughLines函数中保存出直线信息
	
	cv::Canny(this->gaussImg, cannyEdge, 50, 150, 3);
	cv::HoughLines(cannyEdge, lines, this->rho, this->theta, this->threshold);
	// std::cout << lines.size();
	if (0 == lines.size())
		return;

	if (!rhoTheta.empty())
		rhoTheta.clear();

	for (size_t i = 0; i < lines.size(); ++i)
	{
		float rho = lines[i][0], theta = lines[i][1];
		if (theta < this->maxTheta || theta > PI - this->maxTheta)
		{
			float a = cos(theta);
			float b = sin(theta);
			float x0 = rho * a;
			float y0 = rho * b;
			rhoTheta.push_back(cv::Vec4f(a, b, x0, y0));
		}
	}
	this->cannyImg = cannyEdge;
}

void LineDetector::sfrDetection(const cv::Mat &img, std::vector<LinePoints> &lines,
	int y2, cv::Vec4i &roiRect)
{
	assert(img.data);
	int width = img.cols;
	int height = img.rows;
	int y1 = height;
	if (y2 < 0)
		y2 = height * 2 / 3;             //// y2 后续可能会设计成可以手动调整的参数
	if (roiRect[2] == 0 || roiRect[3] == 0)
	{
		roiRect = cv::Vec4i(width / 6,
			height * 2 / 3,
			width - width / 6 - 2,
			height - 1);
	}
	if (!lines.empty())
		lines.clear();
	preProcession(img, roiRect);             // 通过调用该函数给 this->gaussImg 赋值
	assert(this->gaussImg.data);
	detectionWithHoughLines(this->rhoTheta);

	if (0 == this->rhoTheta.size())
		return;

	std::vector<cv::Vec3i> bottomPoints[5];  // 将直线底部的点分成5个部分

	size_t i = 0;
	while(i < rhoTheta.size())
	{
		float cos_t = this->rhoTheta[i][0];
		float sin_t = this->rhoTheta[i][1];
		float x0 = this->rhoTheta[i][2];
		float y0 = this->rhoTheta[i][3];

		/* 检测是在 ROI 中进行的，这里需要先将检测出来的直线坐标转换回原坐标系中 */
		float r = (x0 + roiRect[0]) * cos_t + (y0 + roiRect[1]) * sin_t;
		
		int x1 = (r - y1 * sin_t) / cos_t;
		int x2 = (r - y2 * sin_t) / cos_t;
		
		/* 将直线按照低点落在五等分中的哪个区域进行归类 */
		int index = x1 / (width / 5);
		if (x1 < 0)
		{
			x1 = 0;
			y1 = r / sin_t;
			int j = 0;
			for (j = 0; j < 5; ++j)
				if (!bottomPoints[j].empty())
					bottomPoints[j].clear();
			i = 0;
		}
		if (x1 > width - 1)
		{
			x1 = width - 1;
			y1 = (r - x1 * cos_t) / sin_t;
			int j = 0;
			for (j = 0; j < 5; ++j)
				if (!bottomPoints[j].empty())
					bottomPoints[j].clear();
			i = 0;
		}
		else
			bottomPoints[index].push_back(cv::Vec3i(x1, x2, i));  // 该直线在rhoTheta中的索引值也需要进行保存
		++i;
	}

	/* 对每个区域中的直线求均值，然后找出离均值最近的直线，使用LinePoints结构进行保存 */
	for (int i = 0; i < 5; ++i)
	{
		if (bottomPoints[i].size() == 0)
			continue;
		int sum = 0;        // 记录每个区域直线底坐标的和
		int mean = 0;       // 记录每个区域直线底坐标的均值
		int min = 0;        // 记录里该区域的均值最近的直线段的索引
		for (auto val : bottomPoints[i])
			sum += val[0];
		mean = sum / (int)bottomPoints[i].size();
		for (int j = 0; j < bottomPoints[i].size(); ++j)
		{
			cv::Vec3i val = bottomPoints[i][j];
			if (abs(val[0] - mean) < abs(bottomPoints[i][min][0] - mean))
				min = j;
		}
		lines.push_back(LinePoints(bottomPoints[i][min][0], y1, 
								   bottomPoints[i][min][1], y2));
	}
}
