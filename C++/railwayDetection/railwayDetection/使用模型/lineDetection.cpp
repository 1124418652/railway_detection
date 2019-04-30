#include "stdafx.h"
#include "lineDetection.h"


void LineDetector::preProcession(const cv::Mat &img, const cv::Vec4i &roiRect)
{
	int width = img.cols;      // ��ȡͼƬ�Ŀ�
	int height = img.rows;     // ��ȡͼƬ�ĸ�
	int lx = roiRect[0];       // ��ȡ ROI ��λ�úͳߴ���Ϣ
	int ly = roiRect[1];
	int rx = roiRect[2];
	int ry = roiRect[3];
	assert(lx >= 0 && lx < width && rx < width && lx < rx);
	assert(ly >= 0 && ly < height && ry < height && ly < ry);

	int rh = ry - ly;
	int rw = rx - lx;

	this->roiRect = roiRect;
	cv::Mat roiSource;         // δ�����ȵ���ǰ�� roi
	cv::Mat roi;               // ���ȵ���֮��� roi
	std::vector<cv::Mat> channels;
	cv::Mat gauss(cv::Size(rw, rh), CV_8UC1);
	roiSource = img(cv::Range(ly, ry), cv::Range(lx, rx)).clone();

	/* ����ͼƬ ROI ������ */
	cv::cvtColor(roiSource, roi, cv::COLOR_BGR2HSV);
	cv::split(roi, channels);
	channels[2] = channels[2] * 0.6;
	cv::merge(channels, roi);
	cv::cvtColor(roi, roi, cv::COLOR_HSV2BGR);

	cv::cvtColor(roi, this->grayImg, cv::COLOR_BGR2GRAY);
	
	/* �������˹ģ�� */
	cv::GaussianBlur(this->grayImg(cv::Range(0, rh / 4), cv::Range::all()),    // �� roi ���� 1/4 �����˲�
		gauss(cv::Range(0, rh / 4), cv::Range::all()), 
		this->ksize, this->sigmaX / 2, this->sigmaY / 4);                      // ���ý�С�ĳ߶�
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
	std::vector<cv::Vec2f> lines;     // ����HoughLines�����б����ֱ����Ϣ
	
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
		y2 = height * 2 / 3;             //// y2 �������ܻ���Ƴɿ����ֶ������Ĳ���
	if (roiRect[2] == 0 || roiRect[3] == 0)
	{
		roiRect = cv::Vec4i(width / 6,
			height * 2 / 3,
			width - width / 6 - 2,
			height - 1);
	}
	if (!lines.empty())
		lines.clear();
	preProcession(img, roiRect);             // ͨ�����øú����� this->gaussImg ��ֵ
	assert(this->gaussImg.data);
	detectionWithHoughLines(this->rhoTheta);

	if (0 == this->rhoTheta.size())
		return;

	std::vector<cv::Vec3i> bottomPoints[5];  // ��ֱ�ߵײ��ĵ�ֳ�5������

	size_t i = 0;
	while(i < rhoTheta.size())
	{
		float cos_t = this->rhoTheta[i][0];
		float sin_t = this->rhoTheta[i][1];
		float x0 = this->rhoTheta[i][2];
		float y0 = this->rhoTheta[i][3];

		/* ������� ROI �н��еģ�������Ҫ�Ƚ���������ֱ������ת����ԭ����ϵ�� */
		float r = (x0 + roiRect[0]) * cos_t + (y0 + roiRect[1]) * sin_t;
		
		int x1 = (r - y1 * sin_t) / cos_t;
		int x2 = (r - y2 * sin_t) / cos_t;
		
		/* ��ֱ�߰��յ͵�������ȷ��е��ĸ�������й��� */
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
			bottomPoints[index].push_back(cv::Vec3i(x1, x2, i));  // ��ֱ����rhoTheta�е�����ֵҲ��Ҫ���б���
		++i;
	}

	/* ��ÿ�������е�ֱ�����ֵ��Ȼ���ҳ����ֵ�����ֱ�ߣ�ʹ��LinePoints�ṹ���б��� */
	for (int i = 0; i < 5; ++i)
	{
		if (bottomPoints[i].size() == 0)
			continue;
		int sum = 0;        // ��¼ÿ������ֱ�ߵ�����ĺ�
		int mean = 0;       // ��¼ÿ������ֱ�ߵ�����ľ�ֵ
		int min = 0;        // ��¼�������ľ�ֵ�����ֱ�߶ε�����
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
