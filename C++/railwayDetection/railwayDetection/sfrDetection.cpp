#include "stdafx.h"
#include "sfrDetection.h"


void sfrDetect(const cv::Mat &frame, std::vector<LinePoints> &lines,
	std::vector<ObstacleInfo> &obstacleList,
	std::deque<std::vector<LinePoints>> &lineDeque, int queueSize,
	LineDetector &detector, ObstacleDetector &obs, bool *initRoi)
{
	if (!frame.data)
	{
		std::cout << "error" << std::endl;
		return;
	}
	
	if (!lines.empty())
		lines.clear();

	if (!obstacleList.empty())
		obstacleList.clear();

	cv::Mat imgLine = frame.clone();
	cv::Mat imgObstacle = frame.clone();
	cv::Vec4i roiRect;                 // ROI�������������ֵ�����Ͻ���������½�����

	int height = frame.rows;
	int width = frame.cols;

	int meanX11 = 0, meanX12 = 0;      // ֱ��1�ľ�ֵ
	int meanX21 = 0, meanX22 = 0;      // ֱ��2�ľ�ֵ
	int y1 = 0, y2 = 0;                // ֱ�ߵ�������

	int numLinePairs = (int)lineDeque.size();
	if (numLinePairs > 0)
	{
		for (int i = 0; i < numLinePairs; ++i)
		{
			meanX11 += lineDeque[i][0].x1;
			meanX12 += lineDeque[i][0].x2;
			meanX21 += lineDeque[i][1].x1;
			meanX22 += lineDeque[i][1].x2;
		}
		meanX11 /= numLinePairs;
		meanX12 /= numLinePairs;
		meanX21 /= numLinePairs;
		meanX22 /= numLinePairs;
	}

	/*
	ָ������ͼƬԤ�����ֱ�߼��� Roi ����
	Roi ��������ǰ�ڼ�⵽��ֱ������ȷ�������ġ�
	*/
	if (lineDeque.size() == 0 || initRoi)       // ���ֱ�߶���Ϊ�գ���ʾǰ��û���ҵ����ʵ�ֱ��
	{
		roiRect = cv::Vec4i(width / 6,          // ͼƬ������������ ROI ������
			height * 2 / 3,
			width - width / 6 - 2,
			height - 1);
	}

	else {                           // ���ֱ�߶��в�Ϊ�գ�˵��ǰ���Ѿ��ҵ��˺��ʵ�ֱ��
									 // ����ǰ�ڼ�⵽��ֱ����������ǰ�� ROI ��С
		int minv = std::min({ meanX11, meanX12, meanX21, meanX22 });
		int maxv = std::max({ meanX11, meanX12, meanX21, meanX22 });
		roiRect = cv::Vec4i(std::max(0, minv - 100),
			height * 2 / 3,
			std::min(maxv + 100, width - 2),
			height - 1);
	}
	// ����ѡȡ��ROI����ֱ�ߵļ��
	detector.sfrDetection(frame, lines, height * 3 / 5, roiRect);

	// ����ֱ�߶����е�ֱ��ֵ�Ե�ǰ��⵽��ֱ�߽���΢��
	int linesSize = (int)lines.size();
	if (0 == linesSize)        //// û�м�⵽ֱ��
	{ 
		if (0 == numLinePairs)    // ����Ϊ��
		{
			*initRoi = true;       // ��Ҫʹ�ó�ʼ����ROI���½���ֱ�߼��
			return;
		}
		else                      // ���в�Ϊ��
		{
			y1 = lineDeque.back()[0].y1;
			y2 = lineDeque.back()[0].y2;
			lines.push_back(LinePoints(meanX11, y1, meanX12, y2));
			lines.push_back(LinePoints(meanX21, y1, meanX22, y2));
		}
	}
	else if (1 == linesSize)   //// ֻ��⵽1��ֱ��
	{ 
		if (0 == numLinePairs)  // ����Ϊ��
		{
			*initRoi = true;
			return;
		}
		else                         // ���в�Ϊ��
		{
			int x1 = lines[0].x1;
			int x2 = lines[0].x2;
			y1 = lines[0].y1;
			y2 = lines[0].y2;

			if (abs(x1 - meanX11) + abs(x2 - meanX12) < 25)    // ��⵽��ֱ��������е�һ���ľ�ֵ�ӽ�
			{
				lines.clear();
				lines.push_back(LinePoints((meanX11 + x1) / 2, y1, (meanX12 + x2) / 2, y2));
				lines.push_back(LinePoints(meanX21, y1, meanX22, y2));
				if (numLinePairs < 5)
					lineDeque.push_back(lines);
				else {
					lineDeque.pop_front();
					lineDeque.push_back(lines);
				}
			}

			else if (abs(x1 - meanX21) + abs(x2 - meanX22) < 25)  // ��⵽��ֱ��������еĵڶ����ľ�ֵ�ӽ�
			{
				lines.clear();
				lines.push_back(LinePoints(meanX11, y1, meanX12, y2));
				lines.push_back(LinePoints((meanX21 + x1) / 2, y1, (meanX22 + x2) / 2, y2));
				if (numLinePairs < 5)
					lineDeque.push_back(lines);
				else {
					lineDeque.pop_front();
					lineDeque.push_back(lines);
				}
			}

			else      // ��⵽��ֱ��������е�ֱ�߶����ӽ�
			{
				lines.clear();
				lines = lineDeque.back();
			}
		}
	}
	else if (2 == linesSize)   //// ��⵽����ֱ��
	{ 
		int x11 = lines[0].x1;
		int x12 = lines[0].x2;
		int x21 = lines[1].x1;
		int x22 = lines[1].x2;
		y1 = lines[0].y1;
		y2 = lines[0].y2;
		if (0 == numLinePairs)       // ֱ�߶���Ϊ��
		{
			*initRoi = true;          // ��Ҫʹ�ó�ʼ����ROI������һ֡�ļ��
			// ��⵽������ֱ�߷��������ߵ�����
			if (abs(x11 - x21) > width / 4 && abs(x11 - x21) < width / 2 &&
				abs(x11 - x21) > abs(x12 - x22) * 1.6 && (x11 - x21) * (x12 - x22) > 0)
			{
				lineDeque.push_back(lines);
			}
			// ��⵽������ֱ�߲����������ߵ�����
			else
			{
				lines.clear();
				return;
			}
		}
		else  // ֱ�߶��в�Ϊ��
		{
			// ��⵽������ֱ�߲����������ߵ�����
			if (abs(x11 - x21) > width / 2 || abs(x11 - x21) < width / 4 ||
				abs(x11 - x21) < abs(x12 - x22) * 1.6 || (x11 - x21) * (x12 - x22) <= 0)
			{
				lines.clear();
				lines.push_back(LinePoints(meanX11, y1, meanX12, y2));
				lines.push_back(LinePoints(meanX21, y1, meanX22, y2));
				*initRoi = true;
			}
			// ��⵽��ֱ�߷��������ߵ��������������еĽ������ϴ�
			else if (abs(meanX11 - x11) + abs(meanX12 - x12) > 200
					 || abs(meanX21 - x21) + abs(meanX22 - x22) > 200)
			{
				lines.clear();
				lines.push_back(LinePoints(meanX11, y1, meanX12, y2));
				lines.push_back(LinePoints(meanX21, y1, meanX22, y2));
				lineDeque.pop_front();
				*initRoi = false;   // ���ڶ����еĽ������һ֡����ROI���е���
			}
			// x11,x12 ������еĵ�һ��ֱ�ߵľ�ֵ�ӽ�
			else if (abs(x11 - meanX11) + abs(x12 - meanX12) < 25)
			{
				lines.clear();
				lines.push_back(LinePoints((x11 + meanX11) / 2, y1, (x12 + meanX12) / 2, y2));
				if (abs(x21 - meanX21) / 2 + abs(x22 - meanX22) / 2 < 25) {
					lines.push_back(LinePoints((x21 + meanX21) / 2, y1, (x22 + meanX22) / 2, y2));
				}
				else
					lines.push_back(LinePoints(meanX21, y1, meanX22, y2));
				*initRoi = false;
			}
			// x21, x22 ������еĵ�һ��ֱ�ߵľ�ֵ�ӽ�
			else if (abs(x21 - meanX11) + abs(x22 - meanX22) < 25)
			{
				lines.clear();
				lines.push_back(LinePoints((x21 + meanX11) / 2, y1, (x22 + meanX12) / 2, y2));
				if (abs(x11 - meanX21) + abs(x12 - meanX22) < 25)
				{
					lines.push_back(LinePoints((x11 + meanX21) / 2, y1, (x12 + meanX22) / 2, y2));
				}
				else
					lines.push_back(LinePoints(meanX21, y1, meanX22, y2));
				*initRoi = false;
			}
			// ��⵽������ֱ��������е�ֱ�����ϴ�
			else
			{
				lines.clear();
				lines.push_back(LinePoints(meanX11, y1, meanX12, y2));
				lines.push_back(LinePoints(meanX21, y1, meanX22, y2));
			}
			if (numLinePairs < 5)
				lineDeque.push_back(lines);
			else {
				lineDeque.pop_front();
				lineDeque.push_back(lines);
			}
		}
	}
	else                       //// ��⵽��������ֱ��
	{ 
		if (0 == numLinePairs)
			return;
		else       // ֱ�߶��в�Ϊ��
		{
			LinePoints tmpLine1, tmpLine2;
			y1 = lines[0].y1;
			y2 = lines[0].y2;
			for (size_t i = 0; i < lines.size(); ++i)
			{
				if (abs(lines[i].x1 - meanX11) + abs(lines[i].x2 - meanX12) < 25
					&& abs(lines[i].x1 - meanX11) < abs(tmpLine1.x1 - meanX11))
					tmpLine1 = LinePoints((lines[i].x1 + meanX11)/2, y1, (lines[i].x2 + meanX12)/2, y2);
				if (abs(lines[i].x1 - meanX21) + abs(lines[i].x2 - meanX22) < 25
					&& abs(lines[i].x1 - meanX21) < abs(tmpLine2.x1 - meanX21))
					tmpLine2 = LinePoints((lines[i].x1 + meanX21)/2, y1, (lines[i].x2 + meanX22)/2, y2);
			}
			if (tmpLine1.x1 == 0)          // ���û��ȡ�����ʵ�tmpLine1
				tmpLine1 = lineDeque.back()[0];
			if (tmpLine2.x1 == 0)          // ���û��ȡ�����ʵ�tmpLine2
				tmpLine2 = lineDeque.back()[1];
			lines.clear();
			lines.push_back(tmpLine1);
			lines.push_back(tmpLine2);
		}
	}

	// ������մ�������ֱ�ߣ�������ϰ�����
	if (lines.size() == 2)
	{
		obs.obstacleDetectionWithModel(frame, lines, obstacleList);
	}
}

void videoTest(std::string filepath)
{
	cv::VideoCapture cap(filepath);
	if (!cap.isOpened())
	{
		std::cerr << "can't open a file" << std::endl;
		return;
	}

	cv::Mat imgLine, imgObstacle;

	std::deque<std::vector<LinePoints>> lineDeque;
	std::vector<LinePoints> lines;
	std::vector<ObstacleInfo> obstacleList;
	LineDetector detector;
	ObstacleDetector obs;
	int num = 0;
	bool initRoi = true;
	clock_t start, end;

	for (;;)
	{
		start = clock();
		num++;
		cv::Mat img;
		cap >> img;
		if (!img.data)
			return;
		std::cout << "��ǰ�ǵ� " << num << "֡" << std::endl;
		imgLine = img.clone();
		imgObstacle = img.clone();
		sfrDetect(img, lines, obstacleList, lineDeque, 5, detector, obs, &initRoi);
		std::cout << "ֱ����Ŀ��" << (int)lines.size() << std::endl;
		std::cout << "�ϰ�����Ŀ��" << (int)obstacleList.size() << std::endl;
		for (int i = 0; i < lines.size(); ++i)
			cv::line(imgObstacle, cv::Point(lines[i].x1, lines[i].y1),
				cv::Point(lines[i].x2, lines[i].y2), cv::Scalar(0, 0, 255), 2);

		if (obstacleList.size() > 0)
		{
			for (int i = 0; i < obstacleList.size(); ++i)
			{
				cv::rectangle(imgObstacle, cv::Rect(obstacleList[i].center.x - obstacleList[i].width / 2,
					obstacleList[i].center.y - obstacleList[i].height / 2,
					obstacleList[i].width, obstacleList[i].height), cv::Scalar(0, 0, 255), 4);
			}
			std::cout << std::endl;
		}

		cv::imshow("imgLine", imgObstacle);
		if (cv::waitKey(30) >= 0)
			break;

		end = clock();
		std::cout << "time used per frame: " << (double)(end - start) / CLOCKS_PER_SEC << " s" << std::endl;
	}
}