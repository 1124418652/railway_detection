#include "stdafx.h"
#include "videoDetection.h"


void videoDetection(int queueSize, VideoFrom vf, std::string filepath)
{
	int queCount = 0;
	int num = 0;
	clock_t start, end;

	cv::VideoCapture cap;
	if (FROM_CAMERA == vf)
		cap = cv::VideoCapture(0);
	else if (FROM_FILE == vf)
		cap = cv::VideoCapture(filepath);

	if (!cap.isOpened())
	{
		std::cerr << "Can't open a camera or file" << std::endl;
		return;
	}

	std::deque<std::vector<LinePoints>> lineDeque;   // 用于保存视频帧中正确检测的直线的队列
	cv::Mat imgLine;
	cv::Mat imgObstacle;
	LineDetector detector;
	ObstacleDetector obs;
	std::vector<LinePoints> lines;                         
	cv::Vec4i roiRect;

	for (;;)
	{
		start = clock();
		cv::Mat img;
		cap >> img;
		if (!img.data)
		{
			std::cout << "error" << std::endl;
			break;
		}
		std::cout << "number of frame: " << num << std::endl;
		num += 1;
		imgLine = img.clone();
		imgObstacle = img.clone();

		int height = img.rows;
		int width = img.cols;

		int meanX11 = 0, meanX12 = 0;    // 用于记录直线队列中的直线的4个坐标均值
		int meanX21 = 0, meanX22 = 0;
		int y1 = 0, y2 = 0;
		
		/* 
		指定进行图片预处理和直线检测的 Roi 区域，
		Roi 区域是由前期检测到的直线重新确定调整的。
		*/
		if (lineDeque.size() == 0)       // 如果直线队列为空，表示前期没有找到合适的直线
		{
			roiRect = cv::Vec4i(width / 6,                  // 图片中用于做检测的 ROI 区域定义
				height * 2 / 3,
				width - width / 6 - 2,
				height - 1);
		}
		 
		else {                           // 如果直线队列不为空，说明前期已经找到了合适的直线
			int numLinePairs = (int)lineDeque.size();
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

			int minv = std::min({ meanX11, meanX12, meanX21, meanX22 });
			int maxv = std::max({ meanX11, meanX12, meanX21, meanX22 });

			// 基于前期检测到的直线来调整当前的 ROI 大小
			roiRect = cv::Vec4i(std::max(0, minv - 100), 
								height * 2 / 3, 
								std::min(maxv + 100, width - 2), 
								height - 1);
		}

		// std::cout << roiRect << std::endl;
		/* 基于选取的 ROI 进行直线的检测 */
		detector.sfrDetection(img, lines, height * 3 / 5, roiRect);
		/* 根据前期检测到的直线队列，对当前检测到的直线进行微调 */
		int linesSize = (int)lines.size();
		std::cout << "number of lines： " << lines.size() << std::endl;
		if (0 == linesSize)         // 当前帧没有检测到直线
		{
			if (0 == (int)lineDeque.size())     // 直线队列为空
				continue;
			else {                              // 直线队列不为空
				y1 = lineDeque.back()[0].y1;
				y2 = lineDeque.back()[0].y2;
				lines.push_back(LinePoints(meanX11, y1, meanX12, y2));
				lines.push_back(LinePoints(meanX21, y1, meanX22, y2));
				queCount += 1;
				if (queCount == 5)
				{
					lineDeque.pop_front();
					queCount = 0;
				}
			}
		}

		else if (1 == linesSize)    // 当前帧只检测到一条直线
		{
			if (0 == (int)lineDeque.size())           /////// 这一种情况先做保留，后续可能需要进行修改
				continue;
			else                                // 直线队列不为空
			{
				int x1 = lines[0].x1;
				int x2 = lines[0].x2;
				y1 = lines[0].y1;
				y2 = lines[0].y2;
				if (abs(x1 - meanX11) < 25 && abs(x2 - meanX12) < 25)
				{
					lines.clear();
					lines.push_back(LinePoints((meanX11 + x1) / 2, y1, (meanX12 + x2) / 2, y2));
					lines.push_back(lineDeque.back()[1]);
					if ((int)lineDeque.size() < 5)
					{
						lineDeque.push_back(lines);
					}
					else {
						lineDeque.pop_front();
						lineDeque.push_back(lines);
					}
				}
				else if (abs(x1 - meanX21) < 25 && abs(x2 - meanX22) < 25)
				{
					lines.clear();
					lines.push_back(LinePoints((meanX21 + x1) / 2, y1, (meanX22 + x2) / 2, y2));
					lines.push_back(lineDeque.back()[0]);
					if ((int)lineDeque.size() < 5)
					{
						lineDeque.push_back(lines);
					}
					else {
						lineDeque.pop_front();
						lineDeque.push_back(lines);
					}
				}
				else
				{
					lines.clear();
					lines = lineDeque.back();
				}
			}
		}

		else if (2 == linesSize)    // 当前帧检测到两条直线
		{
			int x11 = lines[0].x1;
			int x12 = lines[0].x2;
			int x21 = lines[1].x1;
			int x22 = lines[1].x2;
			y1 = lines[0].y1;
			y2 = lines[0].y2;
			if (0 == (int)lineDeque.size())      // 直线队列为空
			{
				if (abs(x11 - x21) > width / 4 && abs(x11 - x21) < width / 2 && 
					abs(x11 - x21) > abs(x12 - x22) * 1.6 && (x11 - x21) * (x12 - x22) > 0)
					lineDeque.push_back(lines);
				else
				{
					lines.clear();
					continue;
				}
			}
			else {                               // 直线队列不为空
				if (abs(x11 - x21) > width / 2 || abs(x11 - x21) < width / 4 || 
					abs(x11 - x21) < abs(x12 - x22) * 1.6 || (x11 - x21) * (x12 - x22) <= 0)
				{
					lines.clear();
					lineDeque.pop_front();
					goto here;
				}
				else if (abs(meanX11 - x11) + abs(meanX12 - x12) > 200 
					|| abs(meanX21 - x21) + abs(meanX22 - x22) > 200)
				{
					lines.clear();
					lineDeque.pop_front();
					goto here;
				}
				else if (abs(x11 - meanX11) < 25 && abs(x12 - meanX12) < 25)
				{
					lines.clear();
					lines.push_back(LinePoints((x11 + meanX11) / 2, y1, (x12 + meanX12) / 2, y2));
					if (abs(x21 - meanX21) < 25 && abs(x22 - meanX22) < 25)
						lines.push_back(LinePoints((x21 + meanX21) / 2, y1, (x22 + meanX22) / 2, y2));
					else
						lines.push_back(LinePoints(meanX21, y1, meanX22, y2));
				}
				else if (abs(x21 - meanX11) < 25 && abs(x22 - meanX12) < 25)
				{
					lines.clear();
					lines.push_back(LinePoints((x21 + meanX11) / 2, y1, (x22 + meanX12) / 2, y2));
					if (abs(x11 - meanX21) < 25 && abs(x12 - meanX22) < 25)
						lines.push_back(LinePoints((x11 + meanX21) / 2, y1, (x12 + meanX22) / 2, y2));
					else
						lines.push_back(LinePoints(meanX21, y1, meanX22, y2));
				}
				else
				{
					//lines.clear();
					//lines.push_back(LinePoints(meanX11, y1, meanX12, y2));
					//lines.push_back(LinePoints(meanX21, y1, meanX22, y2));
					lineDeque.clear();
				}
				if ((int)lineDeque.size() < 5)
					lineDeque.push_back(lines);
				else {
					lineDeque.pop_front();
					lineDeque.push_back(lines);
				}
			}
		}

		else                     // 当前帧检测到两条以上的直线
		{
			y1 = lines[0].y1;
			y2 = lines[0].y2;
			if (lineDeque.size() > 0)      // 直线队列不为空
			{
				LinePoints tmpLine1, tmpLine2;
				for (size_t i = 0; i < lines.size(); ++i)
				{
					if (lines[i].x1 - meanX11 < 5 && lines[i].x1 - meanX11 < tmpLine1.x1 - meanX11) {
						tmpLine1 = lines[i];
					}
					if (lines[i].x1 - meanX21 < 5 && lines[i].x1 - meanX21 < tmpLine2.x1 - meanX21) {
						tmpLine2 = lines[i];
					}
				}
				lines.clear();
				// 完成lines中的直线的微调，并且去除多余的线
				if (tmpLine1.x1 - meanX11 < 5 && tmpLine1.x2 - meanX12 < 5) {
					lines.push_back(LinePoints((meanX11 + tmpLine1.x1) / 2, tmpLine1.y1, 
											   (meanX12 + tmpLine1.x2) / 2, tmpLine1.y2));
					if (tmpLine2.x1 - meanX21 < 5 && tmpLine2.x2 - meanX22 < 5) {
						lines.push_back(LinePoints((meanX21 + tmpLine2.x1) / 2, tmpLine2.y1, 
												   (meanX22 + tmpLine2.x2) / 2, tmpLine2.y2));
					}
					else {
						lines.push_back(LinePoints(meanX21, y1, meanX22, y2));
					}
				}
				else { 
					lines.push_back(LinePoints(meanX11, y1, meanX12, y2));
					if (tmpLine2.x1 - meanX21 < 5 && tmpLine2.x2 - meanX22 < 5) {
						lines.push_back(LinePoints((meanX21 + tmpLine2.x1) / 2, tmpLine2.y1,
							(meanX22 + tmpLine2.x2) / 2, tmpLine2.y2));
					}
					else {
						lines.push_back(LinePoints(meanX21, y1, meanX22, y2));
					}
					lineDeque.pop_front();
				}
				// 将当前帧的直线加入直线队列中
				if (lineDeque.size() < 5)
					lineDeque.push_back(lines);
				else {
					lineDeque.pop_front();
					lineDeque.push_back(lines);
				}
			}
			else                         // 直线队列为空 
			{
				goto here;
			}
		}

		for (int i = 0; i < lines.size(); ++i)
			cv::line(imgObstacle, cv::Point(lines[i].x1, lines[i].y1),
				cv::Point(lines[i].x2, lines[i].y2), cv::Scalar(0, 0, 255), 2);

		if (lines.size() == 2)
		{
			//ObstacleDetector obs;
			std::vector<ObstacleInfo> obstacleList;
		 	obs.obstacleDetectionWithModel(img, lines, obstacleList);
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
		}

		here:
		cv::imshow("imgLine", imgObstacle);
		if (cv::waitKey(30) >= 0)
			break;
		
		end = clock();
		std::cout << "time used per frame: " << (double)(end - start) / CLOCKS_PER_SEC << " s" << std::endl;
	}
}