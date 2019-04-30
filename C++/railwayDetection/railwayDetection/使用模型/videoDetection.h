#pragma once

#ifndef __VIDEO_DETECTION
#define __VIDEO_DETECTION

#include <iostream>
#include <vector>
#include <string>
#include <deque>
#include <ctime>
#include "lineDetection.h"
#include "obstacleDetection.h"

enum VideoFrom        // ���ڱ�����Ƶ�����������ȡ�����ļ�
{
	FROM_CAMERA,
	FROM_FILE
};

void videoDetection(int queueSize, VideoFrom vf = FROM_FILE, std::string filepath = 0);

#endif // !__VIDEO_DETECTION
