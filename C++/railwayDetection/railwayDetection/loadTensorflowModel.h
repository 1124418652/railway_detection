#pragma once
#ifndef __LOAD_TENSORFLOW_MODEL 
#define __LOAD_TENSORFLOW_MODEL
#include <iostream>
#include "include/Python.h"
//#include <windows.h>
#include <opencv2/core.hpp>


/*
 全局解释和线程锁对象
*/
class PyGILThreadLock
{
public:
	PyGILThreadLock()
	{
		//_save = NULL;
		nStatus = 0;
		nStatus = PyGILState_Check();        // 检测当前线程是否拥有GIL
		if (!nStatus)
		{
			gstate = PyGILState_Ensure();    // 如果当前线程没有GIL，则申请获取GIL
			nStatus = 1;
		}
		//_save = PyEval_SaveThread();
		//PyEval_RestoreThread(_save);
	}
	~PyGILThreadLock()
	{
		//_save = PyEval_SaveThread();
		//PyEval_RestoreThread(_save);
		if (nStatus)
		{
			PyGILState_Release(gstate);
		}
	}
private:
	PyGILState_STATE gstate;
	//PyThreadState *_save;
	int nStatus;
};

/*
 *brief 初始化Python解释器和Python文件，导入Python脚本文件和函数，不做
        函数调用处理，如果返回true，则表示可以继续主程序中后续部分。需要
		注意的是在函数中要改变指针本身，而不是指针指向地址的值，所以必须
		要传入指针的指针。否则作为形参的指针只是一个临时变量，和需要改变
		的指针指向同样的地址，但不是同一个指针。
 *params module 指向PyObject类的指针，Python脚本文件（模块）的对象
 *params pFunc 指向PyObject类的指针，Python中的函数对象
 *returns 如果导入成功，返回true，否则返回false
 */
bool loadPyModule(PyObject **module, PyObject **pDict, PyObject **pFunc);

/*
 *brief 将C++中的Mat对象转变为Python中的Numpy数组
 *params img cv::Mat类型的对象，在该函数中需要转为Python中的List类型然后
			 由python脚本转换为ndarray格式
 *params pFunc PyObject类的对象指针,指向需要调用的函数指针
 */
int callPythonFunc(const cv::Mat &img, PyObject *pFunc);
#endif // !__LOAD_TENSORFLOW_MODEL 


