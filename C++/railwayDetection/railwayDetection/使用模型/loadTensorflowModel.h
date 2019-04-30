#pragma once
#ifndef __LOAD_TENSORFLOW_MODEL 
#define __LOAD_TENSORFLOW_MODEL
#include <iostream>
#include "include/Python.h"
//#include <windows.h>
#include <opencv2/core.hpp>


/*
 ȫ�ֽ��ͺ��߳�������
*/
class PyGILThreadLock
{
public:
	PyGILThreadLock()
	{
		//_save = NULL;
		nStatus = 0;
		nStatus = PyGILState_Check();        // ��⵱ǰ�߳��Ƿ�ӵ��GIL
		if (!nStatus)
		{
			gstate = PyGILState_Ensure();    // �����ǰ�߳�û��GIL���������ȡGIL
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
 *brief ��ʼ��Python��������Python�ļ�������Python�ű��ļ��ͺ���������
        �������ô����������true�����ʾ���Լ����������к������֡���Ҫ
		ע������ں�����Ҫ�ı�ָ�뱾��������ָ��ָ���ַ��ֵ�����Ա���
		Ҫ����ָ���ָ�롣������Ϊ�βε�ָ��ֻ��һ����ʱ����������Ҫ�ı�
		��ָ��ָ��ͬ���ĵ�ַ��������ͬһ��ָ�롣
 *params module ָ��PyObject���ָ�룬Python�ű��ļ���ģ�飩�Ķ���
 *params pFunc ָ��PyObject���ָ�룬Python�еĺ�������
 *returns �������ɹ�������true�����򷵻�false
 */
bool loadPyModule(PyObject **module, PyObject **pDict, PyObject **pFunc);

/*
 *brief ��C++�е�Mat����ת��ΪPython�е�Numpy����
 *params img cv::Mat���͵Ķ����ڸú�������ҪתΪPython�е�List����Ȼ��
			 ��python�ű�ת��Ϊndarray��ʽ
 *params pFunc PyObject��Ķ���ָ��,ָ����Ҫ���õĺ���ָ��
 */
int callPythonFunc(const cv::Mat &img, PyObject *pFunc);
#endif // !__LOAD_TENSORFLOW_MODEL 


