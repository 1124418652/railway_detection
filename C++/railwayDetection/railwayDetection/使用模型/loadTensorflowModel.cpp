#include "stdafx.h"
#include "loadTensorflowModel.h"

bool loadPyModule(PyObject **module, PyObject **pDict, PyObject **pFunc)
{
	try {
		Py_SetPythonHome(L"D:/software/Anoconda");
		Py_Initialize();
		PyEval_InitThreads();

		if (!Py_IsInitialized())
		{
			std::cout << "Can't initialize Python!" << std::endl;
		}

		PyRun_SimpleString("import sys");
		PyRun_SimpleString("sys.path.append('./')");

		*module = PyImport_ImportModule("model_test");      // 用于测试模型的 Python 文件
		if (!*module)             // 如果不能正确导入该模块
		{
			std::cout << "Can't open module!" << std::endl;
			Py_Finalize();
			return 0;
		}

		// 从 module 模块中导入函数名为 recognize 的函数
		*pFunc = PyObject_GetAttrString(*module, "predict");
		if (!pFunc)
		{
			std::cout << "Can't open FUNC!" << std::endl;
			Py_Finalize();
		}
		PyObject *load = PyObject_GetAttrString(*module, "load_model");
		if (!load)
		{
			std::cout << "Can't open FUNC!" << std::endl;
			Py_Finalize();
		}
		PyGILState_STATE gstate;
		gstate = PyGILState_Ensure();
		PyObject_CallObject(load, NULL);
		PyGILState_Release(gstate);
		return true;
	}
	catch (std::exception& e)
	{
		std::cout << "Standard exception:" << e.what() << std::endl;
		return false;
	}
}

int callPythonFunc(const cv::Mat &img, PyObject *pFunc)
{
	if (!img.data)
	{
		return -1;
	}
	if (!pFunc)
	{
		return -1;
	}
	cv::Mat tmp = img.clone();
	int rows = img.rows;
	int cols = img.cols;
	int channels = img.channels();
	int res = -1;      // 表示预测结果的返回值，-1表示预测失败
	PyObject *pArgs = PyTuple_New(4);
	PyObject *pList = PyList_New(rows * cols * channels);
	PyObject *pReturn = NULL;
	//std::cout << rows << std::endl;
	for (int row = 0; row < rows; ++row)
	{
		cv::Vec3b *p = tmp.ptr<cv::Vec3b>(row);
		for (int col = 0; col < cols; ++col)
		{
			PyList_SetItem(pList, row * cols * 3 + col * 3 + 0, Py_BuildValue("i", p[col][0]));
			PyList_SetItem(pList, row * cols * 3 + col * 3 + 1, Py_BuildValue("i", p[col][1]));
			PyList_SetItem(pList, row * cols * 3 + col * 3 + 2, Py_BuildValue("i", p[col][2]));
		}
	}

	PyTuple_SetItem(pArgs, 0, pList);
	PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", cols));
	PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", rows));
	PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", channels));

	PyGILState_STATE gstate;
	gstate = PyGILState_Ensure();
	pReturn = PyObject_CallObject(pFunc, pArgs);
	PyGILState_Release(gstate);
	if (!pReturn)
	{
		return -1;
	}
	PyArg_Parse(pReturn, "i", &res);
	return res;
}