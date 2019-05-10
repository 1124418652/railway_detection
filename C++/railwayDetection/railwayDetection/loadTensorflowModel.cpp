#include "stdafx.h"
#include "loadTensorflowModel.h"

bool loadPyModule(PyObject **module, PyObject **pDict, PyObject **pFunc)
{
	try {
		Py_SetPythonHome(L"D:/software/Anoconda/envs/Python36");
		Py_Initialize();
		PyEval_InitThreads();

		if (!Py_IsInitialized())
		{
			std::cout << "Can't initialize Python!" << std::endl;
		}

		PyRun_SimpleString("import sys");
		PyRun_SimpleString("sys.path.append('./')");

		*module = PyImport_ImportModule("logistic_regression");      // ���ڲ���ģ�͵� Python �ļ�
		if (!*module)             // ���������ȷ�����ģ��
		{
			std::cout << "Can't open module!" << std::endl;
			Py_Finalize();
			return 0;
		}

		// �� module ģ���е��뺯����Ϊ recognize �ĺ���
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
		std::cout << "load model..." << std::endl;
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
		return -1;
	if (!pFunc)
		return -1;
	cv::Mat tmp = img.clone();
	int rows = img.rows;
	int cols = img.cols;
	int channels = img.channels();
	int res = -1;      // ��ʾԤ�����ķ���ֵ��-1��ʾԤ��ʧ��
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

	if (pList)
		Py_DECREF(pList);
	if (pArgs)
		Py_DECREF(pArgs);
	if (pReturn)
		Py_DECREF(pReturn);

	return res;
}

int callPythonFunc(const std::vector<cv::Mat> &obsTmpList, PyObject *pFunc,
	std::vector<int> &predictRes)
{
	if (obsTmpList.empty())
		return -1;
	if (!pFunc)
		return -1;
	if (!predictRes.empty())
		predictRes.clear();
	int numImg = (int)obsTmpList.size();
	int rows = obsTmpList[0].rows;
	int cols = obsTmpList[0].cols;
	int channels = obsTmpList[0].channels();
	int imgSize = rows * cols * channels;
	int res = -1;
	PyObject *pArgs = PyTuple_New(4);
	PyObject *pList = PyList_New(numImg * imgSize);
	PyObject *pReturn = NULL;

	for (int i = 0; i < numImg; ++i)
	{
		cv::Mat tmp = obsTmpList[i].clone();
		for (int row = 0; row < rows; ++row)
		{
			cv::Vec3b *p = tmp.ptr<cv::Vec3b>(row);
			for (int col = 0; col < cols; ++col)
			{
				PyList_SetItem(pList, i * imgSize + row * cols * 3 + col * 3 + 0, 
					Py_BuildValue("i", p[col][0]));
				PyList_SetItem(pList, i * imgSize + row * cols * 3 + col * 3 + 1, 
					Py_BuildValue("i", p[col][1]));
				PyList_SetItem(pList, i * imgSize + row * cols * 3 + col * 3 + 2, 
					Py_BuildValue("i", p[col][2]));
			}
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
	PyObject *pListItem = NULL;
	int listNum = PyList_Size(pReturn);
	for (int j = 0; j < listNum; ++j)
	{
		int tmp;
		pListItem = PyList_GetItem(pReturn, j);
		PyArg_Parse(pListItem, "i", &tmp);
		std::cout << tmp << "\t";
		predictRes.push_back(tmp);
	}
	std::cout << std::endl;
	if (pListItem) { Py_DECREF(pListItem); };
	if (pList)
		Py_DECREF(pList);
	if (pArgs)
		Py_DECREF(pArgs);
	if (pReturn)
		Py_DECREF(pReturn);
}