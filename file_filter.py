# -*- coding: utf-8 -*-
"""
# project: File Filter
# author:  xhj
# email:   1124418652@qq.com
# date:    2018 10/9
"""

import os
import sys
import re

__all__ = ["isCodeFile", "findFile", "write_to_file", "modify_path"]

path = os.path.abspath(".")
ignorePath = os.path.join(path, ".gitignore")

def isCodeFile(filename):
	if os.path.splitext(filename)[1].lower() in [".py", ".h", ".cpp", ".c", ".m", ".txt"]:
		return True

	return False

# def findFile(path, file_list):
# 	tmp_path = path

# 	if os.path.isfile(tmp_path):
# 		file_list.append(tmp_path)

# 	elif os.path.isdir(tmp_path):
# 		for name in os.listdir(tmp_path):
# 			tmp_path = os.path.join(path, name)
# 			findFile(tmp_path, file_list)
# 	return file_list

def findFile(path):
	tmp_path = path
	file_list = []

	if os.path.isfile(tmp_path) and not isCodeFile(tmp_path):   # 非代码文件的文件路径
		file_list.append(tmp_path)

	elif os.path.isdir(tmp_path):           # 该路劲表示的是文件夹
		for name in os.listdir(tmp_path):
			tmp_path = os.path.join(path, name)
			file_list.extend(findFile(tmp_path))

	return file_list

def write_to_file(path, file_list):
	with open(path, "w") as fw:
		for line in file_list:
			fw.write(line + "\n")

def modify_path(cur_path, file_list):
	res_list = []

	for file in file_list:
		res_list.append("/".join(file[len(cur_path) + 1:].split("\\")))   # windows 系统下路径分割符切换

	return res_list

def main():
	ignore_file = findFile(path)
	# write_to_file(ignorePath, ignore_file)
	file_list = modify_path(path, ignore_file)
	write_to_file(ignorePath, file_list)

if __name__ == '__main__':
	main()
