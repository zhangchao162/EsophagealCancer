#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/11 16:58
# @Author  : zhangchao
# @File    : time_helper.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import time


def get_running_time(func):
    def func_time(*args, **kwargs):
        t0 = time.time()
        print(f"{get_format_time()} Method: '{func.__name__}' Running...")
        res = func(*args, **kwargs)
        t1 = time.time()
        print(f"  Running time: {((t1 - t0) // 60)} min {((t1 - t0) % 60):.4f} s")
        return res

    return func_time


def get_format_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
