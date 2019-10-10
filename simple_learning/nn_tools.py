#!/usr/bin/env python
# -*- coding: gb18030 -*-
import os
import sys
import datetime
import argparse
reload(sys) #Python2.5 初始化后会删除 sys.setdefaultencoding 这个方法，我们需要重新载入
sys.setdefaultencoding('gb18030')

# *********************************************************************************************
import tensorflow as tf
import numpy as np

class NN_simple_tool(object):
    def __init__(self):
        return
    def normalize(self, X):
        mean = np.mean(X)
        std = np.std(X)
        X = (X - mean) / std
        return X

    def add_connect(self, inputs, in_size, out_size, activation_fun=None):
        weight = tf.Variable(tf.random_normal([in_size, out_size]))
        bias = tf.Variable(tf.zeros([1,out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, weight) + bias
        if not activation_fun:
            return Wx_plus_b
        else:
            return activation_fun(Wx_plus_b)

def main_run(argv):
    """主函数"""
    return 0

if __name__ == '__main__':
    start = datetime.datetime.now()
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--option", required = True, help="readme txt")
    #args = parser.parse_args()
    #main_run(args)
    main_run(sys.argv)
    end = datetime.datetime.now()
    # print sys.argv[0]+' this program has finnished....takes %d'%((end - start).total_seconds())

