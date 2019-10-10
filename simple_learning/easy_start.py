#!/usr/bin/env python
# -*- coding: gb18030 -*-
import os
import sys
import datetime
import argparse
import nn_tools
reload(sys) #Python2.5 ��ʼ�����ɾ�� sys.setdefaultencoding ���������������Ҫ��������
sys.setdefaultencoding('gb18030')
import tensorflow as tf
import numpy as np

# *********************************************************************************************
#�򵥵�������������Իع�
class Regression(object):
    def __init__(self):
        self.tools = nn_tools.NN_simple_tool()
        return
    def create_sample(self):
        X = np.linspace(-1, 1, 300)
        X = X.reshape(-1, 1).astype('float32')
        noise = np.random.normal(0, 0.05, X.shape).astype('float32')
        y = np.square(X) - 0.5 + noise
        return X, y
    def create_model(self, X, y):
        con1 = self.tools.add_connect(X, 1, 10, tf.nn.tanh) # ���� relu �� sigmoid ���� 
        predict_y = self.tools.add_connect(con1, 10, 1)
        loss = tf.reduce_mean(tf.square(y - predict_y)) #��С����
        optimizer = tf.train.AdamOptimizer(0.05) #�Ż��㷨
        train = optimizer.minimize(loss)#�Ż�Ŀ��
        init = tf.global_variables_initializer()#��ʼ������ṹ����
        return init, loss
    def train(self, init, loss):
        ckpt = tf.train.get_checkpoint_state('./tmp/z_model.ckpt')
        with tf.Session() as sess:
            sess.run(init)
            for i in range(300):
                writer = tf.summary.FileWriter('graphs', sess.graph)
                sess.run(init)
                if i % 10 == 0:
                    print i, 'loss:', sess.run(loss)
            out = sess.run(predict_y)
        return 

def main_run():
    reg = Regression()
    X, y = reg.create_sample()
    init, loss = reg.create_model(X, y)
    reg.train(init, loss)
    return 0

if __name__ == '__main__':
    start = datetime.datetime.now()
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--option", required = True, help="readme txt")
    #args = parser.parse_args()
    #main_run(args)
    main_run()
    end = datetime.datetime.now()
    # print sys.argv[0]+' this program has finnished....takes %d'%((end - start).total_seconds())

