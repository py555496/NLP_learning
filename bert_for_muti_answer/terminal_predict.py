# encoding=utf-8

"""
基于命令行的在线预测方法
@Author: Macan (ma_cancan@163.com) 
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow
import numpy as np
import codecs
import pickle
import os
import ujson as json
import datetime

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer,encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer,encoding='utf-8')

import BERT_NER
from bert import tokenization, modeling
from bert_base.train.train_helper import get_args_parser
"""
from bert_base.train.models import create_model, InputFeatures
from bert_base.bert import tokenization, modeling
from bert_base.train.train_helper import get_args_parser
"""
args = get_args_parser()
args.max_seq_length = 256
args.do_lower_case = True
args.save_checkpoints_steps = 3000
BERT_NER.FLAGS.max_seq_length = 256

#model_dir = r'/home/work/zhoujing07/tag_aggr/dev/dev_workplace/bert_zh_ner_v2_model/output/result_dir/'
model_dir = r'/home/work/zhoujing/url_intent/tag_aggr/dqa_tag_aggr_web_server/crf_bert_model_server/output/result_dir/'
bert_dir = './cased_L-12_H-768_A-12'

is_training=False
use_one_hot_embeddings=False
batch_size=1

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess=tf.Session(config=gpu_config)
model=None

global graph
input_ids_p, mask, label_ids_p, segment_ids_p = None, None, None, None


print('checkpoint path:{}'.format(os.path.join(model_dir, "checkpoint")))
if not os.path.exists(os.path.join(model_dir, "checkpoint")):
    raise Exception("failed to get checkpoint. going to return ")

# 加载label->id的词典
with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}

with codecs.open(os.path.join(model_dir, 'label_list.pkl'), 'rb') as rf:
    label_list = pickle.load(rf)
num_labels = len(label_list)

graph = tf.get_default_graph()
tf.reset_default_graph()
with graph.as_default():
    print("going to restore checkpoint")
    #sess.run(tf.global_variables_initializer())
    input_ids_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_ids")
    mask = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="mask")
    segment_ids_p = tf.placeholder(tf.int64, [batch_size, args.max_seq_length], name="segment_ids")
    label_ids_p = tf.placeholder(tf.int64, [batch_size, args.max_seq_length], name="label_ids")

    bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))
    #(total_loss, logits, trans, pred_ids) = BERT_NER.create_model(
    #(total_loss, logits, pred_ids, viterbi_score) = BERT_NER.create_model(
    result_dict = BERT_NER.create_model(
        bert_config=bert_config, is_training=False, input_ids=input_ids_p, mask=mask, segment_ids=segment_ids_p,
        labels=label_ids_p, num_labels=num_labels, use_one_hot_embeddings=False, dropout_rate=1.0)
    pred_ids = result_dict["predicts"]

    #ckpt_path = "output/result_dir/model.ckpt-20607"
    #reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    #var_to_shape_map=reader.get_variable_to_shape_map()
    #var_to_shape_map['crf_loss/transitions'] = var_to_shape_map['crf_loss/transition']
    #variables_to_restore = slim.get_variables_to_restore(include=var_to_shape_map.keys())

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=args.do_lower_case)


def predict_online():
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    def convert(lines):
        data_process = BERT_NER.NerProcessor()
        predict_examples = data_process._create_example(lines, 'test')
        feature, _, _ = BERT_NER.convert_single_example(0, predict_examples[0], label_list, args.max_seq_length, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids],(batch_size, args.max_seq_length))
        input_mask = np.reshape([feature.mask],(batch_size, args.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids],(batch_size, args.max_seq_length))
        label_ids =np.reshape([feature.label_ids],(batch_size, args.max_seq_length))
        return input_ids, input_mask, segment_ids, label_ids

    global graph
    with graph.as_default():
        #print(id2label)
        ##start server
        import socket
        s = socket.socket()         # 创建 socket 对象
        host = "10.99.52.22" # 获取本地主机名
        port = 8817                # 设置端口
        s.bind((host, port))        # 绑定端口 8817
        s.listen(5)                 # 等待客户端连接
        print("server started.....")
        while True:
            #print('input the test sentence:')
            c,addr = s.accept()     # 建立客户端连接
            contents = ""
            start = datetime.datetime.now()
            data_len = 0
            first = True
            while True:
                if first:
                    first = False
                    contents = c.recv(64).decode()
                    if len(contents) > 12:
                        data_len = contents.split("\1")[0]
                        contents = contents.split("\1")[1]
                    if len(data_len) == 0:
                        break
                    continue
                contents += c.recv(4096 * 4).decode()

                print(len(contents), int(data_len))
                if len(contents) >= int(data_len):
                    break
                end = datetime.datetime.now()
                delta_time = end - start
                if delta_time.seconds > 14:
                    break
            conts = json.loads(contents)
            start = datetime.datetime.now()
            predict_out_str_arr = []
            for content in conts:
                lines = []
                words = []
                labels = []
                content = content.strip()
                content = content.replace(" ", "ɔ")
                content = content.replace("\u3000", "ɔ")
                if len(content) < 2:
                    print(content)
                    continue
                l = " ".join(['O'] * len(list(content)))
                w = " ".join(list(content))
                words = list(content)
                lines.append([l, w])
                # print('your input is:{}'.format(content))
                data_process = BERT_NER.NerProcessor()
                predict_examples = data_process._create_example(lines, 'test')

                input_ids, input_mask, segment_ids, label_ids = convert(lines)

                feed_dict = {input_ids_p: input_ids,
                             mask: input_mask,
                             segment_ids_p: segment_ids,
                             label_ids_p : label_ids}
                # run session t current feed_dict result
                feature = graph.get_operation_by_name("crf_loss/transition").outputs[0]
                #pred_ids_result, batch_feature = sess.run([pred_ids, feature], feed_dict)
                pred_ids_result = sess.run([pred_ids], feed_dict)
                #pred_ids_result = sess.run([pred_ids], feed_dict)
                #print("feature:", batch_feature)
                print("pred_ids_result:", pred_ids_result)
                pred_label_result = convert_id_to_label(pred_ids_result[0], id2label)
                print(list(content))
                print(len(words), len(pred_ids_result[0]), len(pred_label_result[0]))
                predict_out_str_arr.append([list(content), pred_label_result])
                #string = json.dumps([list(content), pred_label_result, logits_out[0].tolist(), layer_info])
                #print("zhoujing", len(string))
                #predict_out_str_arr.append(string)
            send_back_data = json.dumps(predict_out_str_arr)
            send_back_str = "{}\1{}".format(len(send_back_data), send_back_data)
            #c.sendall(str(len(send_back_str.encode())).encode())
            c.sendall(send_back_str.encode())
            #todo: 组合策略
            #result = strage_combined_link_org_loc(content, pred_label_result[0])
            print('time used: {} sec'.format((datetime.datetime.now() - start).total_seconds()))

def convert_id_to_label(pred_ids_result, idx2label):
    """
    将id形式的结果转化为真实序列结果
    :param pred_ids_result:
    :param idx2label:
    :return:
    """
    result = []
    for row in range(batch_size):
        curr_seq = []
        #for ids in pred_ids_result[row][0]:
        for ids in pred_ids_result[row][0]:
            if ids == 0:
                break
            curr_label = idx2label[ids]
            #if curr_label in ['[CLS]', '[SEP]']:
            if curr_label in ['[CLS]']:
                continue
            curr_seq.append(curr_label)
        result.append(curr_seq)
    return result

if __name__ == "__main__":
    predict_online()

