from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import datetime
import argparse
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.nn as nn
import torch
import tqdm
import matplotlib.pyplot as plt
from torchviz import make_dot
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#文本预处理，用transformer的embeding
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import DebertaForSequenceClassification, DebertaTokenizer
from datasets import load_dataset
data = load_dataset('clue', 'tnews')
#tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
tokenizer = AutoTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese', use_fast=False)

x_train = data['train']['sentence']
x_test = data['test']['sentence']
y_train = data['train']['label']
y_test = data['test']['label']
#x_train = x_train.reshape(-1, 1)
#x_test = x_test.reshape(-1, 1)
x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.2)
train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=128) 
train_labels = torch.tensor(y_train) 
dev_encoding = tokenizer(x_dev, truncation=True, padding=True, max_length=128) 
dev_labels = torch.tensor(y_dev)

class QueryDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    #系统迭代器,这里idx是为了让后面的Dataloader成批处理成迭代器，按idx映射成对应数据
    def __getitem__(self, idx):
        #item = {key: torch.tensor(val[idx]) for key, val in tqdm.tqdm(self.encodings.items())}
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item
    def __len__(self):
        return len(self.labels)
train_dataset = QueryDataset(train_encoding, y_train)
dev_dataset = QueryDataset(dev_encoding, y_dev)
#train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=128)
#dev_encoding = tokenizer(x_dev, truncation=True, padding=True, max_length=128)
#train_dataset = torch.utils.data.TensorDataset(train_encoding['input_ids'], train_encoding['attention_mask'], train_labels)
#dev_dataset = torch.utils.data.TensorDataset(dev_encoding['input_ids'], dev_encoding['attention_mask'], test_labels)
#train_dataset = QueryDataset(train_encoding, y_train)
#dev_dataset = QueryDataset(dev_encoding, y_dev)
# 单个读取到批量读取
batch_size = 16
class_num = 15
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=128, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=128, shuffle=True)


class my_bert_model(nn.Module):
    def __init__(self, freeze_bert=False, hidden_size=768, max_position_embeddings=1024, base_path='./model/', save_model_name='tc_model.pt'):
        super().__init__()
        self.pre_trained_model = 'IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese'
        self.bert = AutoModelForSequenceClassification.from_pretrained('./mlm_new_save_model/', num_labels=15)
        #self.bert =  DebertaForSequenceClassification.from_pretrained('./mlm_new_save_model/', num_labels=15)
        #config = BertConfig.from_pretrained(self.pre_trained_model)
        #config.update({'output_hidden_states':True})
        #self.bert = BertModel.from_pretrained(self.pre_trained_model, config=config)
        #print("{} path is {}" % (self.pre_model_name, transformers.torch_cache_home))
        #self.fc = nn.Linear(hidden_size * 8, 15)
        #self.fc = nn.Linear(16, 15)
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            os.system("mkdir -p {}".format(self.base_path))
        self.save_model_name = save_model_name
        self.save_checkpoint_steps = 50
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        #print("outputs", outputs[0].shape)
        #result = self.fc(outputs[0])
        return outputs[0]
        #return result
    def _save(self, epoch, step, optim):
        checkpoint = {
            'model': self.state_dict(),
            'epoch': epoch,
            'step': step,
            'optim': optim,
        }
        checkpoint_path_newest = '%s' % (self.base_path + self.save_model_name)
        torch.save(checkpoint, checkpoint_path_newest)
    def _load(self):
        save_model_path = self.base_path + self.save_model_name
        if os.path.exists(save_model_path):
            print(save_model_path)
            checkpoint = torch.load(save_model_path)
            self.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']
            optim = checkpoint['optim']
            print('加载 epoch {} step {} 成功！'.format(start_epoch, start_step))
            return start_epoch, start_step, optim

        else:
            start_epoch = 0
            start_step = 0
            optim = None
            print('无保存模型，将从头开始训练！')
            return start_epoch, start_step, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device='cpu'
print(device, "is used")
model = my_bert_model(freeze_bert=False).to(device)
start_epoch = 0
start_step = 0
start_epoch, start_step, optim = model._load()
if optim == None:
    optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

criterion = nn.CrossEntropyLoss().to(device)
total_steps = len(train_loader) * 1
scheduler = get_linear_schedule_with_warmup(optim,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

def train(epoch, start_step):
    print(epoch, start_step)
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for batch in tqdm.tqdm(train_loader):
        #当前iter_num下从断点开始训练,后面的epoch start_step会失效
        if start_step > 0:
            if iter_num < start_step:
                iter_num += 1
                continue
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        pre_labels = model(input_ids, attention_mask=attention_mask)
        #变成onehot
        b_y = labels.reshape(-1, 1).to(device)
        #b_y = torch.zeros(batch_size, class_num).scatter_(1, b_y, 1).to(device)
        b_y = torch.zeros(batch_size, class_num).to(device).scatter_(1, b_y, 1)
        #print(pre_labels, b_y)
        loss = criterion(pre_labels, labels)
        total_train_loss += loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step()
        iter_num += 1
        if iter_num % 100 == 0:
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (epoch, iter_num, loss.item(), iter_num/total_iter * 100))
            acc_arr = []
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                with torch.no_grad():
                    pre_labels = model(input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(pre_labels,dim=1)
                    accuracy = torch.mean(torch.eq(preds, labels).to(device).float())
                    acc_arr.append(accuracy)
                break
            acc_p = sum(acc_arr) / float(len(acc_arr))
            #history1.log(iter_num, train_loss=loss, test_accuracy=acc_p)
            #with canvas1:
            #    canvas1.draw_plot(history1['train_loss'])
            #    canvas1.draw_plot(history1['test_accuracy'])
            print("avg_acc = %.2f" % (acc_p))
            model._save(epoch, iter_num, optim)
for epoch in range(start_epoch, 15):
    print("------------Epoch: %d ----------------" % epoch)
    train(epoch, start_step)
    if start_step > 0:
        #热启动的start step只生效一次
        start_step = 0
    os.system('cp ./model/tc_model.pt ./save_model/')
"""
# 加载预训练模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese', use_fast=False)
#model = AutoModelForMaskedLM.from_pretrained(ori_model)
ori_model = './output/checkpoint-65000/'
model = AutoModelForSequenceClassification.from_pretrained(ori_model, num_labels=2)

# 输入文本
text = "我再北京天安门"

# 对文本进行tokenize和padding
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# 使用模型进行推理
outputs = model(**inputs)

# 获取预测结果
predictions = outputs.logits.argmax(dim=1)

# 打印结果
if predictions == 1:
    print("Positive sentiment detected.")
else:
    print("Negative sentiment detected.")
"""
