from transformers import RobertaTokenizer, RobertaForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import AutoModelForMaskedLM, AutoTokenizer, FillMaskPipeline

from datasets import load_dataset

from sklearn.model_selection import train_test_split

"""
data = load_dataset('clue', 'tnews')
#pre_trained_model = 'bert-base-chinese'
#tokenizer = BertTokenizer.from_pretrained(pre_trained_model)

x_train = data['train']['sentence']
x_test = data['test']['sentence']
y_train = data['train']['label']
y_test = data['test']['label']
#x_train = x_train.reshape(-1, 1)
#x_test = x_test.reshape(-1, 1)
x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.2)
sentence = x_train + x_dev
labels = y_train +y_dev
d = {}
for s, l in zip(sentence, labels):
    if not l in d:
        d[l] = []
    d[l].append(s)
corpus_sentences = []
for k, v in d.items():
    for sen in v:
        corpus_sentences.append(sen)
    corpus_sentences.append('[unk]')

with open("./my_corpus.txt", 'w', encoding='utf8') as wr:
    for g in corpus_sentences:
        wr.write(g + "\n")
exit()
"""

# 加载领域特定的语料库
corpus_file = "my_corpus.txt"

# 初始化tokenizer和模型
ori_model = 'roberta-base'
ori_model = 'IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese'
#IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese
ori_model = 'IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese'
tokenizer=AutoTokenizer.from_pretrained(ori_model, use_fast=False)
model=AutoModelForMaskedLM.from_pretrained(ori_model)
#tokenizer = RobertaTokenizer.from_pretrained(ori_model)
#model = RobertaForMaskedLM.from_pretrained(ori_model)

# 对语料库进行编码
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=corpus_file,
    block_size=128,
)

# 初始化训练参数
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    save_steps=5000,
    save_total_limit=2,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15),
    train_dataset=dataset,
)

# 进行微调
trainer.train()
