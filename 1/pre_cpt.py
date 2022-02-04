import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import os
import argparse
import time
from sklearn.model_selection import *
from transformers import *
from torch.autograd import Variable
import pkuseg
from transformers import DebertaV2Tokenizer
from model import CPTForConditionalGeneration
from accelerate import Accelerator
# from evaluate_new import *
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import BertTokenizer, BartForConditionalGeneration
accelerator = Accelerator()
device = accelerator.device
CFG = {  # 训练的参数配置
    'fold_num': 5,  # 五折交叉验证
    'seed': 2022,
    'model': r'C:\semeval\1\7_0.9817_0.9394_0.9732_1.0844_0.5315_0.2319_0.3356_0.3167_0.3302_0.3794_0.3387_0.2434_0.2756_0.2264_0.1898',  # 预训练模型
    'max_len': 100,  # 文本截断的11最大长度
    'epochs': 8,
    'train_bs': 6,  # batch_size，可根据自己的显存调整
    'valid_bs': 50,
    'lr': 1e-5,  # 学习率
    'num_workers': 0,
    'accum_iter': 8,  # 梯度累积，相当于将batch_size*2
    'weight_decay': 1e-4,  # 权重衰减，防止过拟合
    'device': 0,
    'ues_r_drop': True,
    'train':False
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG['seed'])  # 固定随机种子

torch.cuda.set_device(CFG['device'])



tokenizer = DebertaV2Tokenizer.from_pretrained(CFG['model'])


class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        sen = self.df[idx]

        return sen
def n_gram_mask(inputs):
    input_ids, output_ids = [], []
    rands = np.random.random(len(inputs))
    idx = 0
    while idx < len(inputs):
        if rands[idx] < 0.2:  # 需要mask
            ngram = np.random.choice([1, 2, 3,4], p=[0.85, 0.05, 0.05,0.05])  # 若要mask，进行x_gram mask的概率
            L = idx + 1
            R = idx + ngram  # 最终需要mask的右边界（开）
            while L < R and L < len(rands):
                rands[L] = np.random.random() * 0.1  # 强制mask
                L += 1
            idx = R
            if idx < len(inputs):
                rands[idx] = 1  # 禁止mask片段的下一个token被mask，防止一大片连续mask
        idx += 1
    for r, i in zip(rands, inputs):
        if i == 0:
            input_ids.append(i)
            output_ids.append(-100)
        else:
            if r < 0.15:
                input_ids.append(250101)
                output_ids.append(i)  # mask预测自己
            elif r < 0.16:
                input_ids.append(i)
                output_ids.append(i)  # 自己预测自己
            elif r < 0.2:
                input_ids.append(np.random.randint(0,250100))
                output_ids.append(i)  # 随机的一个词预测自己，随机词不会从特殊符号中选取，有小概率抽到自己

            else:
                input_ids.append(i)
                output_ids.append(-100)  # 保持原样不预测
    return input_ids,output_ids

def collate_fn(data):
    input_ids,attention_mask, token_type_ids,labels= [],[], [], []
    for x in data:
        text = tokenizer(x, padding='max_length', truncation=True, max_length=32)
        attention_mask.append(text['attention_mask'])
        token_type_ids.append(text['token_type_ids'])

        input, label = n_gram_mask(text['input_ids'])
        input_ids.append(input)
        text = tokenizer(x, padding='max_length', truncation=True, max_length=35, return_tensors='pt')
        label = text['input_ids']
        label = torch.where(label == tokenizer.pad_token_id, -100, label)
        labels.append(label)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    input_ids = torch.tensor(input_ids)
    labels = torch.stack(labels)
    return input_ids, attention_mask, token_type_ids,labels

def collate_fn2(data):
    input_ids,attention_mask, token_type_ids,labels= [],[], [], []
    for x in data:
        text = tokenizer(x, padding='max_length', truncation=True, max_length=150)
        attention_mask.append(text['attention_mask'])
        token_type_ids.append(text['token_type_ids'])

        input,label = n_gram_mask(text['input_ids'])
        input_ids.append(input)
        text = tokenizer(x, padding='max_length', truncation=True, max_length=200,return_tensors='pt')
        label = text['input_ids']
        label = torch.where(label==tokenizer.pad_token_id,-100,label)
        labels.append(label)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    input_ids = torch.tensor(input_ids)
    labels = torch.stack(labels)
    return input_ids, attention_mask, token_type_ids,labels
class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_model(model,train_loader):  # 训练一个epoch
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()
    f1s = AverageMeter()
    K = 3
    # optimizer.zero_grad()

    tk = tqdm(train_loader, desc='dirs')
    print(device)
    optimizer.zero_grad()

    for step, (input_ids, attention_mask, token_type_ids,labels) in enumerate(tk):

        with autocast():  # 使用半精度训练
            output = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = output.loss
            scaler.scale(loss).backward()

        if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
            scheduler.step()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        losses.update(loss.item())

        tk.set_postfix(loss=losses.avg)

    return losses.avg


train = pd.read_pickle('datas1.pk')
train_set = MyDataset(train)

model = CPTForConditionalGeneration.from_pretrained(CFG['model'] ).to(device)  # 模型
#model.load_state_dict(torch.load('4_T5_1.6154526329307242.pt'),strict=True)


scaler = GradScaler()
from transformers import Adafactor

optimizer = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])  # AdamW优化器
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True,
                          num_workers=CFG['num_workers'])
scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // CFG['accum_iter'],
                                            CFG['epochs'] * len(train_loader) // CFG['accum_iter'])
fgm = 'FGM(model)'
pgd = 'PGD(model)'



for epoch in range(CFG['epochs']):
    train = pd.read_pickle('datas1.pk')
    train_set = MyDataset(train)
    train_loader = DataLoader(train_set, batch_size=46, collate_fn=collate_fn, shuffle=True,
                              num_workers=CFG['num_workers'])
    train_loader = accelerator.prepare(train_loader)
    train_loss1 = train_model(model, train_loader)

    train = pd.read_pickle('datas2.pk')
    train_set = MyDataset(train)
    train_loader = DataLoader(train_set, batch_size=11, collate_fn=collate_fn2, shuffle=True,
                              num_workers=CFG['num_workers'])
    train_loader = accelerator.prepare(train_loader)
    train_loss2 = train_model(model,train_loader)

    path = 'new/CPTsave/{}_cpt_{}_{}'.format(str(epoch), str(train_loss1),str(train_loss2))
    os.mkdir(path)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
