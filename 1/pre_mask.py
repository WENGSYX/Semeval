import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
import time
from sklearn.model_selection import *
from transformers import AutoModelForMaskedLM,BertTokenizer,AdamW,get_cosine_schedule_with_warmup,AutoTokenizer,DebertaV2ForMaskedLM
from transformers.deepspeed import HfDeepSpeedConfig
from torch.autograd import Variable
from accelerate import Accelerator
from functools import reduce
accelerator = Accelerator()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0)
parser.add_argument("--deepspeed_config")
args = parser.parse_args()

CFG = { #训练的参数配置
    'fold_num': 5, #五折交叉验证
    'seed': 1012,
    'model': 'mdeberta', #预训练模型
    'max_len': 32, #文本截断的最大长度
    'epochs': 16,
    'train_bs': 80, #batch_size，可根据自己的显存调整
    'valid_bs': 80,
    'lr': 1e-5, #学习率
    'num_workers': 0,
    'accum_iter': 1, #梯度累积，相当于将batch_size*2
    'weight_decay': 1e-4, #权重衰减，防止过拟合
    'device': 0,
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

seed_everything(CFG['seed']) #固定随机种子

torch.cuda.set_device(CFG['device'])
device = accelerator.device

tokenizer = AutoTokenizer.from_pretrained(CFG['model'])
import random

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

        input,label = n_gram_mask(text['input_ids'])
        input_ids.append(input)
        labels.append(label)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    return input_ids, attention_mask, token_type_ids,labels

def collate_fn2(data):
    input_ids,attention_mask, token_type_ids,labels= [],[], [], []
    for x in data:
        text = tokenizer(x, padding='max_length', truncation=True, max_length=150)
        attention_mask.append(text['attention_mask'])
        token_type_ids.append(text['token_type_ids'])

        input,label = n_gram_mask(text['input_ids'])
        input_ids.append(input)
        labels.append(label)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
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

import time
def train_model(model, fgm,pgd,train_loader):  # 训练一个epoch
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()
    f1s  = AverageMeter()
    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)

    for step, (inputs, attention_mask, token_type_ids,labels) in enumerate(tk):
        #inputs,labels = mask_tokens(inputs)
        with autocast():
            output = model(inputs, attention_mask, token_type_ids, labels=labels)
            loss = output.loss
            scaler.scale(loss).backward()
            if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
                """
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                """
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        losses.update(loss.item(), inputs.size(0))
        tk.set_postfix(loss=losses.avg)

    return losses.avg


def test_model(model, val_loader):  # 验证
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()
    y_truth, y_pred = [], []

    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
            input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
                device), token_type_ids.to(device), y.to(device).long()

            output = model(input_ids, attention_mask, token_type_ids).logits

            y_truth.extend(y.cpu().numpy())
            y_pred.extend(output.argmax(1).cpu().numpy())

            loss = criterion(output, y)

            acc = (output.argmax(1) == y).sum().item() / y.size(0)

            losses.update(loss.item(), y.size(0))
            accs.update(acc, y.size(0))

            tk.set_postfix(loss=losses.avg, acc=accs.avg)
    F1,GOLD,PRED,intersection,F11,GOLD1,PRED1,intersection1 = macro_f1(pred=y_pred,gold=y_truth)
    print('GOLD:{} {}'.format(GOLD,GOLD1))
    print('PRED:{} {}'.format(PRED,PRED1))
    print('intersection:{} {}'.format(intersection,intersection1))
    print('F1:{} {}'.format(F1,F11))
    return losses.avg, accs.avg,F11

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name1='bert.embeddings.word_embeddings.weight',emb_name2='bert.embeddings.position_embeddings.weight',emb_name3='bert.embeddings.token_type_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name1 in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
            if param.requires_grad and emb_name2 in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
            if param.requires_grad and emb_name3 in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name1='bert.embeddings.word_embeddings.weight',emb_name2='bert.embeddings.position_embeddings.weight',emb_name3='bert.embeddings.token_type_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name1 in name:
                assert name in self.backup
                param.data = self.backup[name]
            if param.requires_grad and emb_name2 in name:
                assert name in self.backup
                param.data = self.backup[name]
            if param.requires_grad and emb_name3 in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name1='bert.embeddings.word_embeddings.weight',emb_name2='bert.embeddings.position_embeddings.weight',emb_name3='bert.embeddings.token_type_embeddings.weight', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name1 in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)
            if param.requires_grad and emb_name2 in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)
            if param.requires_grad and emb_name3 in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name1='bert.embeddings.word_embeddings.weight',emb_name2='bert.embeddings.position_embeddings.weight',emb_name3='bert.embeddings.token_type_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name1 in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
            if param.requires_grad and emb_name2 in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
            if param.requires_grad and emb_name3 in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


from transformers import DebertaV2Config
config = DebertaV2Config.from_pretrained(CFG['model'])
model = DebertaV2ForMaskedLM._from_config(config)

model = model.to(device)
scaler = GradScaler()
optimizer = AdamW(model.parameters(),lr=CFG['lr'], weight_decay=CFG['weight_decay'])

criterion = nn.CrossEntropyLoss()
train = pd.read_pickle('datas1.pk')
train_set = MyDataset(train)
train_loader = DataLoader(train_set, batch_size=80, collate_fn=collate_fn, shuffle=True,
                          num_workers=CFG['num_workers'])
scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // CFG['accum_iter'],
                                            CFG['epochs'] * len(train_loader) // CFG['accum_iter'])
# get_cosine_schedule_with_warmup策略，学习率先warmup一个epoch，然后cos式下降
fgm = 'FGM(model)'
pgd = 'PGD(model)'




for epoch in range(CFG['epochs']):
    print('epoch:', epoch)
    time.sleep(0.2)
    train = pd.read_pickle('datas1.pk')
    train_set = MyDataset(train)
    train_loader = DataLoader(train_set, batch_size=70, collate_fn=collate_fn, shuffle=True,
                              num_workers=CFG['num_workers'])
    train_loader = accelerator.prepare(train_loader)
    train_loss1 = train_model(model, fgm, pgd, train_loader)

    train = pd.read_pickle('datas2.pk')
    train_set = MyDataset(train)
    train_loader = DataLoader(train_set, batch_size=18, collate_fn=collate_fn2, shuffle=True,
                              num_workers=CFG['num_workers'])
    train_loader = accelerator.prepare(train_loader)
    train_loss2 = train_model(model, fgm, pgd, train_loader)

    path = 'new/save/{}_deberta_{}_{}.pt'.format(str(epoch), str(train_loss1),str(train_loss2))
    os.mkdir(path)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
