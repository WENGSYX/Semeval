
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
from transformers import AutoModelForSequenceClassification,AutoTokenizer,AdamW,get_cosine_schedule_with_warmup,AutoModelForSequenceClassification
#from transformers.deepspeed import HfDeepSpeedConfig
from torch.autograd import Variable
from accelerate import Accelerator
accelerator = Accelerator()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1)
parser.add_argument("--deepspeed_config",default=-1)
args = parser.parse_args()

CFG = { #训练的参数配置
    'fold_num': 5, #五折交叉验证
    'seed': 2002,
    'model': 'deberta_v3_large', #预训练模型
    'max_len': 112, #文本截断的最大长度
    'epochs': 16,
    'train_bs': 10, #batch_size，可根据自己的显存调整
    'valid_bs': 10,
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



train = pd.read_pickle('Entrain.pk')
valid = pd.read_pickle('Endev.pk')
tokenizer = AutoTokenizer.from_pretrained(CFG['model'])


class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        data = self.df[idx]

        return data


def collate_fn(data):
    input_ids, attention_mask, token_type_ids = [], [], []
    for x in data:
        text = tokenizer(str(x['sen']), padding='max_length',truncation=True, max_length=CFG['max_len'])
        input_ids.append(text['input_ids'])
        attention_mask.append(text['attention_mask'])
        token_type_ids.append(text['token_type_ids'])

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    label = torch.tensor([x['label'] for x in data])
    return input_ids, attention_mask, token_type_ids,label

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
def train_model(model, fgm,train_loader,epoch):  # 训练一个epoch
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()
    f1s  = AverageMeter()
    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
    for step, (input_ids, attention_mask, token_type_ids,y) in enumerate(tk):

        output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        loss = criterion(output,y)
        scaler.scale(loss).backward()
        fgm.attack()  # 在embedding上添加对抗扰动
        output2 = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        loss2 = criterion(output2,y)
        scaler.scale(loss2).backward()
        fgm.restore()
        if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        losses.update(loss.item(), input_ids.size(0))
        acc = (output.argmax(1) == y).sum().item() / y.size(0)
        accs.update(acc,y.size(0))
        tk.set_postfix(loss=losses.avg,acc=accs.avg)
        if step == 0:
            log(['开始新一轮训练','现轮数'.format(epoch),'训练起始损失：{}'.format(str(loss.item())),'总batch数：{}'.format(len(tk))],path)

    log(['训练最终损失：{}'.format(str(loss.item())),'训练平均损失：{}'.format(losses.avg),'训练平均准确值：{}'.format(accs.avg),'结束本轮训练'],path)
    return losses.avg


from sklearn.metrics import f1_score
def test_model(model, val_loader,epoch):  # 验证
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()
    task2_acces = AverageMeter()
    y_truth, y_pred = [], []
    pred = []
    gold = []

    log(['开始测试','现轮数'.format(epoch)],path)
    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
            p,g = [],[]
            output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
            for o in range(output.size(0)):
                if o % 2 == 0:
                    p.append(output[o,1])
                else:
                    g.append(output[o,1])
            pred.extend(output.argmax(1).cpu().tolist())
            gold.extend(y.cpu().tolist())
            loss = criterion(output,y)
            acc = (output.argmax(1) == y).sum().item() / y.size(0)
            task2_acc = sum(torch.stack(p)>torch.stack(g))/len(g)
            losses.update(loss.item(), y.size(0))
            accs.update(acc, y.size(0))
            task2_acces.update(task2_acc.item())
            tk.set_postfix(loss=losses.avg, acc=accs.avg,task2_acc=task2_acces.avg)

    score = f1_score(pred,gold)
    log(['测试损失：{}'.format(str(losses.avg)),'测试准确率：{}'.format(accs.avg),'任务二准确率：{}'.format(task2_acces.avg),'结束本轮测试','F1测试分数:{}'.format(score)],path)
    return losses.avg, accs.avg,score




def log(text,path):
    with open(path+'/log.txt','a',encoding='utf-8') as f:
        f.write('-----------------{}-----------------'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        for i in text:
            f.write(i)
            print(i)
            f.write('\n')
        f.write('\n')
import os
def log_start(log_name):
    if log_name == '':
        log_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    os.mkdir('log/' + log_name)

    with open('log/' + log_name + '/python_file.py', 'a', encoding='utf-8') as f:
        with open(os.path.basename(__file__),'r',encoding='utf-8') as f2:
            file = f2.read()
        f.write(file)

    path = 'log/' + log_name
    with open(path+'/log.txt', 'a', encoding='utf-8') as f:
        f.write('log_name')
        f.write('\n')
    return path
def macro_f1(pred,gold):
    intersection = 0
    GOLD = 0
    PRED = 0
    intersection1 = 0
    GOLD1 = 0
    PRED1 = 0
    F1s = []
    f = []
    for l in range(2):
        l_p = []
        l_g = []
        for i in range(len(pred)):
            p = pred[i]
            g = gold[i]

            if g == l:
                l_g.append(i)
            if p == l:
                l_p.append(i)

        l_g = set(l_g)
        l_p = set(l_p)

        TP = len(l_g & l_p)
        FP = len(l_p) - TP
        FN = len(l_g) - TP
        precision = TP/(TP+FP+ 0.0000000001)
        recall    = TP/(TP+FN+ 0.0000000001)
        F1        = (2*precision*recall)/(precision+recall+0.0000000001)
        F1s.append(F1)
    return F1s

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name1='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name1 in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)


    def restore(self, emb_name1='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name1 in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
train_set = MyDataset(train)
valid_set = MyDataset(valid)
train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=False,
                          num_workers=CFG['num_workers'])
valid_loader = DataLoader(valid_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False,
                          num_workers=CFG['num_workers'])
best_acc = 0
#import deepspeed
#ds_config = 'ds_config.json'
from transformers import DebertaV2ForSequenceClassification
model = DebertaV2ForSequenceClassification.from_pretrained(CFG['model'],num_labels = 2)  # 模型


model = model.to(device)
scaler = GradScaler()
#from transformers import Adafactor
#optimizer = Adafactor(model.parameters(),relative_step=False, lr=CFG['lr'], weight_decay=CFG['weight_decay'])  # AdamW优化器
optimizer = AdamW(model.parameters(),lr=CFG['lr'], weight_decay=CFG['weight_decay'])
criterion = nn.CrossEntropyLoss()
#criterion = FocalLoss()
scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // CFG['accum_iter'],
                                            CFG['epochs'] * len(train_loader) // CFG['accum_iter'])
# get_cosine_schedule_with_warmup策略，学习率先warmup一个epoch，然后cos式下降
fgm = FGM(model)
#pgd = PGD(model)

train_loader,val_loader = accelerator.prepare(train_loader,valid_loader)

log_name = 'baseline'
path = log_start(log_name)

for epoch in range(CFG['epochs']):

    train_loss = train_model(model, fgm,train_loader,epoch)
    val_loss, val_acc, score = test_model(model, val_loader,epoch)
    path2 = path+'/{}_{}_model.pt'.format(epoch,score)
    os.mkdir(path2)
    model.save_pretrained(path2)
    tokenizer.save_pretrained(path2)

