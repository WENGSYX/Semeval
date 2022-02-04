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
from transformers import AutoModelForSequenceClassification,AutoTokenizer,AdamW,get_cosine_schedule_with_warmup, RoFormerModel,RoFormerTokenizer,AutoModel
from torch.autograd import Variable
from accelerate import Accelerator
import time

accelerator = Accelerator()
import transformers
transformers.logging.set_verbosity_error()

CFG = { #训练的参数配置
    'seed': 0,#随机种子
    'model': r'C:\semeval\1\new\save\7_deberta_3.378416373266286_3.7309876189846607.pt', #预训练模型
    'max_len': 100, #文本截断的最大长度
    'epochs': 16,#训练轮数
    'train_bs': 50, #batch_size，根据显存调整，对比学习中一般越大越好
    'lr': 1e-5, #学习率
    'num_workers': 0,
    'accum_iter': 1, #梯度累积
    'weight_decay': 1e-4, #权重衰减，防止过拟合
    'device': 0,
}

# 固定随机种子
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(CFG['seed'])

#设置显卡
torch.cuda.set_device(CFG['device'])
device = accelerator.device

#读取数据
train =  pd.read_pickle('datas1.pk')
train.extend(pd.read_pickle('datas2.pk'))
random.shuffle(train)
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
        text = tokenizer(x, padding='max_length', truncation=True, max_length=CFG['max_len'])
        input_ids.append(text['input_ids'])
        attention_mask.append(text['attention_mask'])
        text = tokenizer(x, padding='max_length', truncation=True, max_length=CFG['max_len'])
        input_ids.append(text['input_ids'])
        attention_mask.append(text['attention_mask'])

    input_ids = torch.tensor(input_ids,device=device)
    attention_mask = torch.tensor(attention_mask,device=device)
    return input_ids, attention_mask

# 为了tqdm实时显示loss和acc
class AverageMeter:
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


#此处计算对比学习损失，直接看代码有点难理解，但可配合PPT理解。本质上，是将A1与A2拉近，A1与B1/B2/C1/C2等拉远；B1与B2拉近，B1与A1/A2/C1/C2拉远；C1与C2拉近，C1与A1/A2/B1/B2拉远.t是一个超参数，建议0.05
def compute_loss(y_pred, t=0.05, device="cuda"):
    idxs = torch.arange(0, y_pred.shape[0], device=device)
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12
    similarities = similarities / t
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)

#训练模型
def train_model(model,train_loader):  # 训练一个epoch
    model.train()
    losses = AverageMeter()
    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
    for step, (input_ids, attention_mask) in enumerate(tk):
        #开启pf16训练

        #取bert的最后一层输出值，计算损失
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = compute_loss(output)
        loss.backward()

        if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
        #显示损失
        losses.update(loss.item(), input_ids.size(0))
        tk.set_postfix(loss=losses.avg)


    return losses.avg


#处理数据
train_set = MyDataset(train)
train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True,
                          num_workers=CFG['num_workers'])

#加载模型与tokenizer
from model import SimDeberta
model = SimDeberta.from_pretrained(CFG['model'])

model = model.to(device)
scaler = GradScaler()

optimizer = AdamW(model.parameters(),lr=CFG['lr'], weight_decay=CFG['weight_decay'])
criterion = nn.CrossEntropyLoss()

# get_cosine_schedule_with_warmup策略，学习率先warmup一个epoch，然后cos式下降
scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // CFG['accum_iter'],
                                            CFG['epochs'] * len(train_loader) // CFG['accum_iter'])

train_loader = accelerator.prepare(train_loader)


time.sleep(0.2)
#训练与测试
for i in range(CFG['epochs']):
    loss = train_model(model,  train_loader)
    path = 'new/save_sim/{}_{}'.format(str(i), str(loss))
    os.mkdir(path)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)