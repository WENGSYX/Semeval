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
from transformers import AutoModelForSequenceClassification,AutoTokenizer,AdamW,get_cosine_schedule_with_warmup, RoFormerModel,RoFormerTokenizer,AutoModel
from torch.autograd import Variable
from accelerate import Accelerator
import time

accelerator = Accelerator()
import transformers
transformers.logging.set_verbosity_error()

CFG = { #训练的参数配置
    'seed': 0,#随机种子
    'model': r'C:\semeval\1\new\save_sim\0_0.14047589513887224', #预训练模型
    'max_len': 110, #文本截断的最大长度
    'epochs': 16,#训练轮数
    'train_bs': 88, #batch_size，根据显存调整，对比学习中一般越大越好
    'lr': 1e-5, #学习率
    'num_workers': 0,
    'accum_iter': 1, #梯度累积
    'weight_decay': 1e-4, #权重衰减，防止过拟合
    'device': 0,
    'vectors':'sgns'
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
def get_data(filename):
    with open(filename,'r',encoding='utf-8') as f:
        data = json.load(f)
    return data
def get_refs(data):
    refs = []
    for i in data:
        refs.append(i[CFG['vectors']])
    return refs
train = get_data('train/en.train.json')
train.extend(get_data('train/es.train.json'))
train.extend(get_data('train/fr.train.json'))
train.extend(get_data('train/it.train.json'))
train.extend(get_data('train/ru.train.json'))

dev_en = get_data('train/en.dev.json')
dev_es = get_data('train/es.dev.json')
dev_fr = get_data('train/fr.dev.json')
dev_it = get_data('train/it.dev.json')
dev_ru = get_data('train/ru.dev.json')

dev_en_refs = get_refs(dev_en)
dev_es_refs = get_refs(dev_es)
dev_fr_refs = get_refs(dev_fr)
dev_it_refs = get_refs(dev_it)
dev_ru_refs = get_refs(dev_ru)

tokenizer = AutoTokenizer.from_pretrained(CFG['model'])


class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):
        gloss = self.df[idx]['gloss']
        vector = self.df[idx][CFG['vectors']]
        return gloss,vector


def collate_fn(data):
    input_ids, attention_mask, labels = [], [], []
    for x in data:
        text = tokenizer(x[0], padding='max_length', truncation=True, max_length=CFG['max_len'])
        input_ids.append(text['input_ids'])
        attention_mask.append(text['attention_mask'])
        labels.append(torch.tensor(x[1]))
    labels = torch.stack(labels)
    input_ids = torch.tensor(input_ids,device=device)
    attention_mask = torch.tensor(attention_mask,device=device)
    return input_ids, attention_mask,labels

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
    torch.autograd.set_grad_enabled(True)
    model.train()
    losses = AverageMeter()
    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
    for step, (input_ids, attention_mask,labels) in enumerate(tk):
        #开启pf16训练

        #取bert的最后一层输出值，计算损失
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(output,labels)
        loss.backward()

        if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
        #显示损失
        losses.update(loss.item())
        tk.set_postfix(loss=losses.avg)
        if step == 0:
            log(['开始新一轮训练','现轮数:{}'.format(epoch),'训练起始损失：{}'.format(str(loss.item())),'总batch数：{}'.format(len(tk))],path)

    log(['训练最终损失：{}'.format(str(loss.item())),'训练平均损失：{}'.format(losses.avg),'结束本轮训练'],path)
    return losses.avg
def test_model(model,val_loader,name,refs):  # 训练一个epoch
    model.eval()
    losses = AverageMeter()
    optimizer.zero_grad()
    pred = []
    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for step, (input_ids, attention_mask,labels) in enumerate(tk):

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(output,labels)

            losses.update(loss.item())
            tk.set_postfix(loss=losses.avg)
            pred.extend(output.tolist())
    mse,cos,rnk = score(pred,refs)
    log(['{}'.format(name),'测试最终损失：{}'.format(str(loss.item())),'测试平均损失：{}'.format(losses.avg),'测试MSE：{}'.format(mse),'测试COS:{}'.format(cos),'测试RNK:{}'.format(rnk),'结束本轮测试'],path)
    return round(mse,4),round(cos,4),round(rnk,4)
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

    try:
        os.mkdir('new/log/' + log_name)
    except:
        log_name = log_name + time.strftime("_%Y_%m_%d_%H_%M_%S", time.localtime())
        os.mkdir('new/log/' + log_name)

    with open('new/log/' + log_name + '/python_file.py', 'a', encoding='utf-8') as f:
        with open(os.path.basename(__file__),'r',encoding='utf-8') as f2:
            file = f2.read()
        f.write(file)

    path = 'new/log/' + log_name
    with open(path+'/log.txt', 'a', encoding='utf-8') as f:
        f.write('log_name')
        f.write('\n')
    return path
import collections
import torch.nn.functional as F
def rank_cosine(preds, targets):
    assocs = preds @ F.normalize(targets).T
    refs = torch.diagonal(assocs, 0).unsqueeze(1)
    ranks = (assocs >= refs).sum(1).float().mean().item()
    return ranks / preds.size(0)

def score(submission,reference):
    vec_archs = [CFG['vectors']]
    all_preds = collections.defaultdict(list)
    all_refs = collections.defaultdict(list)

    all_preds[vec_archs[0]] = submission
    all_refs[vec_archs[0]] = reference

    torch.autograd.set_grad_enabled(False)
    all_preds = {arch: torch.tensor(all_preds[arch]) for arch in vec_archs}
    all_refs = {arch: torch.tensor(all_refs[arch]) for arch in vec_archs}

    # 2. compute scores
    MSE_scores = {
        arch: F.mse_loss(all_preds[arch], all_refs[arch]).item() for arch in vec_archs
    }
    cos_scores = {
        arch: F.cosine_similarity(all_preds[arch], all_refs[arch]).mean().item()
        for arch in vec_archs
    }
    rnk_scores = {
        arch: rank_cosine(all_preds[arch], all_refs[arch]) for arch in vec_archs
    }
    return MSE_scores[vec_archs[0]],cos_scores[vec_archs[0]],rnk_scores[vec_archs[0]]

def get_loader(data):
    data_set = MyDataset(data)
    data_loader = DataLoader(data_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=False,
                          num_workers=CFG['num_workers'])
    data_loader = accelerator.prepare(data_loader)
    return data_loader
train_set = MyDataset(train)
train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True,
                          num_workers=CFG['num_workers'])
dev_en_loader = get_loader(dev_en)
dev_es_loader = get_loader(dev_es)
dev_fr_loader = get_loader(dev_fr)
dev_it_loader = get_loader(dev_it)
dev_ru_loader = get_loader(dev_ru)
#加载模型与tokenizer
from model import SimDeberta
model = SimDeberta.from_pretrained(CFG['model'])

model = model.to(device)
scaler = GradScaler()

optimizer = AdamW(model.parameters(),lr=CFG['lr'], weight_decay=CFG['weight_decay'])
criterion = nn.MSELoss()

# get_cosine_schedule_with_warmup策略，学习率先warmup一个epoch，然后cos式下降
scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // CFG['accum_iter'],
                                            CFG['epochs'] * len(train_loader) // CFG['accum_iter'])

train_loader = accelerator.prepare(train_loader)

log_name = 'reverse_dictionary_all_'+CFG['vectors']
path = log_start(log_name)
time.sleep(0.2)
#训练与测试
for epoch in range(CFG['epochs']):
    loss = train_model(model, train_loader)
    mse1, cos1, rnk1 = test_model(model, dev_en_loader, 'en', dev_en_refs)
    mse2, cos2, rnk2 = test_model(model, dev_es_loader, 'es', dev_es_refs)
    mse3, cos3, rnk3 = test_model(model, dev_fr_loader, 'fr', dev_fr_refs)
    mse4, cos4, rnk4 = test_model(model, dev_it_loader, 'it', dev_it_refs)
    mse5, cos5, rnk5 = test_model(model, dev_ru_loader, 'ru', dev_ru_refs)

    path2 = '{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(path,str(epoch), str(mse1),str(mse2),str(mse3),str(mse4),str(mse5),str(cos1),str(cos2),str(cos3),str(cos4),str(cos5),str(rnk1),str(rnk2),str(rnk3),str(rnk4),str(rnk5))
    os.mkdir(path2)
    model.save_pretrained(path2)
    tokenizer.save_pretrained(path2)