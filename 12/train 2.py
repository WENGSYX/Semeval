import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from transformers import AutoModelForSequenceClassification,AutoTokenizer,AdamW,get_cosine_schedule_with_warmup,AutoModelForSequenceClassification,BertTokenizer
#from transformers.deepspeed import HfDeepSpeedConfig
#from scorer import run_evaluation
from torch.autograd import Variable
#from deberta import DebertaForMultipleChoice
from transformers import AutoTokenizer,DebertaV2ForTokenClassification
from accelerate import Accelerator
from sklearn.metrics import f1_score
accelerator = Accelerator()

import argparse
from allennlp.nn import util
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1)
parser.add_argument("--deepspeed_config",default=-1)
args = parser.parse_args()

CFG = { #训练的参数配置
    'fold_num': 5, #五折交叉验证
    'seed': 42,
    'model': r'C:\semeval\12\log\ner_long\2_0.7233381543389998', #预训练模型
    'max_len': 720, #文本截断的最大长度
    'epochs': 20,
    'train_bs': 3, #batch_size，可根据自己的显存调整
    'valid_bs': 6,
    'lr': 6e-6, #学习率
    'num_workers': 0,
    'accum_iter': 1, #梯度累积，相当于将batch_size*2
    'weight_decay': 1e-5, #权重衰减，防止过拟合
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


train = pd.read_pickle('train.pk')
valid = pd.read_pickle('dev.pk')


tokenizer = AutoTokenizer.from_pretrained(CFG['model'])

class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # print(sen1)
        data = self.df[idx]

        return data


def collate_fn(data):
    input_ids, attention_masks,labels = [], [], []

    for x in data:
        input_id = [tokenizer.cls_token_id]
        label = [-100]
        for n in range(len(x[0])):
            id = tokenizer(x[0][n]).input_ids[1:-1]
            input_id += id
            label += [x[1][n]] * len(id)

        input_len = len(input_id)+1
        input_id += [tokenizer.sep_token_id]
        input_id = torch.tensor(input_id)
        input_id_all = tokenizer(''.join(x[0])).input_ids
        if len(input_id_all) == len(input_id):
            input_id = input_id_all
            if input_len <= CFG['max_len']:
                attention_mask = [1] * input_len + [0] * (CFG['max_len'] - input_len)
                input_id = input_id + [tokenizer.pad_token_id] * (CFG['max_len'] - input_len)
                label = label + [-100] + [-100] * (CFG['max_len'] - input_len)
            else:
                attention_mask = [1] * CFG['max_len']
                input_id = input_id[:CFG['max_len'] - 1] + [tokenizer.sep_token_id]
                label = label[:CFG['max_len'] - 1] + [-100]
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(label)

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_masks)
    labels = torch.tensor(labels)

    return input_ids, attention_mask,labels

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
def train_model(model,fgm,train_loader,epoch):  # 训练一个epoch
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()
    f1s  = AverageMeter()
    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
    for step, (input_ids, attention_mask,labels) in enumerate(tk):
        if len(input_ids) != 0:
            output = model(input_ids, attention_mask=attention_mask,labels=labels)
            active_loss = attention_mask.view(-1) == 1
            active_logits = output.logits.view(-1, 4)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(-100).type_as(labels)
            )
            loss = criterion(active_logits,active_labels)
            loss.backward()

            if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()

            losses.update(loss.item())

        tk.set_postfix(loss=losses.avg)
        if step == 0:
            log(['开始新一轮训练','现轮数：{}'.format(epoch),'训练起始损失：{}'.format(str(loss.item())),'总batch数：{}'.format(len(tk))],path)
    log(['训练最终损失：{}'.format(str(loss.item())),'训练平均损失：{}'.format(losses.avg),'结束本轮训练'],path)
    return losses.avg


def get_F1(y_true,y_pred):
    if y_true == []:
        if y_pred == []:
            return 1
        else:
            return 0
    else:
        if y_pred == []:
            return 0

        y_true = set(y_true)
        y_pred = set(y_pred)

        TP = len(y_true.intersection(y_pred))
        FP = len(y_pred)-TP
        FN = len(y_true)-TP

        P = TP/(TP+FP)+1e-9
        R = TP/(TP+FN)+1e-9

        F1 = (2 * P * R)/(P+R)+1e-9
        return F1
def test_model(model, val_loader,epoch,valid):  # 验证
    model.eval()

    losses = AverageMeter()
    primary_f1s,symbol_f1s,ordered_f1s = AverageMeter(),AverageMeter(),AverageMeter()
    f1s = AverageMeter()
    log(['开始测试','现轮数：{}'.format(epoch)],path)
    tk = tqdm(val_loader)
    output_words = []
    with torch.no_grad():
        for idx, (input_ids, attention_mask,labels) in enumerate(tk):

            output = model(input_ids, attention_mask=attention_mask,labels=labels)
            active_loss = attention_mask.view(-1) == 1
            active_logits = output.logits.view(-1, 4)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(-100).type_as(labels)
            )
            loss = criterion(active_logits, active_labels)
            losses.update(loss.item())
            logits = output.logits.argmax(2)
            logits = torch.where(labels==-100,-100,logits)

            input_ids = input_ids.tolist()
            for logitsn in range(len(logits)):
                t = logits[logitsn]
                la = labels[logitsn]
                last_l = -1
                primarys,symbols,ordereds=[],[],[]
                primary,symbol,ordered=[],[],[]
                for ln in range(len(t)):
                    l = t[ln]
                    if l != last_l:
                        if primary != []:
                            primarys.append(tokenizer.decode(primary))
                            primary = []
                        if symbol != []:
                            symbols.append(tokenizer.decode(symbol))
                            symbol = []
                        if ordered != []:
                            ordereds.append(tokenizer.decode(ordered))
                            ordered = []
                    if l == 1:
                        primary.append(input_ids[logitsn][ln])
                    if l == 2:
                        symbol.append(input_ids[logitsn][ln])
                    if l == 3:
                        ordered.append(input_ids[logitsn][ln])
                    last_l = l

            if idx % 10 == 0:
                log([str(primarys_true)]+[str(primarys)],path)
                log([str(symbols_true)] + [str(symbols)], path)
            tk.set_postfix(loss=losses.avg,f1=f1s.avg,primary_f1=primary_f1s.avg,symbol_f1=symbol_f1s.avg,ordered_f1=ordered_f1s.avg)

    log(['结束本轮测试']+['测试F1:{}'.format(f1s.avg)]+['Primary_f1:{}'.format(primary_f1s.avg)]+['Symbol_f1:{}'.format(symbol_f1s.avg)]+['Ordered_f1:{}'.format(ordered_f1s.avg)],path)
    return f1s.avg





def log(text,path):
    with open(path+'/log.txt','a',encoding='utf-8') as f:
        f.write('-----------------{}-----------------'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        f.write('\n')
        for i in text:
            f.write(i)
            print(i)
            f.write('\n')
        f.write('\n')
import os
def log_start(log_name):
    if log_name == '':
        log_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    else:
        try:
            os.mkdir('log/' + log_name)
        except:
            log_name += time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            os.mkdir('log/' + log_name)
    with open('log/' + log_name + '/python_file.py', 'a', encoding='utf-8') as f:
        with open(os.path.basename(__file__),'r',encoding='utf-8') as f2:
            file = f2.read()
        f.write(file)

    path = 'log/' + log_name
    with open(path+'/log.txt', 'a', encoding='utf-8') as f:
        f.write(log_name)
        f.write('\n')
    return path
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs,targets):
        if self.logits:
            BCE_loss = F.cross_entropy(inputs, targets, reduce=False)
        else:
            targets = torch.eye(35)[targets.reshape(-1)].to(device)
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
class In_trust_Loss(nn.Module):
    def __init__(self, alpha=1, beta=0.8, delta=0.5, num_classes=35):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.delta = delta
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        # self.crf = CRF(num_tags= num_classes, batch_first=True)

    def forward(self, logits, labels, label_attention):
        # loss_mask = labels.gt(0)
        # Loss CRF
        labels = torch.where(labels == -100, 0, labels)

        ce = util.sequence_cross_entropy_with_logits(logits, labels,
                                                     label_attention[:, :].contiguous(),
                                                     average="token")
        # ce = self.cross_entropy(logits,labels)
        # Loss In_trust
        active_logits = logits.view(-1, self.num_classes)
        active_labels = labels.view(-1)

        pred = F.softmax(active_logits, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(active_labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        dce = (-1 * torch.sum(pred * torch.log(pred * self.delta + label_one_hot * (1 - self.delta)), dim=1))

        # Loss

        loss = self.alpha * ce - self.beta * dce.mean()
        return loss


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

criterion = FocalLoss()

train_set = MyDataset(train)
valid_set = MyDataset(valid)
train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True,
                          num_workers=CFG['num_workers'])
valid_loader = DataLoader(valid_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False,
                          num_workers=CFG['num_workers'])

best_acc = 0

model = DebertaV2ForTokenClassification.from_pretrained(CFG['model'],num_labels = 4)

model = model.to(device)
scaler = GradScaler()
optimizer = AdamW(model.parameters(),lr=CFG['lr'], weight_decay=CFG['weight_decay'])


scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // CFG['accum_iter'],
                                            CFG['epochs'] * len(train_loader) // CFG['accum_iter'])

train_loader,val_loader = accelerator.prepare(train_loader,valid_loader)

log_name = 'ner_long'
path = log_start(log_name)

fgm = FGM(model)
for epoch in range(CFG['epochs']):

    train_loss = train_model(model,fgm,train_loader,epoch)
    score = test_model(model, val_loader,epoch,valid)
    path2 = '{}/{}_{}'.format(path, str(epoch), str(score))
    os.mkdir(path2)
    model.save_pretrained(path2)
    tokenizer.save_pretrained(path2)



