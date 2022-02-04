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
from model import SimDeberta
accelerator = Accelerator()
import transformers
transformers.logging.set_verbosity_error()
from utils import compute_metrics_task2
CFG = { #训练的参数配置
    'seed': 0,#随机种子
    'model': 'deberta_v3_large', #预训练模型
    'max_len': 800, #文本截断的最大长度
    'epochs': 16,#训练轮数
    'train_bs': 7, #batch_size，根据显存调整，对比学习中一般越大越好
    'lr': 1e-5, #学习率
    'num_workers': 0,
    'accum_iter': 4, #梯度累积
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
train = pd.read_pickle('train.pk')
valid = pd.read_pickle('dev.pk')
true = []
for i in valid:
    true.append(i['overall'])
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
    input_ids, attention_mask, token_type_ids,label = [], [], [],[]
    for x in data:

        text = tokenizer('Given a pair of news articles, are they covering the same news story?',x['sen1'])
        input_id = text['input_ids']
        if len(input_id) >220:
            input_id = input_id[:219]+[tokenizer.sep_token_id]
        else:
            input_id += [tokenizer.sep_token_id]
        token_type_id = [0] * len(input_id)

        text2 = tokenizer(x['sen2'])
        input_id2 = text['input_ids']
        input_id+= input_id2
        if len(input_id) >= 420:
            input_id = input_id[:419]+[tokenizer.sep_token_id]
        else:
            input_id+= [tokenizer.sep_token_id]
        attention = [1] * len(input_id)
        token_type_id += [1] * (len(input_id)-len(token_type_id))


        attention += [0] * (420-len(input_id))
        token_type_id += [0] * (420-len(input_id))
        input_id += [tokenizer.pad_token_id] * (420-len(input_id))
        input_ids.append(input_id)
        attention_mask.append(attention)
        token_type_ids.append(token_type_id)
        label.append((x['overall']-1)/3)

    input_ids = torch.tensor(input_ids,device=device)
    attention_mask = torch.tensor(attention_mask,device=device)
    token_type_ids = torch.tensor(token_type_ids,device=device)
    label = torch.tensor(label)
    return input_ids, attention_mask,token_type_ids,label

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


#训练模型
def train_model(model,fgm,train_loader):  # 训练一个epoch
    model.train()
    losses = AverageMeter()
    acces = AverageMeter()
    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
    for step, (input_ids, attention_mask,token_type_ids,label) in enumerate(tk):
        #开启pf16训练

        #取bert的最后一层输出值，计算损失
        output = model(input_ids, attention_mask,token_type_ids).clamp(min=0,max=1)
        loss = criterion(output.float(),label.float())
        loss.backward()

        fgm.attack()  # 在embedding上添加对抗扰动
        output2 = model(input_ids, attention_mask,token_type_ids).clamp(min=0,max=1)
        loss2 = criterion(output2.float(),label.float())
        scaler.scale(loss2).backward()
        fgm.restore()

        if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
        #显示损失
        losses.update(loss.item())
        acces.update((loss/len(label)).item())
        tk.set_postfix(loss=losses.avg,error=acces.avg)
        if step == 0:
            log(['开始新一轮训练','现轮数:{}'.format(epoch),'训练起始损失：{}'.format(str(loss.item())),'总batch数：{}'.format(len(tk))],path)

    log(['训练最终损失：{}'.format(str(loss.item())),'训练平均损失：{}'.format(losses.avg),'结束本轮训练'],path)
    return losses.avg


def test_model(model,train_loader):  # 训练一个epoch
    model.eval()
    losses = AverageMeter()
    acces = AverageMeter()
    optimizer.zero_grad()
    pred = []
    with torch.no_grad():
        tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
        for step, (input_ids, attention_mask,token_type_ids,label) in enumerate(tk):
            output = model(input_ids, attention_mask, token_type_ids).clamp(min=0, max=1)
            loss = criterion(output.float(),label.float())

            pred.extend(output)
            losses.update(loss.item())
            acces.update(loss / len(label))
            tk.set_postfix(loss=losses.avg,error=acces.avg)

    return pred
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
from transformers import DebertaV2PreTrainedModel,DebertaV2Config,DebertaV2Model
class Debertafortask2(DebertaV2PreTrainedModel):
    def __init__(self, config: DebertaV2Config):
        super().__init__(config)
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
    ):
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output.mean(1))[:,0]
        return logits
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
#处理数据
train_set = MyDataset(train)
train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True,
                          num_workers=CFG['num_workers'])
valid_set = MyDataset(valid)
valid_loader = DataLoader(valid_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=False,
                          num_workers=CFG['num_workers'])
#加载模型与tokenizer

model = Debertafortask2.from_pretrained(CFG['model'])

model = model.to(device)
scaler = GradScaler()

optimizer = AdamW(model.parameters(),lr=CFG['lr'], weight_decay=CFG['weight_decay'])
criterion = nn.MSELoss()

# get_cosine_schedule_with_warmup策略，学习率先warmup一个epoch，然后cos式下降
scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // CFG['accum_iter'],
                                            CFG['epochs'] * len(train_loader) // CFG['accum_iter'])

train_loader,valid_loader = accelerator.prepare(train_loader,valid_loader)
log_name = '交互式baseline_fgm_mse'
path = log_start(log_name)
fgm = FGM(model)
for epoch in range(CFG['epochs']):
    time.sleep(0.2)
    #训练与测试
    loss = train_model(model, fgm, train_loader)
    pred = test_model(model,valid_loader)
    log_test,l = compute_metrics_task2((torch.stack(pred).cpu().numpy()*3)+1,true)
    log(['epoch:{}'.format(epoch)]+log_test,path)
    path2 = path + '/{}_{}_{}'.format(str(epoch), str(l['rho']),str(l['p-value']))
    os.mkdir(path2)
    model.save_pretrained(path2)
    tokenizer.save_pretrained(path2)