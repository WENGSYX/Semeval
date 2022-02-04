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
    'model': r'C:\semeval\1\new\log\reverse_dictionary_all_sgns\4_0.9799_0.9331_0.9643_1.0742_0.5275_0.243_0.353_0.336_0.3517', #预训练模型
    'max_len': 110, #文本截断的最大长度
    'epochs': 16,#训练轮数
    'train_bs': 60, #batch_size，根据显存调整，对比学习中一般越大越好
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


tokenizer = AutoTokenizer.from_pretrained(CFG['model'])


class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):
        gloss = self.df[idx]['gloss']
        vector = []
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




from model import SimDeberta
model_char = SimDeberta.from_pretrained(r'C:\semeval\1\new\log\reverse_dictionary_all_to_it_char\4_0.3396_0.7443_0.4331')
model_char = model_char.to(device)

model_sgns = SimDeberta.from_pretrained(r'C:\semeval\1\new\log\reverse_dictionary_all_to_it_sgns\2_1.0571_0.3524_0.243')
model_sgns = model_sgns.to(device)




filename = r'it.test.revdict.json'
with open('C:/semeval/1/test/'+filename, 'r', encoding='utf-8') as f:
    data = json.load(f)
data_set = MyDataset(data)
data_loader = DataLoader(data_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=False,
                          num_workers=CFG['num_workers'])


pred_char,pred_sgns = [],[]
with torch.no_grad():
    tk = tqdm(data_loader, total=len(data_loader), position=0, leave=True,desc=filename)
    for step, (input_ids, attention_mask, labels) in enumerate(tk):
        output_char = model_char(input_ids=input_ids, attention_mask=attention_mask)
        output_sgns = model_sgns(input_ids=input_ids, attention_mask=attention_mask)


        pred_char.extend(output_char.tolist())
        pred_sgns.extend(output_sgns.tolist())


result = []
for i in range(len(data)):
    result.append({'id':data[i]['id'],'gloss':data[i]['gloss'],'sgns':pred_sgns[i],'char':pred_sgns[i]})
import json
with open('result/{}'.format(filename),'w',encoding='utf-8') as f:
    json.dump(result,f)