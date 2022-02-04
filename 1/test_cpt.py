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


tokenizer = AutoTokenizer.from_pretrained(r'C:\semeval\1\new\log\eval_defmod_all_ru2\11_cpt')


class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        sen = self.df[idx]

        return sen


def collate_fn(data):
    input_ids,attention_mask, token_type_ids,labels= [],[], [], []
    for x in data:
        input_ids.append(x['sgns'])

    input_ids = torch.tensor(input_ids)
    return input_ids




from model import CPTForConditionalGeneration
model = CPTForConditionalGeneration.from_pretrained(r'C:\semeval\1\new\log\eval_defmod_all_ru2\11_cpt').to(device)


filename = r'ru.test.defmod.json'
with open('C:/semeval/1/test/'+filename, 'r', encoding='utf-8') as f:
    data = json.load(f)
data_set = MyDataset(data)
data_loader = DataLoader(data_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=False,
                          num_workers=CFG['num_workers'])

outs = []
tk = tqdm(data_loader)
with torch.no_grad():
    for inputs in tk:
        inputs = inputs.to(device)
        output = tokenizer.batch_decode(model.generate(inputs, max_length=100))
        outs.extend(output)


result = []
for i in range(len(data)):
    result.append({'id':data[i]['id'],'gloss':outs[i]})
import json
with open('result/{}'.format(filename),'w',encoding='utf-8') as f:
    json.dump(result,f)