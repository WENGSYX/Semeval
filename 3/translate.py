import os
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
import time
from sklearn.model_selection import *
from transformers import *
from torch.autograd import Variable
import sacrebleu
import json

CFG = { #训练的参数配置
    'fold_num': 5, #五折交叉验证
    'seed': 2,
    'model': 'facebook/m2m100_1.2B', #预训练模型
    'max_len': 64, #文本截断的最大长度
    'epochs': 8,
    'train_bs': 20, #batch_size，可根据自己的显存调整
    'valid_bs': 1,
    'lr': 1e-4, #学习率
    'num_workers': 0,
    'accum_iter': 4, #梯度累积，相当于将batch_size*2
    'weight_decay': 1e-4, #权重衰减，防止过拟合
    'device': 0,
    'sen1':'it',
    'sen2':'en'
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


sen1_tokenizer = M2M100Tokenizer.from_pretrained(CFG['model'])
sen1_tokenizer.src_lang = CFG['sen1']
sen2_tokenizer = M2M100Tokenizer.from_pretrained(CFG['model'])
sen2_tokenizer.src_lang = CFG['sen2']


model = M2M100ForConditionalGeneration.from_pretrained(CFG['model']).to(device)  # 模型




def fanyi(data_name):
    data = pd.read_pickle(data_name)
    ds = []
    for d in tqdm(data):
        text = d['sen']
        encoded = sen1_tokenizer(text, return_tensors="pt").to(device)
        output_tokens = model.generate(**encoded, forced_bos_token_id=sen2_tokenizer.get_lang_id(CFG['sen2']))
        output = sen2_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        ds.append({'sen':output,'label':d['label']})
    pd.to_pickle(ds,data_name)
    return None

fanyi('it_subtask1_train.pk')
fanyi('it_subtask1_valid.pk')
fanyi('it_subtask2_train.pk')
fanyi('it_subtask2_valid.pk')