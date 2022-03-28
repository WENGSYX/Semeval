import json
import os
from tqdm import tqdm
os.chdir(r'C:\semeval\1')

datas = []
for file in os.listdir('train'):
    with open('train/'+file,'r',encoding='utf-8') as f:
        data = json.load(f)
    for d in data:
        datas.append(d['gloss'])
for file in os.listdir('test'):
    with open('test/'+file,'r',encoding='utf-8') as f:
        data = json.load(f)
    for d in data:
        try:
            datas.append(d['gloss'])
        except:
            pass

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('mdeberta')
data1 = []
data2 = []
for i in datas:
    if len(tokenizer(i).input_ids) < 32:
        data1.append(i)
    else:
        data2.append(i)

import pandas as pd
pd.to_pickle(data1,'datas1.pk')
pd.to_pickle(data2,'datas2.pk')
