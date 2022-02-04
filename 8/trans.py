import pandas as pd
import os

os.chdir('C:\semeval\8')

train = pd.read_csv('semeval-2022_task8_train-data_batch.csv')

data2 = {}
import json

for f in os.listdir('train'):
    for p in os.listdir('train/' + f):
        if 'json' in p:
            with open('train/' + f + '/' + p, 'r', encoding='utf-8') as c:
                data2[p[:-5]] = json.load(c)['text']
d = []
type = []
for i in range(len(train)):
    l = train.iloc[i]
    try:
        if l['url1_lang'] != 'en':
            d.append([l['url1_lang'],data2[l['pair_id'].split('_')[0]].replace('\n', '')])
            type.append(l['url1_lang'])
    except:
        pass
    try:
        if l['url2_lang'] != 'en':
            d.append([l['url2_lang'],data2[l['pair_id'].split('_')[1]].replace('\n', '')])
            type.append(l['url2_lang'])
    except:
        pass
type = list(set(type))
for i in type:
    exec('{} = []'.format(i))

for p in d:
    for i in p[1].split('.'):
        exec('{}.append(i)'.format(p[0]))
for i in type:
    exec('{} = list(set({}))'.format(i,i))
    exec("pd.to_pickle({},'{}.pk')".format(i,i))
pd.to_pickle(type,'type.pk')



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

os.chdir('C:\semeval\8')

type = 'ar'

device = 0
model = M2M100ForConditionalGeneration.from_pretrained('D:\python\比赛\天池\翻译\m2m').to(device)  # 模型

sen1_tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_1.2B')
sen1_tokenizer.src_lang = type
sen2_tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_1.2B')
sen2_tokenizer.src_lang = 'en'

data = pd.read_pickle(type+'.pk')
nums = []
for x in data:
    nums.append(len(sen1_tokenizer(x).input_ids))
nums.sort()
nums = nums[int(0.95*len(nums))]
print(nums)


class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        sen = self.df[idx]

        return sen

def collate_fn(data):
    input_ids, attention_mask, token_type_ids,label = [], [], [],[]
    for x in data:
        text = sen1_tokenizer(x,padding='max_length', truncation=True, max_length=nums, return_tensors='pt')
        input_ids.append(text['input_ids'].squeeze().tolist())
        attention_mask.append(text['attention_mask'].squeeze().tolist())
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    return input_ids, attention_mask

def fanyi(text):
    input_ids, attention_mask, token_type_ids,label = [], [], [],[]
    for x in text:
        text = sen1_tokenizer(x,padding='max_length', truncation=True, max_length=nums, return_tensors='pt')
        input_ids.append(text['input_ids'].squeeze().tolist())
        attention_mask.append(text['attention_mask'].squeeze().tolist())
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)

    encoded = sen1_tokenizer(text, return_tensors="pt").to(device)
    output_tokens = model.generate(**encoded, forced_bos_token_id=sen2_tokenizer.get_lang_id('en'))
    output = sen2_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    return output

data = MyDataset(data)
data = DataLoader(data, batch_size=24, collate_fn=collate_fn, shuffle=False,
                  num_workers=0)

result = []
model.eval()

with torch.no_grad():
    tk = tqdm(data, total=len(data), position=0, leave=True)
    for idx, (input_ids, attention_mask) in enumerate(tk):
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(
            device)

        output_tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       forced_bos_token_id=sen2_tokenizer.get_lang_id('en'))

        output = sen2_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        result.extend(output)
pd.to_pickle(result,type+'_en.pk')



type = pd.read_pickle('type.pk')
for i in type:
    exec('{}=dict()'.format(i))
    d1 = pd.read_pickle('{}.pk'.format(i))
    d2 = pd.read_pickle('{}_en.pk'.format(i))
    for p in range(len(d1)):
        exec('{}[d1[p]]=d2[p]'.format(i))
    exec('{}[""]=""'.format(i))


train = pd.read_csv('semeval-2022_task8_train-data_batch.csv')
data2 = {}
import json
for f in os.listdir('train'):
    for p in os.listdir('train/' + f):
        if 'json' in p:
            with open('train/' + f + '/' + p, 'r', encoding='utf-8') as c:
                data2[p[:-5]] = json.load(c)['text']
d = []
type = []
for i in range(len(train)):
    l = train.iloc[i]
    try:
        if l['url1_lang'] != 'en':
            d.append([l['url1_lang'],data2[l['pair_id'].split('_')[0]].replace('\n', '')])
            type.append(l['url1_lang'])
    except:
        pass
    try:
        if l['url2_lang'] != 'en':
            d.append([l['url2_lang'],data2[l['pair_id'].split('_')[1]].replace('\n', '')])
            type.append(l['url2_lang'])
    except:
        pass


data = {}
for i in d:
    sen = ''
    for p in i[1].split('.'):
        exec('sen+={}[p]'.format(i[0]))
        sen+='. '
    data[i[1]]=sen
pd.to_pickle(data,'train_forest_sen.pk')

da = []
forest = pd.read_pickle('train_forest_sen.pk')
for i in range(len(train)):
    l = train.iloc[i]
    try:
        if l['url1_lang'] != 'en':
            sen1 = forest[data2[l['pair_id'].split('_')[0]].replace('\n', '')]
        else:
            sen1 = data2[l['pair_id'].split('_')[0]].replace('\n', '')
        if l['url2_lang'] != 'en':
            sen2 = forest[data2[l['pair_id'].split('_')[1]].replace('\n', '')]
        else:
            sen2 = data2[l['pair_id'].split('_')[1]].replace('\n', '')
        da.append({'sen1': sen1,
                   'sen2': sen2, 'overall': l['Overall']})
    except:
        pass

import random
random.shuffle(da)
pd.to_pickle(da[:700],'dev.pk')
pd.to_pickle(da[700:],'train.pk')