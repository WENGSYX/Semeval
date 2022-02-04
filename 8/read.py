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
for i in range(len(train)):
    l = train.iloc[i]
    try:
        d.append({'sen1': data[l['pair_id'].split('_')[0]].replace('\n', ''),
                  'sen2': data[l['pair_id'].split('_')[1]].replace('\n', ''), 'overall': l['Overall']})
    except:
        pass
import random
random.shuffle(d)
pd.to_pickle(d[:700],'dev.pk')
pd.to_pickle(d[700:],'train.pk')


train = pd.read_csv('semeval-2022_task8_eval_data_202201.csv')

data = {}
import json

for f in os.listdir('eval'):
    for p in os.listdir('eval/' + f):
        if 'json' in p:
            with open('eval/' + f + '/' + p, 'r', encoding='utf-8') as c:
                data[p[:-5]] = json.load(c)['text']

d = []
for i in range(len(train)):
    l = train.iloc[i]
    try:
        d.append({'sen1': data[l['pair_id'].split('_')[0]].replace('\n', ''),
                  'sen2': data[l['pair_id'].split('_')[1]].replace('\n', ''), 'overall': l['Overall']})
    except:
        pass
pd.to_pickle(d,'eval.pk')