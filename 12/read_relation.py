import json
import os

import pandas as pd

os.chdir(r'C:\semeval\12')

"""
('PRIMARY', 'PRIMARY') {'Corefer-Description'}
('PRIMARY', 'SYMBOL')  {'Count', 'Direct','Corefer-Symbol'}
('SYMBOL', 'SYMBOL') {'Corefer-Symbol'}
"""
set_pp,set_ps,set_ss = set(),set(),set()
import random
def get_data(filename):
    with open(filename,'r',encoding='utf-8') as f:
        data = json.load(f)
    pp = []
    ps = []
    ss = []

    for i in list(data.values()):
        for t in list(i['relation'].values()):
            if (i['entity'][t['arg0']]['label'],i['entity'][t['arg1']]['label']) == ('PRIMARY', 'PRIMARY'):
                pp.append(
                    {'text': i['text'], 'arg0': i['entity'][t['arg0']]['text'], 'arg1': i['entity'][t['arg1']]['text'],
                     'label': {'Corefer-Description': 1}[t['label']]})
                set_pp.add((i['entity'][t['arg0']]['text'],i['entity'][t['arg1']]['text']))
            if (i['entity'][t['arg0']]['label'],i['entity'][t['arg1']]['label']) == ('PRIMARY', 'SYMBOL'):
                ps.append(
                    {'text': i['text'], 'arg0': i['entity'][t['arg0']]['text'], 'arg1': i['entity'][t['arg1']]['text'],
                     'label': {'Count': 1,'Direct':2,'Corefer-Symbol':3}[t['label']]})
                set_ps.add((i['entity'][t['arg0']]['text'], i['entity'][t['arg1']]['text']))
            if (i['entity'][t['arg0']]['label'],i['entity'][t['arg1']]['label']) == ('SYMBOL', 'SYMBOL'):
                ss.append(
                    {'text': i['text'], 'arg0': i['entity'][t['arg0']]['text'], 'arg1': i['entity'][t['arg1']]['text'],
                     'label': {'Corefer-Symbol':1}[t['label']]})
                set_ss.add((i['entity'][t['arg0']]['text'], i['entity'][t['arg1']]['text']))
        if len(i['entity'].keys()) !=0:
            for _ in range(18):
                a = random.choice(list(i['entity'].keys()))
                b = random.choice(list(i['entity'].keys()))
                if i['entity'][a]['label'] == 'PRIMARY' and i['entity'][b]['label'] == 'PRIMARY':
                    if (i['entity'][a]['text'],i['entity'][b]['text']) not in set_pp:
                        pp.append(
                            {'text': i['text'], 'arg0': i['entity'][a]['text'], 'arg1': i['entity'][b]['text'],
                             'label': 0})
                        set_pp.add((i['entity'][a]['text'],i['entity'][b]['text']))
            for _ in range(40):
                a = random.choice(list(i['entity'].keys()))
                b = random.choice(list(i['entity'].keys()))
                if i['entity'][a]['label'] == 'PRIMARY' and i['entity'][b]['label'] == 'SYMBOL':
                    if (i['entity'][a]['text'],i['entity'][b]['text']) not in set_ps:
                        ps.append(
                            {'text': i['text'], 'arg0': i['entity'][a]['text'], 'arg1': i['entity'][b]['text'],
                             'label': 0})
                        set_ps.add((i['entity'][a]['text'],i['entity'][b]['text']))
            for _ in range(25):
                a = random.choice(list(i['entity'].keys()))
                b = random.choice(list(i['entity'].keys()))
                if i['entity'][a]['label'] == 'SYMBOL' and i['entity'][b]['label'] == 'SYMBOL':
                    if (i['entity'][a]['text'],i['entity'][b]['text']) not in set_ss:
                        ss.append(
                            {'text': i['text'], 'arg0': i['entity'][a]['text'], 'arg1': i['entity'][b]['text'],
                             'label': 0})
                        set_ss.add((i['entity'][a]['text'],i['entity'][b]['text']))


    return pp,ps,ss

train_pp,train_ps,train_ss = [] ,[], []

for filename in os.listdir('data_public'):
    pp,ps,ss = get_data('data_public/'+filename)
    train_pp.extend(pp)
    train_ps.extend(ps)
    train_ss.extend(ss)
import pandas as pd
pd.to_pickle(train_pp,'train_pp.pk')
pd.to_pickle(train_ps,'train_ps.pk')
pd.to_pickle(train_ss,'train_ss.pk')
train = []
for i in train_pp:
    if i['label'] !=0:
        train.append({'text':i['text'],'arg0':i['arg0'],'arg1':i['arg1'],'label':1})
    else:
        train.append({'text': i['text'], 'arg0': i['arg0'], 'arg1': i['arg1'], 'label': 0})
for i in train_ps:
    if i['label'] !=0:
        train.append({'text':i['text'],'arg0':i['arg0'],'arg1':i['arg1'],'label':1})
    else:
        train.append({'text': i['text'], 'arg0': i['arg0'], 'arg1': i['arg1'], 'label': 0})
for i in train_ss:
    if i['label'] !=0:
        train.append({'text':i['text'],'arg0':i['arg0'],'arg1':i['arg1'],'label':1})
    else:
        train.append({'text': i['text'], 'arg0': i['arg0'], 'arg1': i['arg1'], 'label': 0})

pd.to_pickle(train,'train_re.pk')

dev_pp,dev_ps,dev_ss = [] ,[], []

for filename in os.listdir('data_dev_reference'):
    pp,ps,ss = get_data('data_dev_reference/'+filename)
    dev_pp.extend(pp)
    dev_ps.extend(ps)
    dev_ss.extend(ss)
import pandas as pd
pd.to_pickle(dev_pp,'dev_pp.pk')
pd.to_pickle(dev_ps,'dev_ps.pk')
pd.to_pickle(dev_ss,'dev_ss.pk')