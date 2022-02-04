import pandas as pd
import os

os.chdir(r'C:\semeval\6')

En = pd.read_csv('train.En.csv')

datas = []
dev = []
n = 0
r = range(len(En))
r = list(r)
import random
random.shuffle(r)
for i in r:
    d = En.iloc[i]
    if d['sarcastic'] == 1:
        if n < 170:
            dev.append({'sen':d['tweet'],'label':1})
            dev.append({'sen':d['rephrase'],'label':0})
        else:
            datas.append({'sen':d['tweet'],'label':1})
            datas.append({'sen':d['rephrase'],'label':0})
        n+=1
    else:
        datas.append({'sen':d['tweet'],'label':0})

pd.to_pickle(datas,'Entrain.pk')
pd.to_pickle(dev,'Endev.pk')



Ar = pd.read_csv('train.Ar.csv')

datas = []
dev = []
n = 0
r = range(len(Ar))
r = list(r)
import random
random.shuffle(r)
for i in r:
    d = Ar.iloc[i]
    if d['sarcastic'] == 1:
        if n < 170:
            dev.append({'sen':d['tweet'],'label':1})
            dev.append({'sen':d['rephrase'],'label':0})
        else:
            datas.append({'sen':d['tweet'],'label':1})
            datas.append({'sen':d['rephrase'],'label':0})
        n+=1
    else:
        datas.append({'sen':d['tweet'],'label':0})

pd.to_pickle(datas,'Artrain.pk')
pd.to_pickle(dev,'Ardev.pk')