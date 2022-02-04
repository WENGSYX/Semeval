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


datas = []
for file in os.listdir('train'):
    with open('train/'+file,'r',encoding='utf-8') as f:
        data = json.load(f)
    gloss = []

    for d in data:
        datas.append(d['gloss'])