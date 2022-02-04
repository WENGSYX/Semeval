import json
import os

import pandas as pd

os.chdir(r'C:\semeval\12')

def get_data(filename):
    with open(filename,'r',encoding='utf-8') as f:
        data = json.load(f)
    datas_primary = []
    datas_symbol = []
    for i in list(data.values()):
        label_primary = ''
        label_symbol = ''
        for t in list(i['entity'].values()):
            if t['label'] == 'PRIMARY':
                label_primary += t['text']
                label_primary += ' '+ '<extra_id_0>'+' '
            if t['label'] == 'SYMBOL':
                label_symbol += t['text']
                label_symbol += ' '+ '<extra_id_0>'+' '
        label_primary = label_primary[:-13]
        label_symbol = label_symbol[:-13]
        if label_primary != '':
            datas_primary.append([i['text'],label_primary])
        if label_symbol != '':
            datas_symbol.append([i['text'],label_symbol])
    return datas_primary,datas_symbol

train_primary = []
train_symbol = []

for filename in os.listdir('data_public'):
    p,s = get_data('data_public/'+filename)
    train_primary.extend(p)
    train_symbol.extend(s)
import pandas as pd
pd.to_pickle(train_primary,'train_primary.pk')
pd.to_pickle(train_symbol,'train_symbol.pk')



dev_primary = []
dev_symbol = []

for filename in os.listdir('data_dev_reference'):
    p,s = get_data('data_dev_reference/'+filename)
    dev_primary.extend(p)
    dev_symbol.extend(s)
import pandas as pd
pd.to_pickle(dev_primary,'dev_primary.pk')
pd.to_pickle(dev_symbol,'dev_symbol.pk')