import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from transformers import AutoModelForSequenceClassification,AutoTokenizer,AdamW,get_cosine_schedule_with_warmup,AutoModelForSequenceClassification,BertTokenizer
#from transformers.deepspeed import HfDeepSpeedConfig
#from scorer import run_evaluation
from torch.autograd import Variable
#from deberta import DebertaForMultipleChoice
from transformers import AutoTokenizer,DebertaV2ForTokenClassification
from accelerate import Accelerator
from sklearn.metrics import f1_score
accelerator = Accelerator()

CFG = { #训练的参数配置
    'fold_num': 5, #五折交叉验证
    'seed': 42,
    'model': r'C:\semeval\12\log\ner_baseline\14_0.8933182677981327', #预训练模型
    'max_len': 1024, #文本截断的最大长度
    'epochs': 20,
    'train_bs': 6, #batch_size，可根据自己的显存调整
    'valid_bs': 6,
    'lr': 1e-5, #学习率
    'num_workers': 0,
    'accum_iter': 1, #梯度累积，相当于将batch_size*2
    'weight_decay': 1e-5, #权重衰减，防止过拟合
    'device': 0,
}

torch.cuda.set_device(CFG['device'])
device = accelerator.device


tokenizer = AutoTokenizer.from_pretrained(CFG['model'])


def collate_fn(data):
    input_ids, attention_masks,labels = [], [], []

    for x in data:
        input_id = [tokenizer.cls_token_id]
        label = [-100]
        for n in range(len(x[0])):
            id = tokenizer(x[0][n]).input_ids[1:-1]
            input_id += id
            label += [x[1][n]] * len(id)

        input_len = len(input_id)+1
        if input_len <= CFG['max_len']:
            attention_mask = [1] * input_len + [0] * (CFG['max_len'] - input_len)
            input_id = input_id + [tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (CFG['max_len'] - input_len)
            label = label + [-100] + [-100] * (CFG['max_len'] - input_len)
        else:
            attention_mask = [1] * CFG['max_len']
            input_id = input_id[:CFG['max_len'] - 1] + [tokenizer.sep_token_id]
            label = label[:CFG['max_len'] - 1] + [-100]
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        labels.append(label)

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_masks)
    labels = torch.tensor(labels)

    return input_ids, attention_mask,labels



from transformers import DebertaV2ForSequenceClassification
import re
model_ner = DebertaV2ForTokenClassification.from_pretrained(r'C:\semeval\12\log\ner_long2\10_0.757334820601059',num_labels = 4)
model_ner = model_ner.to(device)

model_ss = DebertaV2ForSequenceClassification.from_pretrained(r'C:\semeval\12\log\re_baseline_ss\7_0.9776902887139106',num_labels=4)
model_ss = model_ss.to(device)

model_ps = DebertaV2ForSequenceClassification.from_pretrained(r'C:\semeval\12\log\re_baseline_ps\8_0.9728120873649804',num_labels=4)
model_ps = model_ps.to(device)

model_pp = DebertaV2ForSequenceClassification.from_pretrained(r'C:\semeval\12\log\re_baseline_pp\14_0.7868852459016393',num_labels=4)
model_pp = model_pp.to(device)



for file in os.listdir('data_dev_reference'):
    if 'json' in file:
        results = {}
        with open('data_dev_reference/{}'.format(file),'r',encoding='utf-8') as f:
            data = json.load(f)
        for da in tqdm(list(data.values())):
            text = da['text']
            text_token = tokenizer(text,return_tensors='pt')
            input_ids = text_token['input_ids'].to(device)
            attention_mask = text_token['attention_mask'].to(device)
            output = model_ner(input_ids=input_ids,attention_mask=attention_mask).logits

            input_ids = input_ids.tolist()[0]
            logits = output.argmax(2)[0]
            last_l = -1
            primarys, symbols, ordereds = [], [], []
            primary, symbol, ordered = [], [], []
            dis_ids = []
            d_n = 1
            entity = {}
            for logitsn in range(1,len(logits)-1):
                l = logits[logitsn]
                if l != last_l:
                    if primary != []:
                        token = tokenizer.decode(primary)
                        if text.count(token) == 1:
                            entity["T{}".format(d_n)]={
                                "eid": "T{}".format(d_n),
                                "label": "PRIMARY",
                                "start": text.find(token),
                                "end": text.find(token)+len(token),
                                "text": token
                              }
                            d_n += 1
                        else:
                            spans = []
                            span_token = ''
                            for ts in text.split(token):
                                span_token += ts
                                span_token += token
                                spans.append(len(tokenizer(span_token).input_ids))
                            spans_min = min(spans)
                            spans = []
                            span_token = ''
                            for ts in text.split(token):
                                span_token += ts
                                span_token += token
                                if len(tokenizer(span_token).input_ids) == spans_min:
                                    result = span_token

                            entity["T{}".format(d_n)] = {
                                "eid": "T{}".format(d_n),
                                "label": "PRIMARY",
                                "start": len(result)-len(token),
                                "end": len(result),
                                "text": token
                            }
                            d_n += 1
                        primary = []
                    if symbol != []:
                        token = tokenizer.decode(symbol)
                        if text.count(token) == 1:
                            entity["T{}".format(d_n)]={
                                "eid": "T{}".format(d_n),
                                "label": "SYMBOL",
                                "start": text.find(token),
                                "end": text.find(token)+len(token),
                                "text": token
                              }
                            d_n += 1
                        else:
                            spans = []
                            span_token = ''
                            for ts in text.split(token):
                                span_token += ts
                                span_token += token
                                spans.append(len(tokenizer(span_token).input_ids))
                            spans_min = min(spans)
                            spans = []
                            span_token = ''
                            for ts in text.split(token):
                                span_token += ts
                                span_token += token
                                if len(tokenizer(span_token).input_ids) == spans_min:
                                    result = span_token

                            entity["T{}".format(d_n)] = {
                                "eid": "T{}".format(d_n),
                                "label": "SYMBOL",
                                "start": len(result)-len(token),
                                "end": len(result),
                                "text": token
                            }
                            d_n += 1
                        symbol = []
                    if ordered != []:
                        token = tokenizer.decode(ordered)
                        if text.count(token) == 1:
                            entity["T".format(d_n)]={
                                "eid": "T{}".format(d_n),
                                "label": "ORDERED",
                                "start": text.find(token),
                                "end": text.find(token)+len(token),
                                "text": token
                              }
                            d_n += 1
                        else:
                            spans = []
                            span_token = ''
                            for ts in text.split(token):
                                span_token += ts
                                span_token += token
                                spans.append(len(tokenizer(span_token).input_ids))
                            spans_min = min(spans)
                            spans = []
                            span_token = ''
                            for ts in text.split(token):
                                span_token += ts
                                span_token += token
                                if len(tokenizer(span_token).input_ids) == spans_min:
                                    result = span_token

                            entity["T{}".format(d_n)] = {
                                "eid": "T{}".format(d_n),
                                "label": "ORDERED",
                                "start": len(result)-len(token),
                                "end": len(result),
                                "text": token
                            }
                            d_n += 1
                        ordered = []
                if l == 1:
                    primary.append(input_ids[logitsn])
                if l == 2:
                    symbol.append(input_ids[logitsn])
                if l == 3:
                    ordered.append(input_ids[logitsn])
                last_l = l

            for en in list(entity.values()):
                if text[en['start']:en['end']] != en['text']:
                    print('11111111111111111111111111')

            relation = {}
            data_pp,data_ps,data_ss = [],[],[]
            for key in range(len(list(entity.keys()))):
                for key_2 in range(key+1,len(list(entity.keys()))):
                    k1 = entity[list(entity.keys())[key]]
                    k2 = entity[list(entity.keys())[key_2]]

                    if k1['label'] == 'PRIMARY' and k2['label'] == 'PRIMARY':
                        data_pp.append({'text': text, 'arg0': k1['text'], 'arg1': k2['text'],'arg0_id':k1['eid'],'arg1_id':k2['eid'],
                                     'label': 0})
                    if k1['label'] == 'PRIMARY' and k2['label'] == 'SYMBOL':
                        data_ps.append({'text': text, 'arg0': k1['text'], 'arg1': k2['text'],'arg0_id':k1['eid'],'arg1_id':k2['eid'],
                                     'label': 0})
                    if k1['label'] == 'SYMBOL' and k2['label'] == 'SYMBOL':
                        data_ss.append({'text': text, 'arg0': k1['text'], 'arg1': k2['text'],'arg0_id':k1['eid'],'arg1_id':k2['eid'],
                                     'label': 0})

            d_n = 1
            for d in data_pp:
                text_input = tokenizer(d['arg0'] + ' [SEP] ' + d['arg1'], d['text'],return_tensors='pt')
                input_ids = text_input['input_ids'].to(device)
                attention_mask = text_input['attention_mask'].to(device)
                token_type_ids = text_input['token_type_ids'].to(device)
                output = model_pp(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                output = output.logits.argmax(1).item()
                if output == 1:
                    relation["R{}".format(d_n)]={
                                "rid": "R{}".format(d_n),
                                "label": "Corefer-Description",
                                "arg0": d['arg0_id'],
                                "arg1": d['arg1_id']
                              }
                    d_n+=1

            for d in data_ps:
                text_input = tokenizer(d['arg0'] + ' [SEP] ' + d['arg1'], d['text'],return_tensors='pt')
                input_ids = text_input['input_ids'].to(device)
                attention_mask = text_input['attention_mask'].to(device)
                token_type_ids = text_input['token_type_ids'].to(device)
                output = model_ps(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                output = output.logits.argmax(1).item()
                if output == 1:
                    relation["R{}".format(d_n)]={
                                "rid": "R{}".format(d_n),
                                "label": "Count",
                                "arg0": d['arg0_id'],
                                "arg1": d['arg1_id']
                              }
                    d_n+=1
                if output == 2:
                    relation["R{}".format(d_n)]={
                                "rid": "R{}".format(d_n),
                                "label": "Direct",
                                "arg0": d['arg0_id'],
                                "arg1": d['arg1_id']
                              }
                    d_n+=1
                if output == 3:
                    relation["R{}".format(d_n)]={
                                "rid": "R{}".format(d_n),
                                "label": "Corefer-Symbol",
                                "arg0": d['arg0_id'],
                                "arg1": d['arg1_id']
                              }
                    d_n+=1

            for d in data_ss:
                text_input = tokenizer(d['arg0'] + ' [SEP] ' + d['arg1'], d['text'],return_tensors='pt')
                input_ids = text_input['input_ids'].to(device)
                attention_mask = text_input['attention_mask'].to(device)
                token_type_ids = text_input['token_type_ids'].to(device)
                output = model_ss(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                output = output.logits.argmax(1).item()
                if output == 1:
                    relation["R{}".format(d_n)]={
                                "rid": "R{}".format(d_n),
                                "label": "Corefer-Symbol",
                                "arg0": d['arg0_id'],
                                "arg1": d['arg1_id']
                              }
                    d_n+=1

            da['entity'] = entity
            da['relation'] = relation
            results[da['id']]=da

            with open('result_dev/{}'.format(file),'w',encoding='utf-8') as f:
                json.dump(results,f,indent=4)




