import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
import argparse
import time
from sklearn.model_selection import *
from transformers import *
from torch.autograd import Variable
import pkuseg
from transformers import DebertaV2Tokenizer
from model import CPTForConditionalGeneration
from accelerate import Accelerator
# from evaluate_new import *
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import BertTokenizer, BartForConditionalGeneration
accelerator = Accelerator()
device = accelerator.device
CFG = {  # 训练的参数配置
    'fold_num': 5,  # 五折交叉验证
    'seed': 2022,
    'model': r'C:\semeval\1\new\CPTsave\5_cpt_4.368918971734954_5.213990181603111',  # 预训练模型
    'max_len': 150,  # 文本截断的11最大长度
    'epochs': 8,
    'train_bs': 6,  # batch_size，可根据自己的显存调整
    'valid_bs': 50,
    'lr': 1e-5,  # 学习率
    'num_workers': 0,
    'accum_iter': 8,  # 梯度累积，相当于将batch_size*2
    'weight_decay': 1e-4,  # 权重衰减，防止过拟合
    'device': 0,
    'ues_r_drop': True,
    'train':False
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


seed_everything(CFG['seed'])  # 固定随机种子

torch.cuda.set_device(CFG['device'])

tokenizer = DebertaV2Tokenizer.from_pretrained(CFG['model'])

def get_data(filename):
    with open(filename,'r',encoding='utf-8') as f:
        data = json.load(f)
    return data

train = get_data('train/en.train.json')
train.extend(get_data('train/fr.train.json'))
train.extend(get_data('train/ru.train.json'))
train.extend(get_data('train/es.train.json'))
train.extend(get_data('train/it.train.json'))



def get_refs(data):
    refs = []
    for i in data:
        refs.append(i['gloss'])
    return refs
dev_en = get_data('train/en.dev.json')
dev_es = get_data('train/es.dev.json')
dev_fr = get_data('train/fr.dev.json')
dev_it = get_data('train/it.dev.json')
dev_ru = get_data('train/ru.dev.json')

dev_en_refs = get_refs(dev_en)
dev_es_refs = get_refs(dev_es)
dev_fr_refs = get_refs(dev_fr)
dev_it_refs = get_refs(dev_it)
dev_ru_refs = get_refs(dev_ru)


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
        text = tokenizer(x['gloss'], padding='max_length', truncation=True, max_length=110, return_tensors='pt')
        label = text['input_ids']
        label = torch.where(label == tokenizer.pad_token_id, -100, label)
        labels.append(label)
    input_ids = torch.tensor(input_ids)
    labels = torch.stack(labels)
    return input_ids,labels


class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
import collections
#from nltk import word_tokenize as tokenize2
def get_score(submission,reference):
    # 1. read contents
    ## define accumulators for lemma-level BLEU and MoverScore
    reference_lemma_groups = collections.defaultdict(list)
    all_preds, all_tgts = [], []

    id_to_lemma = {}
    pbar = tqdm.tqdm(total=len(submission), desc="S-BLEU", disable=None)
    for sub, ref in zip(submission, reference):
        assert sub["id"] == ref["id"], "Mismatch in submission and reference files!"
        all_preds.append(sub)
        all_tgts.append(ref)
        sub = tokenize2(sub)
        ref = tokenize2(ref)
        sub["sense-BLEU"] = bleu([sub["gloss"]], ref["gloss"])
        reference_lemma_groups[(ref["word"], ref["pos"])].append(ref["gloss"])
        id_to_lemma[sub["id"]] = (ref["word"], ref["pos"])
        pbar.update()
    pbar.close()
    ## compute lemma-level BLEU
    for sub in tqdm.tqdm(submission, desc="L-BLEU", disable=None):
        sub["lemma-BLEU"] = max(
            bleu([sub["gloss"]], g)
            for g in reference_lemma_groups[id_to_lemma[sub["id"]]]
        )
    lemma_bleu_average = sum(s["lemma-BLEU"] for s in submission) / len(submission)
    sense_bleu_average = sum(s["sense-BLEU"] for s in submission) / len(submission)
    ## compute MoverScore
    # moverscore_average = np.mean(mv_sc.word_mover_score(
    #     all_tgts,
    #     all_preds,
    #     collections.defaultdict(lambda:1.),
    #     collections.defaultdict(lambda:1.),
    #     stop_words=[],
    #     n_gram=1,
    #     remove_subwords=False,
    #     batch_size=1,
    # ))
    moverscore_average = mover_corpus_score(all_preds, [all_tgts])
    # 3. write results.
    # logger.debug(f"Submission {args.submission_file}, \n\tMvSc.: " + \
    #     f"{moverscore_average}\n\tL-BLEU: {lemma_bleu_average}\n\tS-BLEU: " + \
    #     f"{sense_bleu_average}"
    # )
    with open(args.output_file, "a") as ostr:
        print(f"MoverScore_{summary.lang}:{moverscore_average}", file=ostr)
        print(f"BLEU_lemma_{summary.lang}:{lemma_bleu_average}", file=ostr)
        print(f"BLEU_sense_{summary.lang}:{sense_bleu_average}", file=ostr)
    return (
        args.submission_file,
        moverscore_average,
        lemma_bleu_average,
        sense_bleu_average,
    )
def train_model(model,train_loader):  # 训练一个epoch
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()
    f1s = AverageMeter()
    K = 3
    # optimizer.zero_grad()

    tk = tqdm(train_loader, desc='dirs')
    print(device)
    optimizer.zero_grad()

    for step, (input_ids,labels) in enumerate(tk):

        with autocast():  # 使用半精度训练
            output = model(input_ids,labels=labels)

            loss = output.loss
            scaler.scale(loss).backward()

        if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
            scheduler.step()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        losses.update(loss.item())

        tk.set_postfix(loss=losses.avg)

    return losses.avg

def test_model(model,valid_loader,valid_refs):
    model.eval()
    tk = tqdm(valid_loader)
    outs = []
    with torch.no_grad():
        for inputs in tk:
            output = tokenizer.batch_decode(model.generate(inputs))
            outs.extend(output)
    score = get_score(outs,valid_refs)
    return score

def log(text,path):
    with open(path+'/log.txt','a',encoding='utf-8') as f:
        f.write('-----------------{}-----------------'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        for i in text:
            f.write(i)
            print(i)
            f.write('\n')
        f.write('\n')
import os
def log_start(log_name):
    if log_name == '':
        log_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    try:
        os.mkdir('new/log/' + log_name)
    except:
        log_name = log_name + time.strftime("_%Y_%m_%d_%H_%M_%S", time.localtime())
        os.mkdir('new/log/' + log_name)

    with open('new/log/' + log_name + '/python_file.py', 'a', encoding='utf-8') as f:
        with open(os.path.basename(__file__),'r',encoding='utf-8') as f2:
            file = f2.read()
        f.write(file)

    path = 'new/log/' + log_name
    with open(path+'/log.txt', 'a', encoding='utf-8') as f:
        f.write('log_name')
        f.write('\n')
    return path
train_set = MyDataset(train)

model = CPTForConditionalGeneration.from_pretrained(CFG['model'] ).to(device)  # 模型
#model.load_state_dict(torch.load('4_T5_1.6154526329307242.pt'),strict=True)


scaler = GradScaler()
from transformers import Adafactor

optimizer = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])  # AdamW优化器
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True,
                          num_workers=CFG['num_workers'])
scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // CFG['accum_iter'],
                                            CFG['epochs'] * len(train_loader) // CFG['accum_iter'])
fgm = 'FGM(model)'
pgd = 'PGD(model)'


log_name = 'eval_defmod_all_'+CFG['vectors']
path = log_start(log_name)


for epoch in range(CFG['epochs']):

    train_loss1 = train_model(model, train_loader)
    valid_score = test_model(model,valid_loader)
    score = test_model(model,valid_loader)

    path2 = '{}/{}_cpt_{}'.format(str(path),str(epoch), str(score))
    os.mkdir(path2)
    model.save_pretrained(path2)
    tokenizer.save_pretrained(path2)
