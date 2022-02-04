import pandas as pd
import os
import random
import collections

def get_data(df):
    data = []
    for i in range(len(df)):
        data.append({'sen':df.iloc[i]['Sentence'],'label':df.iloc[i]['Labels']})
    return data

def get_word(data):
    words = []
    for i in data:
        words.extend(i['sen'].split(' '))
    return words

def gatas(en_words,en_subtask1,datas):
    en_words = collections.Counter(en_words)
    en_keywords = []

    for i in en_words.items():
        if i[1] <= 100:
            en_keywords.append(i[0])
    for i in en_subtask1:
        i = i['sen']
        key_n = 0
        key_w = []
        for p in en_keywords:
            if ' '+p+' ' in i:
                key_n += 1
                key_w.append(p)
        if key_n == 2:
            if key_w[0] not in key_w[1] and key_w[1] not in key_w[0]:
                datas.append({'sen': i, 'label': key_w})
    return datas

datas = []
en_subtask1 = pd.read_csv('train_subtask-1/en/En-Subtask1-fold_0.tsv',sep='\t')
en_subtask1 = en_subtask1.append(pd.read_csv('train_subtask-1/en/En-Subtask1-fold_1.tsv',sep='\t'))
en_subtask1 = en_subtask1.append(pd.read_csv('train_subtask-1/en/En-Subtask1-fold_2.tsv',sep='\t'))
en_subtask1 = get_data(en_subtask1)
en_words = get_word(en_subtask1)
datas = gatas(en_words,en_subtask1,datas)


fr_subtask1 = pd.read_csv('train_subtask-1/fr/Fr-Subtask1-fold_0.tsv',sep='\t')
fr_subtask1 = fr_subtask1.append(pd.read_csv('train_subtask-1/fr/Fr-Subtask1-fold_1.tsv',sep='\t'))
fr_subtask1 = fr_subtask1.append(pd.read_csv('train_subtask-1/fr/Fr-Subtask1-fold_2.tsv',sep='\t'))
fr_subtask1 = get_data(fr_subtask1)
fr_words = get_word(fr_subtask1)
datas = gatas(fr_words,fr_subtask1,datas)

it_subtask1 = pd.read_csv('train_subtask-1/it/It-Subtask1-fold_0.tsv',sep='\t')
it_subtask1 = it_subtask1.append(pd.read_csv('train_subtask-1/it/It-Subtask1-fold_1.tsv',sep='\t'))
it_subtask1 = it_subtask1.append(pd.read_csv('train_subtask-1/it/It-Subtask1-fold_2.tsv',sep='\t'))
it_subtask1 = get_data(it_subtask1)
it_words = get_word(it_subtask1)
datas = gatas(it_words,it_subtask1,datas)

def get_data(df):
    data = []
    for i in range(len(df)):
        data.append({'sen':df.iloc[i]['Sentence'],'label':1})
    return data
en = pd.read_csv(r'test_subtask-1\En-Subtask1-test.tsv',sep='\t')
en_test = get_data(en)
en_test_words = get_word(en_test)
datas = gatas(en_test_words,en_test,datas)
fr = pd.read_csv(r'test_subtask-1\Fr-Subtask1-test.tsv',sep='\t')
fr_test = get_data(fr)
fr_test_words = get_word(fr_test)
datas = gatas(fr_test_words,fr_test,datas)
it = pd.read_csv(r'test_subtask-1\It-Subtask1-test.tsv',sep='\t')
it_test = get_data(it)
it_test_words = get_word(it_test)
datas = gatas(it_test_words,it_test,datas)
random.shuffle(datas)
pd.to_pickle(datas[:1000],r'ner_valid.pk')
pd.to_pickle(datas[1000:],r'ner_train.pk')