import pandas as pd
import os
os.chdir(r'C:\semeval\7')

data = pd.read_csv('ClarificationTask_TrainData_Sep23.tsv',sep='\t')
label = pd.read_csv('ClarificationTask_TrainLabels_Sep23.tsv',sep='\t',header=None)
score = pd.read_csv('ClarificationTask_TrainScores_Nov11.tsv',sep='\t',header=None)

n = 0
datas = []
for i in range(len(data)):
    item = data.iloc[i]
    for f in range(1, 6):
        datas.append({'word': item['Filler{}'.format(f)], 'sen': str(
            str(item['Article title']) + '. ' + str(item['Section header']) + '. ' + str(
                item['Previous context']) + ' ' + str(item['Sentence'].replace('______', '[MASK]')) + ' ' + str(
                item['Follow-up context'])).replace('Nan', ''),
                      'label': {'PLAUSIBLE': 2, 'NEUTRAL': 1, 'IMPLAUSIBLE': 0}[label.iloc[n][1]]})
        n += 1
pd.to_pickle(datas,'train_label.pk')
n = 0
datas = []
for i in range(len(data)):
    item = data.iloc[i]
    word = []
    label = []
    for f in range(1, 6):
        word.append(item['Filler{}'.format(f)])
        label.append(score.iloc[n][1])
        n += 1
    datas.append({'word': word, 'sen': str(
        str(item['Article title']) + '. ' + str(item['Section header']) + '. ' + str(
            item['Previous context']) + ' ' + str(item['Sentence'].replace('______', '[MASK]')) + ' ' + str(
            item['Follow-up context'])).replace('Nan', ''),
                  'label': np.array(label).argmax()})

pd.to_pickle(datas,'train_score.pk')

data = pd.read_csv('ClarificationTask_DevData_Oct22a.tsv',sep='\t')
label = pd.read_csv('ClarificationTask_DevLabels_Oct22a.tsv',sep='\t',header=None)
score = pd.read_csv('ClarificationTask_DevScores_Oct22a.tsv',sep='\t',header=None)

n = 0
datas = []
for i in range(len(data)):
    item = data.iloc[i]
    for f in range(1, 6):
        datas.append({'word': item['Filler{}'.format(f)], 'sen': str(
            str(item['Article title']) + '. ' + str(item['Section header']) + '. ' + str(
                item['Previous context']) + ' ' + str(item['Sentence'].replace('______', '[MASK]')) + ' ' + str(
                item['Follow-up context'])).replace('Nan', ''),
                      'label': {'PLAUSIBLE': 2, 'NEUTRAL': 1, 'IMPLAUSIBLE': 0}[label.iloc[n][1]]})
        n += 1
pd.to_pickle(datas,'dev_label.pk')

n = 0
datas = []
for i in range(len(data)):
    item = data.iloc[i]
    word = []
    label = []
    for f in range(1, 6):
        word.append(item['Filler{}'.format(f)])
        label.append(score.iloc[n][1])
        n += 1
    datas.append({'word': word, 'sen': str(
        str(item['Article title']) + '. ' + str(item['Section header']) + '. ' + str(
            item['Previous context']) + ' ' + str(item['Sentence'].replace('______', '[MASK]')) + ' ' + str(
            item['Follow-up context'])).replace('Nan', ''),
                  'label': np.array(label).argmax()})

pd.to_pickle(datas,'dev_score.pk')