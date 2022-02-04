import pandas as pd
import os
import random


def get_data(df):
    data = []
    valid = []
    for i in range(len(df)):
        if df.iloc[i]['ID']%8==0:
            valid.append({'sen':df.iloc[i]['Sentence'],'label':df.iloc[i]['Labels']})
        else:
            data.append({'sen':df.iloc[i]['Sentence'],'label':df.iloc[i]['Labels']})
    return data,valid

en_subtask1 = pd.read_csv('train_subtask-1/en/En-Subtask1-fold_0.tsv',sep='\t')
en_subtask1 = en_subtask1.append(pd.read_csv('train_subtask-1/en/En-Subtask1-fold_1.tsv',sep='\t'))
en_subtask1 = en_subtask1.append(pd.read_csv('train_subtask-1/en/En-Subtask1-fold_2.tsv',sep='\t'))
en_subtask1,en_subtask1_valid = get_data(en_subtask1)

fr_subtask1 = pd.read_csv('train_subtask-1/fr/Fr-Subtask1-fold_0.tsv',sep='\t')
fr_subtask1 = fr_subtask1.append(pd.read_csv('train_subtask-1/fr/Fr-Subtask1-fold_1.tsv',sep='\t'))
fr_subtask1 = fr_subtask1.append(pd.read_csv('train_subtask-1/fr/Fr-Subtask1-fold_2.tsv',sep='\t'))
fr_subtask1,fr_subtask1_valid = get_data(fr_subtask1)

it_subtask1 = pd.read_csv('train_subtask-1/it/It-Subtask1-fold_0.tsv',sep='\t')
it_subtask1 = it_subtask1.append(pd.read_csv('train_subtask-1/it/It-Subtask1-fold_1.tsv',sep='\t'))
it_subtask1 = it_subtask1.append(pd.read_csv('train_subtask-1/it/It-Subtask1-fold_2.tsv',sep='\t'))
it_subtask1,it_subtask1_valid = get_data(it_subtask1)

random.shuffle(en_subtask1)
random.shuffle(fr_subtask1)
random.shuffle(it_subtask1)


pd.to_pickle(en_subtask1_valid,'en_subtask1_valid.pk')
pd.to_pickle(en_subtask1,'en_subtask1_train.pk')

pd.to_pickle(fr_subtask1_valid,'fr_subtask1_valid.pk')
pd.to_pickle(fr_subtask1,'fr_subtask1_train.pk')

pd.to_pickle(it_subtask1_valid,'it_subtask1_valid.pk')
pd.to_pickle(it_subtask1,'it_subtask1_train.pk')




#-------------------------------------------------------------

def get_data(df):
    data = []
    valid = []
    for i in range(len(df)):
        if df.iloc[i]['ID']%8==0:
            valid.append({'sen':df.iloc[i]['Sentence'],'label':df.iloc[i]['Score']})
        else:
            data.append({'sen':df.iloc[i]['Sentence'],'label':df.iloc[i]['Score']})
    return data,valid

en_subtask2 = pd.read_csv('train_subtask-2/en/En-Subtask2-fold_0.tsv',sep='\t')
en_subtask2 = en_subtask2.append(pd.read_csv('train_subtask-2/en/En-Subtask2-fold_1.tsv',sep='\t'))
en_subtask2,en_subtask2_valid = get_data(en_subtask2)

fr_subtask2 = pd.read_csv('train_subtask-2/fr/Fr-Subtask2-fold_0.tsv',sep='\t')
fr_subtask2 = fr_subtask2.append(pd.read_csv('train_subtask-2/fr/Fr-Subtask2-fold_1.tsv',sep='\t'))
fr_subtask2,fr_subtask2_valid = get_data(fr_subtask2)

it_subtask2 = pd.read_csv('train_subtask-2/it/It-Subtask2-fold_0.tsv',sep='\t')
it_subtask2 = it_subtask2.append(pd.read_csv('train_subtask-2/it/It-Subtask2-fold_1.tsv',sep='\t'))
it_subtask2,it_subtask2_valid = get_data(it_subtask2)

random.shuffle(en_subtask2)
random.shuffle(fr_subtask2)
random.shuffle(it_subtask2)

pd.to_pickle(en_subtask2_valid,'en_subtask2_valid.pk')
pd.to_pickle(en_subtask2,'en_subtask2_train.pk')

pd.to_pickle(fr_subtask2_valid,'fr_subtask2_valid.pk')
pd.to_pickle(fr_subtask2,'fr_subtask2_train.pk')

pd.to_pickle(it_subtask2_valid,'it_subtask2_valid.pk')
pd.to_pickle(it_subtask2,'it_subtask2_train.pk')



