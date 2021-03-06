
import json
import os
#os.chdir('D:\达观杯')
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
from transformers import AutoModelForSequenceClassification,AutoTokenizer,AdamW,get_cosine_schedule_with_warmup,AutoModelForSequenceClassification
#from transformers.deepspeed import HfDeepSpeedConfig
from torch.autograd import Variable
from accelerate import Accelerator
from utils import compute_metrics_task1,compute_metrics_task2
accelerator = Accelerator()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1)
parser.add_argument("--deepspeed_config",default=-1)
args = parser.parse_args()

CFG = { #训练的参数配置
    'fold_num': 5, #五折交叉验证
    'seed': 42,
    'model': 'deberta_v3_large_fintune_by2', #预训练模型
    'max_len': 64, #文本截断的最大长度
    'epochs': 16,
    'train_bs': 16, #batch_size，可根据自己的显存调整
    'valid_bs': 16,
    'lr': 1e-5, #学习率
    'num_workers': 0,
    'accum_iter': 1, #梯度累积，相当于将batch_size*2
    'weight_decay': 1e-4, #权重衰减，防止过拟合
    'device': 0,
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

seed_everything(CFG['seed']) #固定随机种子

torch.cuda.set_device(CFG['device'])
device = accelerator.device



train = pd.read_pickle('en_subtask2_train.pk')
train.extend(pd.read_pickle('en_subtask2_valid.pk'))

train.extend(pd.read_pickle('fr_subtask2_train.pk'))
train.extend(pd.read_pickle('fr_subtask2_valid.pk'))

train.extend(pd.read_pickle('it_subtask2_train.pk'))
train.extend(pd.read_pickle('it_subtask2_valid.pk'))
def get_test(data):
    d = []
    for p in range(len(data)):
        item = data.iloc[p]
        d.append({'sen':item['Sentence'],'label':item['Scores']})
    return d

valid = get_test(pd.read_csv('En-Subtask2-scores.tsv',sep='\t'))
valid_fr = get_test(pd.read_csv('Fr-Subtask2-scores.tsv',sep='\t'))
valid_it = get_test(pd.read_csv('It-Subtask2-scores.tsv',sep='\t'))

true_task1 = []
true_fr = []
true_it = []

for i in valid:
    true_task1.append(i['label'])
for i in valid_fr:
    true_fr.append(i['label'])
for i in valid_it:
    true_it.append(i['label'])
tokenizer = AutoTokenizer.from_pretrained(CFG['model'])

from transformers import DebertaV2PreTrainedModel,DebertaV2Config,DebertaV2Model
class Debertafortask2(DebertaV2PreTrainedModel):
    def __init__(self, config: DebertaV2Config):
        super().__init__(config)
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
    ):
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output.mean(1))[:,0]
        return logits

class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        data = self.df[idx]

        return data

def collate_fn(data):
    input_ids, attention_mask, token_type_ids = [], [], []
    for x in data:
        text = tokenizer(x['sen'], padding='max_length', truncation=True, max_length=CFG['max_len'])
        input_ids.append(text['input_ids'])
        attention_mask.append(text['attention_mask'])
        token_type_ids.append(text['token_type_ids'])

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    label = torch.tensor([(x['label']-1)/6 for x in data])
    return input_ids, attention_mask, token_type_ids,label

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

import time
def train_model(model, fgm,pgd,train_loader,epoch):  # 训练一个epoch
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()
    f1s  = AverageMeter()
    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
    for step, (input_ids, attention_mask, token_type_ids,y) in enumerate(tk):

        output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = criterion(output.float(),y.float())
        scaler.scale(loss).backward()

        """
        fgm.attack()  # 在embedding上添加对抗扰动
        output2 = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss2 = criterion(output2.float(),y.float())
        scaler.scale(loss2).backward()
        fgm.restore()
        """
        if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        losses.update(loss.item(), input_ids.size(0))
        acc = abs(output-y).sum().item() / y.size(0)
        accs.update(acc)
        tk.set_postfix(loss=losses.avg,differ=accs.avg)
        if step == 0:
            log(['开始新一轮训练','现轮数'.format(epoch),'训练起始损失：{}'.format(str(loss.item())),'总batch数：{}'.format(len(tk))],path)

    log(['训练最终损失：{}'.format(str(loss.item())),'训练平均损失：{}'.format(losses.avg),'训练平均相差：{}'.format(accs.avg),'结束本轮训练'],path)
    return losses.avg


def test_model(model, val_loader,epoch):  # 验证
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()
    y_truth, y_pred = [], []
    pred = []
    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
            output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(output.float(), y.float())
            pred.extend(output.clamp(min=0,max=1))
            losses.update(loss.item(), y.size(0))
            acc = abs(output - y).sum().item() / y.size(0)
            accs.update(acc)

            tk.set_postfix(loss=losses.avg, acc=accs.avg)

    log(['测试损失：{}'.format(str(losses.avg)),'测试相差：{}'.format(accs.avg),'结束本轮测试'],path)
    return losses.avg, accs.avg,pred


from typing import List

import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr

def score(submission_file: str, reference_file: str, subtask: str) -> float:
    """Assign an overall score to submitted predictions.

    :param submission_file: str path to submission file with predicted ratings
    :param reference_file: str path to file with gold ratings
    :param subtask: str indicating if the predictions are for the ranking or the classification task
    options: 'ranking' or 'classification'
    :return: float score
    """
    predictions = []
    target = []

    submission = pd.read_csv(
        submission_file, sep="\t", header=None, names=["Id", "Label"]
    )


    reference = pd.read_csv(
        reference_file, sep="\t", header=None, names=["Id", "Label"]
    )
    # the reference file must have the same format as the submission file, so we use the same format checker


    if submission.size != reference.size:
        raise ValueError(
            "Submission does not contain the same number of rows as reference file."
        )

    for _, row in submission.iterrows():
        reference_indices = list(reference["Id"][reference["Id"] == row["Id"]].index)

        if not reference_indices:
            raise ValueError(
                f"Identifier {row['Id']} does not appear in reference file."
            )
        elif len(reference_indices) > 1:
            raise ValueError(
                f"Identifier {row['Id']} appears several times in reference file."
            )
        else:
            reference_index = reference_indices[0]

            if subtask == "ranking":
                target.append(float(reference["Label"][reference_index]))
                predictions.append(float(row["Label"]))
            elif subtask == "classification":
                target.append(reference["Label"][reference_index])
                predictions.append(row["Label"])
            else:
                raise ValueError(
                    f"Evaluation mode {subtask} not available: select ranking or classification"
                )

    if subtask == "ranking":
        score = spearmans_rank_correlation(
            gold_ratings=target, predicted_ratings=predictions
        )


    elif subtask == "classification":
        prediction_ints = convert_class_names_to_int(predictions)
        target_ints = convert_class_names_to_int(target)

        score = accuracy_score(y_true=target_ints, y_pred=prediction_ints)


    else:
        raise ValueError(
            f"Evaluation mode {subtask} not available: select ranking or classification"
        )

    return score


def convert_class_names_to_int(labels: List[str]) -> List[int]:
    """Convert class names to integer label indices.

    :param labels:
    :return:
    """
    class_names = ["IMPLAUSIBLE", "NEUTRAL", "PLAUSIBLE"]
    label_indices = []

    for label in labels:
        try:
            label_index = class_names.index(label)
        except ValueError:
            raise ValueError(f"Label {label} is not in label set {class_names}.")
        else:
            label_indices.append(label_index)

    return label_indices


def spearmans_rank_correlation(
    gold_ratings: List[float], predicted_ratings: List[float]
) -> float:
    """Score submission for the ranking task with Spearman's rank correlation.

    :param gold_ratings: list of float gold ratings
    :param predicted_ratings: list of float predicted ratings
    :return: float Spearman's rank correlation coefficient
    """
    if len(gold_ratings) == 1 and len(predicted_ratings) == 1:
        raise ValueError("Cannot compute rank correlation on only one prediction.")

    return spearmanr(a=gold_ratings, b=predicted_ratings)[0]


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name1='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name1 in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)


    def restore(self, emb_name1='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name1 in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='embeddings',is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)


    def restore(self, emb_name='embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, i, t,f):
        if self.logits:
            #targets_list=t.tolist()
            targets = t
            inputs = i
            #for tar in range(len(targets_list)):
                #for p in range(f[targets_list[tar]]):
                    #targets.append(targets_list[tar])
                    #inputs.append(i[tar].tolist())
            #inputs = torch.tensor(inputs,device=device)
            #targets = torch.tensor(targets)
            targets = torch.eye(35)[targets.reshape(-1)].to(device)
            topic_weight = torch.ones_like(targets) + targets * 6
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False,weight=topic_weight)
        else:
            targets = torch.eye(35)[targets.reshape(-1)].to(device)
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
                                                                           torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

import torch.nn as nn
class CB_loss(nn.Module):
    def __init__(self,beta,gamma,epsilon=0.1):
        super(CB_loss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
    def forward(self,logits, labels,loss_type = 'focal'):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
          labels: A int tensor of size [batch].
          logits: A float tensor of size [batch, no_of_classes].
          samples_per_cls: A python list of size [no_of_classes].
          no_of_classes: total number of classes. int
          loss_type: string. One of "sigmoid", "focal", "softmax".
          beta: float. Hyperparameter for Class balanced loss.
          gamma: float. Hyperparameter for Focal loss.
        Returns:
          cb_loss: A float tensor representing class balanced loss
        """
        # self.epsilon = 0.1 #labelsmooth
        beta = self.beta
        gamma = self.gamma

        no_of_classes = logits.shape[1]
        samples_per_cls = torch.Tensor([sum(labels == i) for i in range(logits.shape[1])])
        if torch.cuda.is_available():
            samples_per_cls = samples_per_cls.cuda()

        effective_num = 1.0 - torch.pow(beta, samples_per_cls)
        weights = (1.0 - beta) / ((effective_num)+1e-8)
        # print(weights)
        weights = weights / torch.sum(weights) * no_of_classes
        labels =labels.reshape(-1,1)

        labels_one_hot  = torch.zeros(len(labels), no_of_classes).scatter_(1, labels.cpu(), 1)

        weights = torch.tensor(weights).float()
        if torch.cuda.is_available():
            weights = weights.cuda()
            labels_one_hot = torch.zeros(len(labels), no_of_classes).cuda().scatter_(1, labels, 1).cuda()

        labels_one_hot = (1 - self.epsilon) * labels_one_hot + self.epsilon / no_of_classes
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,no_of_classes)

        if loss_type == "focal":
            cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
        elif loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, pos_weight = weights)
        elif loss_type == "softmax":
            pred = logits.softmax(dim = 1)
            cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
        return cb_loss

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

    os.mkdir('paperlog/' + log_name)

    with open('paperlog/' + log_name + '/python_file.py', 'a', encoding='utf-8') as f:
        with open(os.path.basename(__file__),'r',encoding='utf-8') as f2:
            file = f2.read()
        f.write(file)
    with open('paperlog/' + log_name + '.py', 'a', encoding='utf-8') as f:
        with open(os.path.basename(__file__),'r',encoding='utf-8') as f2:
            file = f2.read()
        f.write(file)
    path = 'paperlog/' + log_name
    with open(path+'/log.txt', 'a', encoding='utf-8') as f:
        f.write(log_name)
        f.write('\n')
    return path
def macro_f1(pred,gold):
    intersection = 0
    GOLD = 0
    PRED = 0
    intersection1 = 0
    GOLD1 = 0
    PRED1 = 0
    F1s = []
    f = []
    for l in range(3):
        l_p = []
        l_g = []
        for i in range(len(pred)):
            p = pred[i]
            g = gold[i]

            if g == l:
                l_g.append(i)
            if p == l:
                l_p.append(i)

        l_g = set(l_g)
        l_p = set(l_p)

        TP = len(l_g & l_p)
        FP = len(l_p) - TP
        FN = len(l_g) - TP
        precision = TP/(TP+FP+ 0.0000000001)
        recall    = TP/(TP+FN+ 0.0000000001)
        F1        = (2*precision*recall)/(precision+recall+0.0000000001)
        F1s.append(F1)
    return F1s


train_set = MyDataset(train)
valid_set = MyDataset(valid)
valid_fr_set = MyDataset(valid_fr)
valid_it_set = MyDataset(valid_it)
train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True,
                          num_workers=CFG['num_workers'])
valid_loader = DataLoader(valid_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False,
                          num_workers=CFG['num_workers'])
valid_fr_loader = DataLoader(valid_fr_set , batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False,
                          num_workers=CFG['num_workers'])
valid_it_loader = DataLoader(valid_it_set , batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False,
                          num_workers=CFG['num_workers'])
best_acc = 0
#import deepspeed
#ds_config = 'ds_config.json'

model = Debertafortask2.from_pretrained(CFG['model'])  # 模型
model.load_state_dict(torch.load(r'C:\semeval\3\log\deberta_large_fintune_all_fr_en_language_mse_fgm_task2\9_0.8148867491533831_0.8674383673431836_0.7155398328989375_model.pt',map_location='cpu'))
#dschf = HfDeepSpeedConfig(ds_config)

model = model.to(device)
scaler = GradScaler()
#from transformers import Adafactor
#optimizer = Adafactor(model.parameters(),relative_step=False, lr=CFG['lr'], weight_decay=CFG['weight_decay'])  # AdamW优化器
optimizer = AdamW(model.parameters(),lr=CFG['lr'], weight_decay=CFG['weight_decay'])
criterion = nn.MSELoss()
#criterion = FocalLoss()
scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // CFG['accum_iter'],
                                            CFG['epochs'] * len(train_loader) // CFG['accum_iter'])
# get_cosine_schedule_with_warmup策略，学习率先warmup一个epoch，然后cos式下降
fgm = FGM(model)
pgd = PGD(model)

train_loader,val_loader,valid_fr_loader,valid_it_loader = accelerator.prepare(train_loader,valid_loader,valid_fr_loader,valid_it_loader)
#model,optimizer,_,scheduler = deepspeed.initialize(model=model,config_params=ds_config,optimizer=optimizer,lr_scheduler=scheduler)

log_name = 'deberta_byfintune_task2'
path = log_start(log_name)

for epoch in range(CFG['epochs']):

    train_loss = train_model(model, fgm, pgd, train_loader,epoch)
    val_loss, val_acc, pred = test_model(model, val_loader,epoch)
    log_1,l1 = compute_metrics_task2((torch.stack(pred).cpu().numpy()*6)+1,true_task1)
    log(['English']+log_1,path)
    val_loss, val_acc, pred = test_model(model, valid_fr_loader, epoch)
    log_fr,lfr = compute_metrics_task2((torch.stack(pred).cpu().numpy()*6)+1,true_fr)
    log(['Frensh']+log_fr,path)
    val_loss, val_acc, pred = test_model(model, valid_it_loader, epoch)
    log_it,lit = compute_metrics_task2((torch.stack(pred).cpu().numpy()*6)+1,true_it)
    log(['It']+log_it,path)
    model_name = path+'/{}_{}_{}_{}_model.pt'.format(epoch,l1['rho'],lfr['rho'],lit['rho'])
    torch.save(model.state_dict(),model_name)
    log(['保存模型:{}'.format(model_name)],path)

