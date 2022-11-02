# Semeval Task-1



##### 本存储库主要包含semeval-task1的主要方法与代码

##### 使用随机初始化的mdeberta模型，首先在五份不同语种的文本数据集中进行MLM预训练，之后和自己对比，进行Simcse预训练，在使用矩阵数据集微调模型之后，通过掩码自回归预训练，来训练decoder层，最后在生成的数据集中微调100epoch

#### 流程

1. score.py是官方文件的评分代码；
2. read.py 读取数据集；
3. pre_mask.py中，使用WWM去预训练模型
4. pre_simtrain.py,将WWM预训练之后的模型，使用simcse进行对比学习预训练；
5. pre_cpt.py 基于simcse模型，加入Decoder层，使用掩码自回归预训练；
6. train_sim.py 基于simcse模型，在**Reverse Dictionary**训练集中微调
7. train_cpt.py 基于cpt模型，在**Definition Modeling** 训练集中进行微调
8. test_cpt.py和test_sim.py分别是Reverse Dictionary与Definition Modeling的生成结果代码；

## Cite
```
@inproceedings{li-etal-2022-lingjing,
    title = "{L}ing{J}ing at {S}em{E}val-2022 Task 1: Multi-task Self-supervised Pre-training for Multilingual Reverse Dictionary",
    author = "Li, Bin  and
      Weng, Yixuan  and
      Xia, Fei  and
      He, Shizhu  and
      Sun, Bin  and
      Li, Shutao",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.semeval-1.4",
    doi = "10.18653/v1/2022.semeval-1.4",
    pages = "29--35",
    abstract = "This paper introduces the approach of Team LingJing{'}s experiments on SemEval-2022 Task 1 Comparing Dictionaries and Word Embeddings (CODWOE). This task aims at comparing two types of semantic descriptions and including two sub-tasks: the definition modeling and reverse dictionary track. Our team focuses on the reverse dictionary track and adopts the multi-task self-supervised pre-training for multilingual reverse dictionaries. Specifically, the randomly initialized mDeBERTa-base model is used to perform multi-task pre-training on the multilingual training datasets. The pre-training step is divided into two stages, namely the MLM pre-training stage and the contrastive pre-training stage. The experimental results show that the proposed method has achieved good performance in the reverse dictionary track, where we rank the 1-st in the Sgns targets of the EN and RU languages. All the experimental codes are open-sourced at https://github.com/WENGSYX/Semeval.",
}

```
