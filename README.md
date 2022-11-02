# Semeval 2022

#### In this folder, the solution of semeval is stored
#### 在本存储库中，包含着Semeval 2022部分任务的代码
### TASK LIST（任务列表）

- [TASK1](https://github.com/WENGSYX/Semeval/tree/main/1)
- [TASK3](https://github.com/WENGSYX/Semeval/tree/main/3)
- [TASK6](https://github.com/WENGSYX/Semeval/tree/main/6)
- [TASK7](https://github.com/WENGSYX/Semeval/tree/main/7)
- [TASK8](https://github.com/WENGSYX/Semeval/tree/main/8)
- [TASK12](https://github.com/WENGSYX/Semeval/tree/main/12)



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


@inproceedings{xia-etal-2022-lingjing,
    title = "{L}ing{J}ing at {S}em{E}val-2022 Task 3: Applying {D}e{BERT}a to Lexical-level Presupposed Relation Taxonomy with Knowledge Transfer",
    author = "Xia, Fei  and
      Li, Bin  and
      Weng, Yixuan  and
      He, Shizhu  and
      Sun, Bin  and
      Li, Shutao  and
      Liu, Kang  and
      Zhao, Jun",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.semeval-1.30",
    doi = "10.18653/v1/2022.semeval-1.30",
    pages = "239--246",
    abstract = "This paper presents the results and main findings of our system on SemEval-2022 Task 3 Presupposed Taxonomies: Evaluating Neural Network Semantics (PreTENS). This task aims at semantic competence with specific attention on the evaluation of language models, which is a task with respect to the recognition of appropriate taxonomic relations between two nominal arguments. Two sub-tasks including binary classification and regression are designed for the evaluation. For the classification sub-task, we adopt the DeBERTa-v3 pre-trained model for fine-tuning datasets of different languages. Due to the small size of the training datasets of the regression sub-task, we transfer the knowledge of classification model (i.e., model parameters) to the regression task. The experimental results show that the proposed method achieves the best results on both sub-tasks. Meanwhile, we also report negative results of multiple training strategies for further discussion. All the experimental codes are open-sourced at https://github.com/WENGSYX/Semeval.",
}
```
