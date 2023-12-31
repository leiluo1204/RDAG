import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import json
import numpy as np
import random
from pandas import DataFrame


class IEMOCAPDataset(Dataset):

    # 数据集名称、数据集拆分、说话人词汇表、情感标签词汇表、args以及分词器。
    # dataset_name 和 split 指定要使用的数据集和数据拆分。
    # speaker_vocab 和 label_vocab 分别是说话人和标签的词汇词典。
    # args 是一个参数字典，tokenizer 是一个 tokenizer 类的实例。
    def __init__(self, dataset_name='IEMOCAP', split='train', speaker_vocab=None, label_vocab=None, args=None,
                 tokenizer=None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.data = self.read(dataset_name, split, tokenizer)
        print(len(self.data))

        self.len = len(self.data)

    # 读取数据集并将其处理成一些对话，其中每个对话都包含了若干个发言者（speakers）、
    # 若干个标签（labels）、若干个文本（utterances）以及若干个特征（features）。
    # 函数返回处理好的对话列表。
    def read(self, dataset_name, split, tokenizer):
        with open('data/%s/%s_data_roberta_v2.json.feature' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)

        # process dialogue
        dialogs = []
        # raw_data = sorted(raw_data, key=lambda x:len(x))
        for d in raw_data:
            # if len(d) < 5 or len(d) > 6:
            #     continue
            utterances = []
            labels = []
            speakers = []
            features = []
            for i, u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                features.append(u['cls'])

            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'features': features
            })
        # 打乱
        random.shuffle(dialogs)
        return dialogs

    # 返回数据集中特定索引的特征、标签、说话人、长度和文本的张量元组。
    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.FloatTensor(self.data[index]['features']), \
               torch.LongTensor(self.data[index]['labels']), \
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances']

    def __len__(self):
        return self.len

    # 每个对话生成一个邻接矩阵。邻接矩阵表示对话中发言人之间的联系。
    # 它通过迭代每个对话中的说话人，并将他们与之前的说话人进行比较来实现。
    # 如果说话人与前一个说话人相同，它就将邻接矩阵中的一个连接设置为1；
    # 如果说话人与前一个说话人不同，它也设置一个连接为1。
    # 然后，它以张量的形式返回所有对话的邻接矩阵。
    def get_adj(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i, s in enumerate(speaker):
                get_local_pred = False
                get_global_pred = False
                for j in range(i - 1, -1, -1):
                    if speaker[j] == s and not get_local_pred:
                        get_local_pred = True
                        a[i, j] = 1
                    elif speaker[j] != s and not get_global_pred:
                        get_global_pred = True
                        a[i, j] = 1
                    if get_global_pred and get_local_pred:
                        break
            adj.append(a)
        return torch.stack(adj)
    # get_adj 和 get_adj_v1 方法用于为每个对话中的说话者图创建邻接矩阵
    def get_adj_v1(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i, s in enumerate(speaker):
                cnt = 0
                for j in range(i - 1, -1, -1):
                    a[i, j] = 1
                    if speaker[j] == s:
                        cnt += 1
                        if cnt == self.args.windowp:
                            break
            adj.append(a)
        return torch.stack(adj)

    # get_s_mask 方法为图中的每个节点创建说话人信息掩码。
    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype=torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1
                        s_onehot[i, j, 1] = 1
                    else:
                        s_onehot[i, j, 0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    # 将样本组合成一个批次。
    # 该方法为特征、标签、speaker_adj、speaker_mask 和长度创建一批张量。
    def collate_fn(self, data):
        '''
        :param data:
            features, labels, speakers, length, utterances
        :return:
            features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''
        max_dialog_len = max([d[3] for d in data])
        feaures = pad_sequence([d[0] for d in data], batch_first=True)  # (B, N, D)
        labels = pad_sequence([d[1] for d in data], batch_first=True, padding_value=-1)  # (B, N )
        adj = self.get_adj_v1([d[2] for d in data], max_dialog_len)
        s_mask, s_mask_onehot = self.get_s_mask([d[2] for d in data], max_dialog_len)
        lengths = torch.LongTensor([d[3] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first=True, padding_value=-1)
        utterances = [d[4] for d in data]

        # spea=[d[2] for d in data]

        return feaures, labels, adj, s_mask, s_mask_onehot, lengths, speakers, utterances