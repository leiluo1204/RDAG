from dataset import *
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
from  transformers import BertTokenizer

def get_train_valid_sampler(trainset):
    size = len(trainset)
    idx = list(range(size))
    # SubsetRandomSampler 随机数据采样器，利用下标来提取dataset中的数据方法
    return SubsetRandomSampler(idx)


def load_vocab(dataset_name):
    # 使用pickle.load（）从数据目录中的pickle文件加载说话者词汇表、标签词汇表和个性向量。
    speaker_vocab = pickle.load(open('data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))
    label_vocab = pickle.load(open('data/%s/label_vocab.pkl' % (dataset_name), 'rb'))
    person_vec_dir = 'data/%s/person_vect.pkl' % (dataset_name)
    # if os.path.exists(person_vec_dir):
    #     print('Load person vec from ' + person_vec_dir)
    #     person_vec = pickle.load(open(person_vec_dir, 'rb'))
    # else:
    #     print('Creating personality vectors')
    #     person_vec = np.random.randn(len(speaker_vocab['itos']), 100)a
    #     print('Saving personality vectors to' + person_vec_dir)
    #     with open(person_vec_dir,'wb') as f:
    #         pickle.dump(person_vec, f, -1)
    person_vec = None

    return speaker_vocab, label_vocab, person_vec

# dataset_name：数据集的名称（默认为“IEMOCAP”
# batch_size：每个批次的大小（默认值为32）
# num_workers：用于数据加载的子进程数（默认值为0）
# pin_memory：是否将固定内存用于GPU数据传输（默认值为False）
# args：要传递给IEMOCAPDataset的其他参数。
def get_IEMOCAP_loaders(dataset_name = 'IEMOCAP', batch_size=32, num_workers=0, pin_memory=False, args = None):
    print('building vocab.. ')
    speaker_vocab, label_vocab, person_vec = load_vocab(dataset_name)
    print('building datasets..')
    trainset = IEMOCAPDataset(dataset_name, 'train',  speaker_vocab, label_vocab, args)
    devset = IEMOCAPDataset(dataset_name, 'dev', speaker_vocab, label_vocab, args)
    # 随机采样
    train_sampler = get_train_valid_sampler(trainset)
    valid_sampler = get_train_valid_sampler(devset)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(devset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=devset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(dataset_name, 'test',  speaker_vocab, label_vocab, args)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, person_vec

def get_IEMOCAP_loaders_v2(dataset_name = 'IEMOCAP', batch_size=32, num_workers=0, pin_memory=False, args = None):
    speaker_vocab, label_vocab, person_vec = load_vocab(dataset_name)
    testset = IEMOCAPDataset(dataset_name, 'test',  speaker_vocab, label_vocab, args)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return test_loader, speaker_vocab, label_vocab, person_vec