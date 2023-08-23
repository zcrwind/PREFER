import numpy as np
import pandas as pd
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from src.utils import ROOT_DIR
dataset_transform = {'sst':'SST-2','rte':'RTE','snli':'SNLI','mnli':'MNLI','qnli': 'QNLI', 'trec':'trec'}

def split_lines(lines, valid_ratio = 0.1):
    total_num = len(lines)
    rand_order = np.random.permutation(total_num)
    train_lines = []
    valid_lines = []
    split_point = int(total_num * (1 - valid_ratio))
    for idx in rand_order[:split_point]:
        train_lines.append(lines[idx])
    for idx in rand_order[split_point:]:
        valid_lines.append(lines[idx])
    return train_lines, valid_lines

def write_lines(lines, header = None, path = None):
    with open(path, 'w', encoding = 'utf-8') as f:
        if header != None:
            f.write(header)
        for line in lines:
            f.write(line)

# dataset_list = ['MNLI', 'QNLI','RTE','trec']
dataset_list = ['mr','trec','RTE']
for dataset_name in dataset_list:
    dataset_path = ROOT_DIR + f'datasets/original/{dataset_name}/'
    suffix = 'tsv' if dataset_name in ['MNLI', 'QNLI','RTE'] else 'csv'
    with open(dataset_path + 'train.' + suffix, 'r', encoding = 'utf-8') as f:
        train_lines = f.readlines()
    header = train_lines[0]
    resplit_train_lines, resplit_valid_lines = split_lines(train_lines[1:])

    if dataset_name == 'MNLI':
        dev_name = 'dev_matched.'
    elif dataset_name == 'trec':
        dev_name = 'test.'
    else:
        dev_name = 'dev.' 

    with open(dataset_path + dev_name + suffix, 'r', encoding = 'utf-8') as f:
        dev_lines = f.readlines()

    write_path = ROOT_DIR + f'datasets/full_dataset/{dataset_name}/'
    if not os.path.exists(write_path):
        os.mkdir(write_path)
    print(len(resplit_train_lines), len(resplit_valid_lines), len(dev_lines))
    write_lines(resplit_train_lines, header, write_path + 'train.' + suffix)
    write_lines(resplit_valid_lines, header, write_path + dev_name + suffix)
    write_lines(dev_lines, None, write_path + 'test.' + suffix)



