from ctypes import Union
from typing import List
from .utils import ROOT_DIR, FEWSHOT_PATH
import datasets
from datasets import Dataset, DatasetDict
import numpy as np
import pandas as pd
import os
import shutil

dataset_transform = {'sst':'SST-2','rte':'RTE','snli':'SNLI','mnli':'MNLI','qnli': 'QNLI', 'trec':'trec','imdb':'imdb','sst-5':'SST-5', 'agnews':'agnews', 'mr':'mr', 'cr':'cr','mpqa':'mpqa','subj':'subj','cola':'CoLA','mnli-mm':'MNLI','mrpc':'MRPC','qqp':'QQP'}

def load_dataset(dataset_name = 'sst', sort_dataset = False, fewshot = False, k = 0, rand_seed = 0, use_valid_for_train = False,
                low_resource = False):
    assert not (fewshot and low_resource), "fewshot and low_resouce cannot be selected simutaneously!"
    if fewshot:
        assert k > 0, f"k must > 0, found {k}"
        if k > 16:
            assert sort_dataset, "sort the dataset before training for acceleration."
        dataset_path = os.path.join(ROOT_DIR, f'datasets/k-shot/{dataset_transform[dataset_name]}/{k}-{rand_seed}/')
    elif low_resource:
        assert not use_valid_for_train, "cannot use valid for training in low resource setting!"
        dataset_path = os.path.join(ROOT_DIR, f'datasets/low-resource-16valid/{dataset_transform[dataset_name]}/{k}-{rand_seed}/')
    else:
        dataset_path = os.path.join(ROOT_DIR, f'datasets/full_dataset/{dataset_transform[dataset_name]}/')
    if dataset_name in ['snli','mnli','qnli']:
        assert (fewshot or low_resource), "NLI dataset (except RTE) does not support full-data training!"
    if dataset_name == 'sst':
        train_dataset, valid_dataset, test_dataset = load_dataset_sst(dataset_path, sort_dataset = sort_dataset)
    elif dataset_name == 'sst5':
        train_dataset, valid_dataset, test_dataset = load_dataset_sst5(dataset_path, sort_dataset = sort_dataset)
    elif dataset_name == 'rte':
        train_dataset, valid_dataset, test_dataset = load_dataset_rte(dataset_path, sort_dataset = sort_dataset)
    elif dataset_name == 'mnli':
        train_dataset, valid_dataset, test_dataset = load_dataset_mnli(dataset_path, sort_dataset = sort_dataset)
    elif dataset_name == 'qnli':
        train_dataset, valid_dataset, test_dataset = load_dataset_qnli(dataset_path, sort_dataset = sort_dataset)
    elif dataset_name == 'snli':
        train_dataset, valid_dataset, test_dataset = load_dataset_snli(dataset_path, sort_dataset = sort_dataset)
    elif dataset_name == 'trec':
        train_dataset, valid_dataset, test_dataset = load_dataset_trec(dataset_path, sort_dataset = sort_dataset)
    elif dataset_name == 'agnews':
        train_dataset, valid_dataset, test_dataset = load_dataset_agnews(dataset_path, sort_dataset = sort_dataset)
    elif dataset_name == 'mr':
        train_dataset, valid_dataset, test_dataset = load_dataset_mr(dataset_path, sort_dataset = sort_dataset)
    else:
        raise NotImplementedError

    if use_valid_for_train:
        combined_train_xs = train_dataset[0] + valid_dataset[0]
        combined_train_ys = train_dataset[1] + valid_dataset[1]
        train_dataset = (combined_train_xs, combined_train_ys)
        valid_dataset = []

    if sort_dataset:
        sorted_train_dataset = sort_dataset_via_length(train_dataset)
        if not use_valid_for_train:
            sorted_valid_dataset = sort_dataset_via_length(valid_dataset)
        else:
            sorted_valid_dataset = []
        sorted_test_dataset = sort_dataset_via_length(test_dataset)
        return sorted_train_dataset, sorted_valid_dataset, sorted_test_dataset
    return train_dataset, valid_dataset, test_dataset


'''
developed by zcr, for rte, snli, mnli, qnli datasets.
temp version: use preprocessed sub-dataset by liulin.
'''
def load_dataset_zcr(
    dataset_name='rte',
    sort_dataset=False,
    fewshot=False,
    k=0,
    rand_seed=0,
    use_valid_for_train=False
):
    if fewshot:
        assert k > 0, f"k must > 0, found {k}"
        if k > 16:
            assert sort_dataset, "sort the dataset before training for acceleration."
        dataset_path = os.path.join(ROOT_DIR, f'datasets/k-shot/{dataset_transform[dataset_name]}/{k}-{rand_seed}/')
    else:
        dataset_path = os.path.join(ROOT_DIR, f'datasets/full_dataset/{dataset_transform[dataset_name]}/')
    if dataset_name in ['snli', 'mnli', 'qnli']:
        assert fewshot, "NLI dataset (except RTE) does not support full-data training!"
    if dataset_name == 'sst':
        train_dataset, valid_dataset, test_dataset = load_dataset_sst(dataset_path, sort_dataset = sort_dataset)
    elif dataset_name == 'sst5':
        train_dataset, valid_dataset, test_dataset = load_dataset_sst5(dataset_path, sort_dataset = sort_dataset)
    elif dataset_name == 'rte':
        train_dataset, valid_dataset, test_dataset = load_dataset_rte(dataset_path, sort_dataset = sort_dataset)
    elif dataset_name == 'mnli':
        train_dataset, valid_dataset, test_dataset = load_dataset_mnli(dataset_path, sort_dataset = sort_dataset)
    elif dataset_name == 'qnli':
        train_dataset, valid_dataset, test_dataset = load_dataset_qnli(dataset_path, sort_dataset = sort_dataset)
    elif dataset_name == 'snli':
        train_dataset, valid_dataset, test_dataset = load_dataset_snli(dataset_path, sort_dataset = sort_dataset)
    elif dataset_name == 'trec':
        train_dataset, valid_dataset, test_dataset = load_dataset_trec(dataset_path, sort_dataset = sort_dataset)
    elif dataset_name == 'agnews':
        train_dataset, valid_dataset, test_dataset = load_dataset_agnews(dataset_path, sort_dataset = sort_dataset)
    elif dataset_name == 'mr':
        train_dataset, valid_dataset, test_dataset = load_dataset_mr(dataset_path, sort_dataset = sort_dataset)
    else:
        raise NotImplementedError

    if use_valid_for_train:
        combined_train_xs = train_dataset[0] + valid_dataset[0]
        combined_train_ys = train_dataset[1] + valid_dataset[1]
        train_dataset = (combined_train_xs, combined_train_ys)
        valid_dataset = []

    if sort_dataset:
        sorted_train_dataset = sort_dataset_via_length(train_dataset)
        if not use_valid_for_train:
            sorted_valid_dataset = sort_dataset_via_length(valid_dataset)
        else:
            sorted_valid_dataset = []
        sorted_test_dataset = sort_dataset_via_length(test_dataset)
        return sorted_train_dataset, sorted_valid_dataset, sorted_test_dataset
    return train_dataset, valid_dataset, test_dataset


def load_dataset_sst(path = 'sst-2/', sort_dataset = False):
    def process_file(file):    
        sentence_list = []
        label_list = []
        with open(os.path.join(path,file),'r',encoding = 'utf-8') as f:
            next(f)
            for line in f:
                sen, label = line.split("\t",1)
                sentence_list.append(sen.strip())
                label_list.append(int(label.strip()))
        return (sentence_list, label_list)

    train_dataset = process_file("train.tsv")
    valid_dataset = process_file("dev.tsv")
    test_dataset = process_file("test.tsv")
    return train_dataset, valid_dataset, test_dataset

def load_dataset_sst5(path = 'original/sst-5/', sort_dataset = False):
    train_df = pd.read_csv(os.path.join(path, 'train.csv'), names = ['label','sentence'])
    valid_df = pd.read_csv(os.path.join(path, 'dev.csv'), names = ['label','sentence'])
    test_df = pd.read_csv(os.path.join(path, 'test.csv'), names = ['label','sentence'])
    train_xs = train_df['sentence'].tolist()
    train_ys = [int(x) for x in train_df['label'].tolist()]
    valid_xs = valid_df['sentence'].tolist()
    valid_ys = [int(x) for x in valid_df['label'].tolist()]
    test_xs = test_df['sentence'].tolist()
    test_ys = [int(x) for x in test_df['label'].tolist()]

    train_dataset = (train_xs, train_ys)
    valid_dataset = (valid_xs, valid_ys)
    test_dataset = (test_xs, test_ys)
    return train_dataset, valid_dataset, test_dataset

def load_dataset_rte(path, sort_dataset = False):
    str2label = {'entailment':1, 'not_entailment':0}
    def process_file(file):    
        sentence_list = []
        label_list = []
        with open(os.path.join(path, file),'r',encoding = 'utf-8') as f:
            next(f)
            for line in f:
                idx, sen1, sen2, label = line.strip().split("\t",3)
                sentence_list.append([sen1.strip(), sen2.strip()])
                label_list.append(str2label[label.strip()])
        return (sentence_list, label_list)
    train_dataset = process_file("train.tsv")
    valid_dataset = process_file("dev.tsv")
    test_dataset = process_file("test.tsv")
    return train_dataset, valid_dataset, test_dataset

def load_dataset_mnli(path, sort_dataset = False):
    str2label = {'entailment':2, 'neutral':1, 'contradiction':0}
    def process_file(file):    
        sentence_list = []
        label_list = []
        with open(os.path.join(path, file),'r',encoding = 'utf-8') as f:
            next(f)
            for line in f:
                items = line.strip().split("\t")
                sen1 = items[8]
                sen2 = items[9]
                label = str2label[items[-1]]
                sentence_list.append([sen1.strip(), sen2.strip()])
                label_list.append(int(label))
        return (sentence_list, label_list)
    train_dataset = process_file("train.tsv")
    valid_dataset = process_file("dev_matched.tsv")
    test_dataset = process_file("test_matched.tsv")
    return train_dataset, valid_dataset, test_dataset

def load_dataset_qnli(path = 'qnli/', sort_dataset = False):
    str2label = {'entailment':1,'not_entailment':0}
    def process_file(file):    
        count = 0
        sentence_list = []
        label_list = []
        with open(os.path.join(path, file),'r',encoding = 'utf-8') as f:
            next(f)
            for line in f:
                count += 1
                idx, sen1, sen2, label = line.strip().split("\t",3)
                sentence_list.append([sen1.strip(), sen2.strip()])
                label_list.append(str2label[label])
        return (sentence_list, label_list)
    train_dataset = process_file("train.tsv")
    valid_dataset = process_file("dev.tsv")
    test_dataset = process_file("test.tsv")
    return train_dataset, valid_dataset, test_dataset

def load_dataset_trec(path, sort_dataset = False):
    train_df = pd.read_csv(os.path.join(path, 'train.csv'), names = ['label','sentence'])
    valid_df = pd.read_csv(os.path.join(path, 'dev.csv'), names = ['label','sentence'])
    test_df = pd.read_csv(os.path.join(path, 'test.csv'), names = ['label','sentence'])
    train_xs = train_df['sentence'].tolist()
    train_ys = [int(x) for x in train_df['label'].tolist()]
    valid_xs = valid_df['sentence'].tolist()
    valid_ys = [int(x) for x in valid_df['label'].tolist()]
    test_xs = test_df['sentence'].tolist()
    test_ys = [int(x) for x in test_df['label'].tolist()]

    train_dataset = (train_xs, train_ys)
    valid_dataset = (valid_xs, valid_ys)
    test_dataset = (test_xs, test_ys)
    return train_dataset, valid_dataset, test_dataset

def load_dataset_snli(path, sort_dataset = False):
    str2label = {'entailment':2, 'neutral':1, 'contradiction':0}
    def process_file(file):    
        sentence_list = []
        label_list = []
        with open(os.path.join(path, file),'r',encoding = 'utf-8') as f:
            next(f)
            for line in f:
                item_list = line.strip().split("\t")
                sen1 = item_list[7]
                sen2 = item_list[8]
                label = str2label[item_list[-1]]
                sentence_list.append([sen1.strip(), sen2.strip()])
                label_list.append(label)
        return (sentence_list, label_list)
    train_dataset = process_file("train.tsv")
    valid_dataset = process_file("dev.tsv")
    test_dataset = process_file("test.tsv")
    return train_dataset, valid_dataset, test_dataset

def load_dataset_agnews(path, sort_dataset = False):
    train_df = pd.read_csv(os.path.join(path, 'train.csv'), names = ['index', 'sentence', 'label'])
    valid_df = pd.read_csv(os.path.join(path, 'dev.csv'), names = ['index', 'sentence', 'label'])
    test_df = pd.read_csv(os.path.join(path, 'test.csv'), names = ['index', 'sentence', 'label'])
    train_xs = train_df['sentence'].tolist()
    train_ys = [int(x) for x in train_df['label'].tolist()]
    valid_xs = valid_df['sentence'].tolist()
    valid_ys = [int(x) for x in valid_df['label'].tolist()]
    test_xs = test_df['sentence'].tolist()
    test_ys = [int(x) for x in test_df['label'].tolist()]

    train_dataset = (train_xs, train_ys)
    valid_dataset = (valid_xs, valid_ys)
    test_dataset = (test_xs, test_ys)
    return train_dataset, valid_dataset, test_dataset

def load_dataset_mr(path, sort_dataset = False):
    train_df = pd.read_csv(os.path.join(path, 'train.csv'), names = ['label','sentence'])
    valid_df = pd.read_csv(os.path.join(path, 'dev.csv'), names = ['label','sentence'])
    test_df = pd.read_csv(os.path.join(path, 'test.csv'), names = ['label','sentence'])
    train_xs = train_df['sentence'].tolist()
    train_ys = [int(x) for x in train_df['label'].tolist()]
    valid_xs = valid_df['sentence'].tolist()
    valid_ys = [int(x) for x in valid_df['label'].tolist()]
    test_xs = test_df['sentence'].tolist()
    test_ys = [int(x) for x in test_df['label'].tolist()]

    train_dataset = (train_xs, train_ys)
    valid_dataset = (valid_xs, valid_ys)
    test_dataset = (test_xs, test_ys)
    return train_dataset, valid_dataset, test_dataset

def load_dataset_cr(path, sort_dataset = False):
    train_df = pd.read_csv(os.path.join(path, 'train.csv'), names = ['label','sentence'])
    valid_df = pd.read_csv(os.path.join(path, 'dev.csv'), names = ['label','sentence'])
    test_df = pd.read_csv(os.path.join(path, 'test.csv'), names = ['label','sentence'])
    train_xs = train_df['sentence'].tolist()
    train_ys = [int(x) for x in train_df['label'].tolist()]
    valid_xs = valid_df['sentence'].tolist()
    valid_ys = [int(x) for x in valid_df['label'].tolist()]
    test_xs = test_df['sentence'].tolist()
    test_ys = [int(x) for x in test_df['label'].tolist()]

    train_dataset = (train_xs, train_ys)
    valid_dataset = (valid_xs, valid_ys)
    test_dataset = (test_xs, test_ys)
    return train_dataset, valid_dataset, test_dataset


def sort_dataset_via_length(dataset):
    text_list = dataset[0]
    label_list = dataset[1]
    if type(text_list[0]) == list and len(text_list[0]) == 2:
        sentence_pair = True
    else:
        sentence_pair = False
    
    if sentence_pair:
        length_list = [len(x[0].split()) + len(x[1].split()) for x in text_list]
    else:
        length_list = [len(x.split()) for x in text_list]
    length_order = np.argsort(length_list, kind = 'stable')
    sorted_text_list = [text_list[x] for x in length_order]
    sorted_label_list = [label_list[x] for x in length_order]
    return (sorted_text_list, sorted_label_list)

def get_task_type(dataset_name):
    if dataset_name in ['sst', 'imdb', 'sst5', 'trec', 'agnews','mr']:
        return False
    elif dataset_name in ['rte','qnli','mnli', 'snli']:
        return True

def get_class_num(dataset_name):
    if dataset_name in ['sst', 'imdb', 'rte', 'qnli','mr']:
        return 2
    elif dataset_name in ['mnli', 'snli']:
        return 3
    elif dataset_name in ['agnews']:
        return 4
    elif dataset_name in ['sst5']:
        return 5
    elif dataset_name in ['trec']:
        return 6


def get_batch_size(model_name):
    if 'deberta' in model_name:
        return 4
    else:
        return 12

def get_weak_cls_num(dataset_name):
    '''
    only for few-shot setting where validation set is also used for training
    '''
    cls_num_dict = {
        'agnews': 20,
        'qnli': 50,
        'rte': 50,
        'trec': 100,
        'snli': 80,
        'mnli': 90,
        'sst': -1,
        'mr': -1,
    }
    return cls_num_dict[dataset_name]


def get_template_list(dataset, model = 'roberta'):
    if model in ['roberta']:
        if dataset == 'sst':
            template_dir_list = [os.path.join(ROOT_DIR, 'templates/t5_sorted_sst/')]
        elif dataset == 'sst-5':
            template_dir_list = [os.path.join(ROOT_DIR, 'templates/t5_sorted_sst-5/')]
        elif dataset == 'rte':
            template_dir_list = [os.path.join(ROOT_DIR, 'templates/t5_sorted_rte/')]
        elif dataset == 'mnli':
            template_dir_list = [os.path.join(ROOT_DIR, 'templates/t5_sorted_mnli/')]
        elif dataset == 'qnli':
            template_dir_list = [os.path.join(ROOT_DIR, 'templates/t5_sorted_qnli/')]
        elif dataset == 'snli':
            template_dir_list = [os.path.join(ROOT_DIR, 'templates/t5_sorted_snli/')]
        elif dataset == 'trec':
            template_dir_list = [os.path.join(ROOT_DIR, 'templates/t5_sorted_trec/')]
        elif dataset == 'agnews':
            template_dir_list = [os.path.join(ROOT_DIR, 'templates/t5_sorted_agnews/')]
        elif dataset == 'mr':
            template_dir_list = [os.path.join(ROOT_DIR, 'templates/t5_sorted_mr/')]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return template_dir_list


def get_full_template_list(dataset, ):
    if dataset == 'sst':
        template_dir_list = [os.path.join(ROOT_DIR, 'templates/full_templates/t5_sorted_sst/')]
    elif dataset == 'rte':
        template_dir_list = [os.path.join(ROOT_DIR, 'templates/full_templates/t5_sorted_rte/')]
    elif dataset == 'mnli':
        template_dir_list = [os.path.join(ROOT_DIR, 'templates/full_templates/t5_sorted_mnli/')]
    elif dataset == 'qnli':
        template_dir_list = [os.path.join(ROOT_DIR, 'templates/full_templates/t5_sorted_qnli/')]
    elif dataset == 'snli':
        template_dir_list = [os.path.join(ROOT_DIR, 'templates/full_templates/t5_sorted_snli/')]
    elif dataset == 'trec':
        template_dir_list = [os.path.join(ROOT_DIR, 'templates/full_templates/t5_sorted_trec/')]
    elif dataset == 'agnews':
        template_dir_list = [os.path.join(ROOT_DIR, 'templates/full_templates/t5_sorted_agnews/')]
    elif dataset == 'mr':
        template_dir_list = [os.path.join(ROOT_DIR, 'templates/full_templates/t5_sorted_mr/')]
    else:
        raise NotImplementedError
    return template_dir_list

def get_template_list_with_filter(dataset, fewshot = False, low = False, fewshot_seed = 13, fewshot_k = 16, topk = 10,
                                  return_source_dir = False):
    assert (fewshot or low)
    if fewshot:
        stat_file_path = os.path.join(ROOT_DIR, f'stat_data_file/{dataset}/roberta-{dataset}-{fewshot_k}shot-seed{fewshot_seed}.csv')
    else:
        stat_file_path = os.path.join(ROOT_DIR, f'stat_data_file/{dataset}/roberta-{dataset}-low{fewshot_k}-seed{fewshot_seed}.csv')

    stat_df = pd.read_csv(stat_file_path)
    valid_acc = stat_df['valid_acc'].to_numpy()
    template_addr_list = stat_df['name'].tolist()
    topk_idxs = np.argsort(-valid_acc)[:topk]
    topk_templates = [template_addr_list[x] for x in topk_idxs]
    topk_valid_accs = [valid_acc[x] for x in topk_idxs]
    print(f"filtered templates: ")
    for i in range(len(topk_templates)):
        print(topk_templates[i], topk_valid_accs[i])
    
    if fewshot:
        filtered_template_save_dir = os.path.join(ROOT_DIR, f'templates/filtered-templates/{dataset}/{fewshot_k}shot-seed{fewshot_seed}')
    else:
        filtered_template_save_dir = os.path.join(ROOT_DIR, f'templates/filtered-templates/{dataset}/low{fewshot_k}-seed{fewshot_seed}')

    if os.path.exists(filtered_template_save_dir):
        if return_source_dir:
            return [filtered_template_save_dir], topk_templates        
        else:
            return [filtered_template_save_dir]
    else:
        os.makedirs(filtered_template_save_dir)
    for template_addr in topk_templates:
        shutil.copy(template_addr, filtered_template_save_dir)

    if return_source_dir:
        return [filtered_template_save_dir], topk_templates
    return [filtered_template_save_dir]


def load_dataset_rte_tmp(path, sort_dataset = False):
    str2label = {'entailment':1,'not_entailment':0}
    def process_file(file):    
        sentence_list = []
        label_list = []
        with open(path + file,'r',encoding = 'utf-8') as f:
            next(f)
            for line in f:
                idx, sen1, sen2, label = line.strip().split("\t",3)
                sentence_list.append([sen1.strip(), sen2.strip()])
                label_list.append(str2label[label.strip()])
        return (sentence_list, label_list)
    train_dataset = process_file("train.tsv")
    valid_dataset = process_file("dev.tsv")
    test_dataset = process_file("dev.tsv")

    if sort_dataset:
        sorted_train_dataset = sort_dataset_via_length(train_dataset)
        sorted_valid_dataset = sort_dataset_via_length(valid_dataset)
        sorted_test_dataset = sort_dataset_via_length(test_dataset)
        return sorted_train_dataset, sorted_valid_dataset, sorted_test_dataset

    return train_dataset, valid_dataset, test_dataset

