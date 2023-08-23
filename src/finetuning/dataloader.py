import datasets
from datasets import Dataset, DatasetDict
import numpy as np
from transformers import DataCollatorWithPadding, PreTrainedTokenizer
import inspect
import torch
import pandas as pd
from ..data_util import sample_k_shot
from ..utils import ROOT_DIR

dataset_transform = {'sst':'SST-2','rte':'RTE','snli':'SNLI','mnli':'MNLI','qnli': 'QNLI', 'trec':'trec','imdb':'imdb','sst-5':'SST-5'}


def remove_unused_columns(model, dataset: Dataset, reserved_columns = []):
    signature = inspect.signature(model.forward)
    _signature_columns = list(signature.parameters.keys())
    _signature_columns += ["label", "label_ids"]
    _signature_columns += reserved_columns
    columns = [k for k in _signature_columns if k in dataset.column_names]
    ignored_columns = list(set(dataset.column_names) - set(_signature_columns))
    return dataset.remove_columns(ignored_columns)

def evaluate_accuracy(eval_dataloader, model):
    total_loss = 0.0
    total_acc = 0.0
    num_batches = len(eval_dataloader)
    for batch in eval_dataloader:
        batch = batch.to(model.device)
        input_ids = batch['input_ids']
        model_output = model(**batch)
        loss = model_output.loss
        logits = model_output.logits
        ys = batch['labels']
        total_acc += torch.argmax(logits,dim = 1).eq(ys).sum().item() / len(ys)
        total_loss += loss.item() / len(ys)
    total_acc /= num_batches
    total_loss /= num_batches
    return total_loss, total_acc

class LocalSSTDataset():
    def __init__(self, tokenizer = None, sort_dataset = False,
                 fewshot = False, k = 0, rand_seed = 0,
                 low_resource = False, low_resource_mode = 'low-resource-16valid') -> None:
        assert not (fewshot and low_resource), "fewshot and low_resouce cannot be selected simutaneously!"
        self.tokenizer = tokenizer
        if fewshot:
            assert k > 0, f"k must > 0, found {k}"
            dataset_path = f'datasets/k-shot/SST-2/{k}-{rand_seed}/'
        elif low_resource:
            dataset_path = ROOT_DIR + f'datasets/low-resource-16valid/SST-2/{k}-{rand_seed}/'            
        else:
            dataset_path =  f'datasets/full_dataset/SST-2/'
        data_files = {"train": dataset_path + "train.tsv", "valid": dataset_path + "dev.tsv", "test": dataset_path + "test.tsv"}
        dataset_dict = datasets.load_dataset("text", data_files = data_files)
        train_set, valid_set, test_set = dataset_dict['train'],dataset_dict['valid'],dataset_dict['test']
        train_set = train_set.select(np.arange(1, len(train_set)))
        valid_set = valid_set.select(np.arange(1, len(valid_set)))
        test_set = test_set.select(np.arange(1, len(test_set)))

        train_set = train_set.map(self.preprocess_fn, batched=True,)
        valid_set = valid_set.map(self.preprocess_fn, batched=True,)    
        test_set = test_set.map(self.preprocess_fn, batched=True,)    

        self.train_dataset = train_set.map(self.tokenize_corpus, batched=True,)
        self.valid_dataset = valid_set.map(self.tokenize_corpus, batched=True,)
        self.test_dataset = test_set.map(self.tokenize_corpus, batched=True,)
        self.data_collator = DataCollatorWithPadding(tokenizer, padding = 'longest')

    def preprocess_fn(self, examples):
        sentence_list = []
        label_list = []
        for example in examples['text']:
            sentence, label = example.strip().split('\t', 1)
            sentence_list.append(sentence)
            label_list.append(int(label))
        return {'sentence': sentence_list, 'label': label_list}

    def tokenize_corpus(self, examples):
        tokenized = self.tokenizer(examples['sentence'], truncation = True, max_length = 512)
        return tokenized

class LocalNLIDataset():
    def __init__(self, dataset_name = 'rte', tokenizer = None,
                 fewshot = False, k = 0, rand_seed = 0,
                 low_resource = False, low_resource_mode = 'low-resource-16valid') -> None:
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        if fewshot:
            assert k > 0, f"k must > 0, found {k}"
            dataset_path = ROOT_DIR + f'datasets/k-shot/{dataset_transform[dataset_name]}/{k}-{rand_seed}/'
        elif low_resource:
            dataset_path = ROOT_DIR + f'datasets/low-resource-16valid/{dataset_transform[dataset_name]}/{k}-{rand_seed}/'            
        else:
            dataset_path = ROOT_DIR + f'datasets/full_dataset/{dataset_transform[dataset_name]}/'
        if dataset_name == 'mnli':
            data_files = {"train": dataset_path + "train.tsv", "valid": dataset_path + "dev_matched.tsv", "test": dataset_path + "test_matched.tsv"}
        else:
            data_files = {"train": dataset_path + "train.tsv", "valid": dataset_path + "dev.tsv", "test": dataset_path + "test.tsv"}
        dataset_dict = datasets.load_dataset("text", data_files = data_files)
        train_set, valid_set, test_set = dataset_dict['train'],dataset_dict['valid'],dataset_dict['test']
        train_set = train_set.select(np.arange(1, len(train_set)))
        valid_set = valid_set.select(np.arange(1, len(valid_set)))
        test_set = test_set.select(np.arange(1, len(test_set)))
        if dataset_name == 'snli':
            train_set = train_set.map(self.preprocess_fn_snli, batched=True,)    
            valid_set = valid_set.map(self.preprocess_fn_snli, batched=True,)    
            test_set = test_set.map(self.preprocess_fn_snli, batched=True,)    
        elif dataset_name == 'mnli':
            train_set = train_set.map(self.preprocess_fn_mnli, batched=True,)    
            valid_set = valid_set.map(self.preprocess_fn_mnli, batched=True,)    
            test_set = test_set.map(self.preprocess_fn_mnli, batched=True,)    
        else:
            train_set = train_set.map(self.preprocess_fn, batched=True,)
            valid_set = valid_set.map(self.preprocess_fn, batched=True,)    
            test_set = test_set.map(self.preprocess_fn, batched=True,)    

        self.train_dataset = train_set.map(self.tokenize_corpus, batched=True,)
        self.valid_dataset = valid_set.map(self.tokenize_corpus, batched=True,)
        self.test_dataset = test_set.map(self.tokenize_corpus, batched=True,)

        self.data_collator = DataCollatorWithPadding(tokenizer, padding = 'longest')

    def preprocess_fn(self, examples):
        str2label = {'entailment':1,'not_entailment':0}
        premise_list = []
        hypothesis_list = []
        label_list = []
        for example in examples['text']:
            idx, prem, hyp, label = example.strip().split('\t', 3)
            premise_list.append(prem.strip())
            hypothesis_list.append(hyp.strip())
            label_list.append(str2label[label])
        return {'premise': premise_list, 'hypothesis': hypothesis_list, 'label': label_list}

    def preprocess_fn_snli(self, examples):
        str2label = {'entailment':2, 'neutral':1, 'contradiction':0}
        premise_list = []
        hypothesis_list = []
        label_list = []
        for example in examples['text']:
            item_list = example.strip().split("\t")
            prem = item_list[7]
            hyp = item_list[8]
            label = str2label[item_list[-1]]
            premise_list.append(prem.strip())
            hypothesis_list.append(hyp.strip())
            label_list.append(label)
        return {'premise': premise_list, 'hypothesis': hypothesis_list, 'label': label_list}

    def preprocess_fn_mnli(self, examples):
        str2label = {'entailment':2, 'neutral':1, 'contradiction':0}
        premise_list = []
        hypothesis_list = []
        label_list = []
        for example in examples['text']:
            item_list = example.strip().split("\t")
            prem = item_list[8]
            hyp = item_list[9]
            label = str2label[item_list[-1]]
            premise_list.append(prem.strip())
            hypothesis_list.append(hyp.strip())
            label_list.append(label)
        return {'premise': premise_list, 'hypothesis': hypothesis_list, 'label': label_list}

    def tokenize_corpus(self, examples):
        if self.dataset_name == 'mnli':
            max_length = 128
        else:
            max_length = 256
        tokenized = self.tokenizer(examples['premise'], examples['hypothesis'], truncation = True, max_length = max_length)
        return tokenized

class LocalTrecDataset():
    def __init__(self, tokenizer = None,
                 fewshot = False, k = 0, rand_seed = 0,
                 low_resource = False, low_resource_mode = 'low-resource-16valid') -> None:
        self.tokenizer = tokenizer
        if fewshot:
            assert k > 0, f"k must > 0, found {k}"
            dataset_path = ROOT_DIR + f'datasets/k-shot/trec/{k}-{rand_seed}/'
        elif low_resource:
            dataset_path = ROOT_DIR + f'datasets/low-resource-16valid/trec/{k}-{rand_seed}/'            
        else:
            dataset_path = ROOT_DIR + f'datasets/full_dataset/trec/'

        data_files = {"train": dataset_path + "train.csv", "valid": dataset_path + "dev.csv", "test": dataset_path + "test.csv"}
        train_set = pd.read_csv(data_files['train'], names = ['label', 'sentence'])
        train_set = datasets.Dataset.from_pandas(train_set) 
        valid_set = pd.read_csv(data_files['valid'], names = ['label', 'sentence'])
        valid_set = datasets.Dataset.from_pandas(valid_set)
        test_set = pd.read_csv(data_files['test'],  names = ['label', 'sentence'])
        test_set = datasets.Dataset.from_pandas(test_set)
        self.train_dataset = train_set.map(self.tokenize_corpus, batched=True,)
        self.valid_dataset = valid_set.map(self.tokenize_corpus, batched=True,)
        self.test_dataset = test_set.map(self.tokenize_corpus, batched=True,)
        self.data_collator = DataCollatorWithPadding(tokenizer, padding = 'longest')

    def tokenize_corpus(self, examples):
        tokenized = self.tokenizer(examples['sentence'], truncation = True, max_length = 512)
        return tokenized

class LocalAGDataset():
    def __init__(self, tokenizer = None,
                 fewshot = False, k = 0, rand_seed = 0,
                 low_resource = False, low_resource_mode = 'low-resource-16valid') -> None:
        self.tokenizer = tokenizer
        if fewshot:
            assert k > 0, f"k must > 0, found {k}"
            dataset_path = ROOT_DIR + f'datasets/k-shot/ag_news/{k}-{rand_seed}/'
        elif low_resource:
            dataset_path = ROOT_DIR + f'datasets/low-resource-16valid/ag_news/{k}-{rand_seed}/'            
        else:
            dataset_path = ROOT_DIR + f'datasets/full_dataset/ag_news/'

        data_files = {"train": dataset_path + "train.csv", "valid": dataset_path + "dev.csv", "test": dataset_path + "test.csv"}
        train_set = pd.read_csv(data_files['train'], names = ['index', 'sentence', 'label'])
        train_set = datasets.Dataset.from_pandas(train_set) 
        valid_set = pd.read_csv(data_files['valid'], names = ['index', 'sentence', 'label'])
        valid_set = datasets.Dataset.from_pandas(valid_set)
        test_set = pd.read_csv(data_files['test'],  names = ['index', 'sentence', 'label'])
        test_set = datasets.Dataset.from_pandas(test_set)
        self.train_dataset = train_set.map(self.tokenize_corpus, batched=True,)
        self.valid_dataset = valid_set.map(self.tokenize_corpus, batched=True,)
        self.test_dataset = test_set.map(self.tokenize_corpus, batched=True,)
        self.data_collator = DataCollatorWithPadding(tokenizer, padding = 'longest')

    def tokenize_corpus(self, examples):
        tokenized = self.tokenizer(examples['sentence'], truncation = True, max_length = 128)
        return tokenized

class LocalMRDataset():
    def __init__(self, tokenizer = None,
                 fewshot = False, k = 0, rand_seed = 0,
                 low_resource = False, low_resource_mode = 'low-resource-16valid') -> None:
        self.tokenizer = tokenizer
        if fewshot:
            assert k > 0, f"k must > 0, found {k}"
            dataset_path = ROOT_DIR + f'datasets/k-shot/mr/{k}-{rand_seed}/'
        elif low_resource:
            dataset_path = ROOT_DIR + f'datasets/low-resource-16valid/mr/{k}-{rand_seed}/'            
        else:
            dataset_path = ROOT_DIR + f'datasets/full_dataset/mr/'

        data_files = {"train": dataset_path + "train.csv", "valid": dataset_path + "dev.csv", "test": dataset_path + "test.csv"}
        train_set = pd.read_csv(data_files['train'], names = ['label', 'sentence'])
        train_set = datasets.Dataset.from_pandas(train_set) 
        valid_set = pd.read_csv(data_files['valid'], names = ['label', 'sentence'])
        valid_set = datasets.Dataset.from_pandas(valid_set)
        test_set = pd.read_csv(data_files['test'],  names = ['label', 'sentence'])
        test_set = datasets.Dataset.from_pandas(test_set)
        self.train_dataset = train_set.map(self.tokenize_corpus, batched=True,)
        self.valid_dataset = valid_set.map(self.tokenize_corpus, batched=True,)
        self.test_dataset = test_set.map(self.tokenize_corpus, batched=True,)
        self.data_collator = DataCollatorWithPadding(tokenizer, padding = 'longest')

    def tokenize_corpus(self, examples):
        tokenized = self.tokenizer(examples['sentence'], truncation = True, max_length = 512)
        return tokenized



class LocalIMDbDataset():
    def __init__(self, tokenizer = None,
                 fewshot = False, k = 0, rand_seed = 0) -> None:
        self.tokenizer = tokenizer
        if fewshot:
            assert k > 0, f"k must > 0, found {k}"
            dataset_path = ROOT_DIR + f'datasets/k-shot/imdb/{k}-{rand_seed}/'
        else:
            dataset_path = ROOT_DIR + f'datasets/full_dataset/imdb/'

        data_files = {"train": dataset_path + "train.csv", "valid": dataset_path + "dev.csv", "test": dataset_path + "test.csv"}
        train_set = pd.read_csv(data_files['train'], names = ['index', 'sentence', 'label'])
        train_set = datasets.Dataset.from_pandas(train_set) 
        valid_set = pd.read_csv(data_files['valid'], names = ['index', 'sentence', 'label'])
        valid_set = datasets.Dataset.from_pandas(valid_set)
        test_set = pd.read_csv(data_files['test'],  names = ['index', 'sentence', 'label'])
        test_set = datasets.Dataset.from_pandas(test_set)
        self.train_dataset = train_set.map(self.tokenize_corpus, batched=True,)
        self.valid_dataset = valid_set.map(self.tokenize_corpus, batched=True,)
        self.test_dataset = test_set.map(self.tokenize_corpus, batched=True,)
        self.data_collator = DataCollatorWithPadding(tokenizer, padding = 'longest')

    def tokenize_corpus(self, examples):
        tokenized = self.tokenizer(examples['sentence'], truncation = True, max_length = 128)
        return tokenized
