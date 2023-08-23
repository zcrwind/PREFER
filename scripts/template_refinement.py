import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np 
import torch

import tqdm
import time
import csv

from src.multicls_trainer import PromptBoostingTrainer
from src.ptuning import RoBERTaVTuningClassification
from src.saver import PredictionSaver
from src.template import SentenceTemplate, TemplateManager
from src.utils import ROOT_DIR, BATCH_SIZE, create_logger, MODEL_CACHE_DIR
from src.data_util import get_class_num, load_dataset, get_task_type, get_full_template_list

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, default = 'sst')
parser.add_argument("--model", type = str, default = 'roberta')
parser.add_argument("--label_set_size", type = int, default = 5)
parser.add_argument("--eval_num", type = int, default = 100)
parser.add_argument("--sort_dataset", action = 'store_true')

parser.add_argument("--fewshot", action = 'store_true')
parser.add_argument("--fewshot_k", type = int, default = 0)
parser.add_argument("--low", action = 'store_true')
parser.add_argument("--low_mode", type = str, choices = ['low-resource-16valid'])
parser.add_argument("--fewshot_seed", type = int, default = 100, choices = [100, 13, 21, 42, 87])

args = parser.parse_args()



if __name__ == '__main__':
    device = torch.device('cuda')
    dataset = args.dataset
    sentence_pair = get_task_type(dataset)
    num_classes = get_class_num(dataset)
    model = args.model

    sort_dataset = args.sort_dataset
    label_set_size = args.label_set_size
    eval_num = args.eval_num

    adaboost_maximum_epoch = 20000

    fewshot = args.fewshot
    low = args.low
    low_mode = args.low_mode
    fewshot_k = args.fewshot_k
    fewshot_seed = args.fewshot_seed
    assert sort_dataset

    wandb_name = f"{model}-{dataset}"
    if fewshot:
        wandb_name += f"-{fewshot_k}shot-seed{fewshot_seed}"
    elif low:
        wandb_name += f"-low{fewshot_k}-seed{fewshot_seed}"
    else:
        raise NotImplementedError

    save_dir = f"stat_data_file/{dataset}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f = open(f"stat_data_file/{dataset}/{wandb_name}.csv", 'w', encoding = 'utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['name', 'template', 'verbalizer', 'train_acc', 'valid_acc'])

    train_dataset, valid_dataset, test_dataset = load_dataset(
        dataset_name=dataset,
        sort_dataset=sort_dataset,
        fewshot=fewshot,
        k=fewshot_k,
        rand_seed=fewshot_seed,
        low_resource=low
    )

    num_training = len(train_dataset[0])
    num_valid = len(valid_dataset[0])
    train_labels = torch.LongTensor(train_dataset[1]).to(device)
    valid_labels = torch.LongTensor(valid_dataset[1]).to(device)

    weight_tensor = torch.ones(num_training, dtype = torch.float32).to(device) / num_training

    vtuning_model = RoBERTaVTuningClassification(model_type = 'roberta-large', cache_dir = MODEL_CACHE_DIR + 'roberta_model/roberta-large/',
                                            device = device, verbalizer_dict = None, sentence_pair = sentence_pair)
    template_dir_list = get_full_template_list(dataset)

    template_manager = TemplateManager(template_dir_list = template_dir_list, output_token = vtuning_model.tokenizer.mask_token, max_template_num = eval_num,
                                        rand_order = False)

    dir_list = "\n\t".join(template_manager.template_dir_list)
    print(f"using templates from: {dir_list}",)

    trainer = PromptBoostingTrainer(adaboost_lr = 1.0, num_classes = num_classes, adaboost_maximum_epoch = adaboost_maximum_epoch)
    
    word2idx = vtuning_model.tokenizer.get_vocab()
    for template_id in tqdm.tqdm(range(eval_num)):
        template = template_manager.change_template()
        str_template = template.visualize()
        template_path = template.template_path

        train_probs = trainer.pre_compute_logits(vtuning_model, template, train_dataset,)
        valid_probs = trainer.pre_compute_logits(vtuning_model, template, valid_dataset,)

        verbalizer, train_error,train_acc, wrong_flags,train_preds = trainer.train(
            train_dataset, vtuning_model, train_probs, train_labels,
            weight_tensor=weight_tensor, label_set_size=label_set_size
        )
        valid_acc, valid_preds, valid_logits = trainer.evaluate(word2idx, valid_probs, verbalizer, valid_labels)

        verbalizer = f"{verbalizer}"
        csv_writer.writerow([template_path, str_template, verbalizer, f"{train_acc}", f"{valid_acc}"])

        del train_preds
        del valid_preds



