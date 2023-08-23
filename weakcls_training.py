import numpy as np 
import torch

import tqdm
import time
import copy
import os

from src.multicls_trainer import PromptBoostingTrainer
from src.ptuning import  RoBERTaVTuningClassification, OPTVTuningClassification
from src.saver import PredictionSaver, TestPredictionSaver
from src.template import SentenceTemplate, TemplateManager
from src.utils import ROOT_DIR, create_logger, MODEL_CACHE_DIR
from src.data_util import get_class_num, load_dataset, get_task_type, get_template_list

import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, default = 'sst')
parser.add_argument("--model", type = str, default = 'roberta')
parser.add_argument("--label_set_size", type = int, default = 5)
parser.add_argument("--max_template_num", type = int, default = 10)
parser.add_argument("--template_dir", type = str, default = '')

parser.add_argument("--pred_cache_dir", type = str, default = '')
parser.add_argument("--use_wandb", action = 'store_true')

parser.add_argument("--sort_dataset", action = 'store_true')

parser.add_argument("--fewshot", action = 'store_true')
parser.add_argument("--fewshot_k", type = int, default = 0)
parser.add_argument("--fewshot_seed", type = int, default = 100, choices = [100, 13, 21, 42, 87])

parser.add_argument("--filter_templates", action = 'store_true')

args = parser.parse_args()



if __name__ == '__main__':
    device = torch.device('cuda')
    dataset = args.dataset
    sentence_pair = get_task_type(dataset)
    num_classes = get_class_num(dataset)
    model = args.model

    pred_cache_dir = args.pred_cache_dir
    sort_dataset = args.sort_dataset
    use_wandb = args.use_wandb
    label_set_size = args.label_set_size
    max_template_num = args.max_template_num

    adaboost_maximum_epoch = 20000

    fewshot = args.fewshot
    fewshot_k = args.fewshot_k
    fewshot_seed = args.fewshot_seed

    filter_templates = args.filter_templates

    suffix = "weakcls"
    wandb_name = f"{model}-{dataset}-template{max_template_num}-{suffix}"

    if use_wandb:
        if fewshot:
            wandb_name += f"-{fewshot_k}shot-seed{fewshot_seed}"
        wandb.init(project = f'vtuning-{dataset}', name = f'{wandb_name}')


    train_dataset, valid_dataset, test_dataset = load_dataset(dataset_name = dataset, sort_dataset = sort_dataset, fewshot = fewshot, k = fewshot_k, rand_seed = fewshot_seed,
                                                            use_valid_for_train = False)

    num_training = len(train_dataset[0])
    num_valid = len(valid_dataset[0])
    num_test = len(test_dataset[0])
    train_labels = torch.LongTensor(train_dataset[1]).to(device)
    valid_labels = torch.LongTensor(valid_dataset[1]).to(device)
    test_labels = torch.LongTensor(test_dataset[1]).to(device)

    weight_tensor = torch.ones(num_training, dtype = torch.float32).to(device) / num_training

    if model == 'roberta':
        vtuning_model = RoBERTaVTuningClassification(model_type = 'roberta-large', cache_dir = os.path.join(MODEL_CACHE_DIR, 'roberta_model/roberta-large/'),
                                                device = device, verbalizer_dict = None, sentence_pair = sentence_pair)
    elif model == 'opt-13b':
        vtuning_model = OPTVTuningClassification(model_type = 'facebook/opt-13b', cache_dir = os.path.join(MODEL_CACHE_DIR, 'opt_model/opt-1.3b/'),
                                                device = device, verbalizer_dict = None, sentence_pair = sentence_pair)
    else:
        raise NotImplementedError
    if args.template_dir == '':
        template_dir_list = get_template_list(dataset, model = model)
    else:
        template_dir_list = [os.path.join(ROOT_DIR, args.template_dir)]
    template_manager = TemplateManager(template_dir_list = template_dir_list, output_token = vtuning_model.tokenizer.mask_token,
                                      rand_order = False)

    dir_list = "\n\t".join(template_manager.template_dir_list)
    print(f"using templates from: {dir_list}",)

    trainer = PromptBoostingTrainer(num_classes = num_classes, adaboost_maximum_epoch = adaboost_maximum_epoch)

    word2idx = vtuning_model.tokenizer.get_vocab()

    best_valid = 0
    best_test = 0
    best_template = None
    best_verbalizer = None

    train_probs, valid_probs = None, None
    all_templates = template_manager.get_all_template()
    iter_num = np.min([len(all_templates), args.max_template_num])

    for model_id in tqdm.tqdm(range(iter_num)):
        del train_probs
        del valid_probs
        template = template_manager.change_template()
        template.visualize()
    
        train_probs = trainer.pre_compute_logits(vtuning_model, template, train_dataset,)
        valid_probs = trainer.pre_compute_logits(vtuning_model, template, valid_dataset,)

        trainer.record_dataset_weights(weight_tensor)

        verbalizer, train_error,train_acc, wrong_flags,train_preds= trainer.train(train_dataset, vtuning_model, train_probs, train_labels,
                                                                                weight_tensor = weight_tensor,label_set_size = label_set_size,
                                                                                )
        print(verbalizer)
        print(f"\ttemplate {model_id + 1} finished")
        print(f"\ttrain error {train_error}, train_acc {train_acc}")
        succ_flag = True
                
        valid_acc, valid_preds, valid_logits = trainer.evaluate(word2idx, valid_probs, verbalizer, valid_labels)
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_template = copy.deepcopy(template)
            best_verbalizer = copy.deepcopy(verbalizer)
        tolog = {
            'train_error': train_error,
            'train_acc': train_acc,
            'valid_acc': valid_acc,
        }
        if use_wandb:
            wandb.log(tolog)
    
    cls_scores = trainer.pre_compute_logits(vtuning_model, best_template, test_dataset)
    test_acc, test_preds, test_logits, = trainer.evaluate(word2idx, cls_scores, best_verbalizer, test_labels)
    best_template.visualize()
    print(f"Best test acc {test_acc}")
