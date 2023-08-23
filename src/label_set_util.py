import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ptuning import BaseModel, RoBERTaVTuningClassification
from src.template import SentenceTemplate


def generate_multicls_l1_label_set_with_cache(train_dataset, vtuning_model: RoBERTaVTuningClassification,
                                              weight_list = [], cache_probs = None, label_set_size = 0, num_classes = 3,
                                              norm_class = False):

    vocab_size = cache_probs.size(1)
    label_indicator = torch.zeros(num_classes, vocab_size).float().to(vtuning_model.device)
    sentence_list, label_list = train_dataset
    if weight_list == []:
        weight_list = torch.ones(len(sentence_list)).float().to(vtuning_model.device)
    else:
        weight_list = torch.FloatTensor(weight_list).to(vtuning_model.device) * len(sentence_list)
        assert len(weight_list) == len(sentence_list)

    batch_labels = torch.LongTensor(label_list).to(vtuning_model.device)
    batch_weights = weight_list.view(-1, 1)

    for i in range(num_classes):
        label_mask = batch_labels == i
        label_multiplier = torch.zeros_like(batch_labels).float()
        if not norm_class:
            label_multiplier.fill_(-1.0)
        else:
            label_multiplier.fill_(-1/(num_classes - 1))
        label_multiplier[label_mask] = 1.0
        label_multiplier = label_multiplier.view(-1,1)
        balanced_score_tensor = cache_probs * label_multiplier
        balanced_score_tensor = balanced_score_tensor * batch_weights
        label_indicator[i,:] = torch.sum(balanced_score_tensor, dim = 0)          

    root = torch.argmax(label_indicator, dim = 0)
    return root, label_indicator

