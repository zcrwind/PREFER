import numpy as np
from transformers import (RobertaForMaskedLM, RobertaTokenizer, GPT2Tokenizer, OPTForCausalLM, AutoTokenizer, AutoModelForMaskedLM)

from typing import List, Optional, Union
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .template import SentenceTemplate
from .utils import ROOT_DIR

class VTuningOutput():
    def __init__(self, positive_probs = None, negative_probs = None, positive_prob = None, negative_prob = None, 
                    pred_labels = None, all_token_probs = None, all_token_logits = None):
        self.positive_probs = positive_probs
        self.negative_probs = negative_probs
        self.positive_prob = positive_prob
        self.negative_prob = negative_prob
        self.pred_labels = pred_labels

        self.all_token_probs = all_token_probs
        self.all_token_logits = all_token_logits

class BaseModel():
    def __init__(self, num_labels = 2, max_length = 512):
        self.num_lables = num_labels
        self.max_length = max_length
    
    def preprocess_input(self, input_list: List[str]):
        '''
        build the input with templates and then conduct tokenization
        input_list: Must be raw text examples
        '''
        raise NotImplementedError
    
    def predict(self, input_list: Union[List[str], str], **kwargs):
        '''
        predict a list of input examples or an single example
        '''
        if type(input_list) == str:
            input_list = [input_list]
            pass
        elif type(input_list) == List[str]:
            pass
        else:
            raise NotImplementedError
        
        self.preprocess_input(input_list)
        pass

class RoBERTaVTuningClassification(BaseModel):
    def __init__(self, model_type, cache_dir = None, finetune_dir = None, num_labels = 2, max_length = 512, sentence_pair = False,
                device = torch.device('cuda'), verbalizer_dict = None,):
        super().__init__(num_labels, max_length)
        self.model_type = model_type
        self.cache_dir = cache_dir
        self.finetune_dir = finetune_dir
        self.num_lables = num_labels
        self.max_length = max_length
        self.sentence_pair = sentence_pair
        if self.finetune_dir == None:
            lm_model = RobertaForMaskedLM.from_pretrained(cache_dir)
            self.tokenizer = RobertaTokenizer.from_pretrained(cache_dir)
        else:
            lm_model = RobertaForMaskedLM.from_pretrained(self.finetune_dir)
            self.tokenizer = RobertaTokenizer.from_pretrained(self.finetune_dir)

        self.device = device
        self.verbalizer_dict = verbalizer_dict

        self.lm_model = lm_model.to(device)
        self.lm_model.eval()
        self.word2idx = self.tokenizer.get_vocab()

        if self.verbalizer_dict is not None:
            self.validate_verbalizer()
        self.freeze_param()

    def freeze_param(self):
        print("freezing the parameters of language model....")
        for param in self.lm_model.parameters():
            param.requires_grad = False

    def validate_verbalizer(self):
        cp_verbalizer = copy.deepcopy(self.verbalizer_dict)
        positive_words = [self.tokenizer.tokenize(x)[0] for x in self.verbalizer_dict['pos']]
        negative_words = [self.tokenizer.tokenize(x)[0] for x in self.verbalizer_dict['neg']]
        cp_verbalizer['pos'] = positive_words
        cp_verbalizer['neg'] = negative_words
        for token in positive_words:
            assert token in self.tokenizer.encoder, f"{token} not in the vocabulary!"
        for token in negative_words:
            assert token in self.tokenizer.encoder, f"{token} not in the vocabulary!"
        
        self.verbalizer_dict = cp_verbalizer

    def preprocess_input(self, input_list: List[str], template: SentenceTemplate):
        if self.sentence_pair:
            assert type(input_list[0]) == list
            text_a_list = [x[0] for x in input_list]
            text_b_list = [x[1] for x in input_list]
            x_prompt = template(text_a_list, text_b_list, tokenizer = self.tokenizer)
        else:
            x_prompt = template(input_list, tokenizer = self.tokenizer)
        return x_prompt

    def wrap(self, input_list: List[str], tempalte: str):
        assert '[INPUT]' in tempalte and '<mask>' in tempalte
        if self.sentence_pair:
            raise NotImplementedError
        else:
            x_prompt = [tempalte.replace('[INPUT]', x) for x in input_list]
        return x_prompt

    def locate_output_token(self, input_ids: torch.LongTensor):
        output_mask = input_ids.eq(self.tokenizer.mask_token_id)
        num_output = torch.count_nonzero(output_mask.int(), dim = 1)
        assert (num_output == 1).all()
        return output_mask
    
    def verbalize(self, token_probs: torch.Tensor):
        positive_words = self.verbalizer_dict['pos']
        negative_words = self.verbalizer_dict['neg']
        
        positive_ids = self.tokenizer.convert_tokens_to_ids(positive_words)
        negative_ids = self.tokenizer.convert_tokens_to_ids(negative_words)

        positive_ids = torch.LongTensor(positive_ids).to(token_probs.device)
        negative_ids = torch.LongTensor(negative_ids).to(token_probs.device)

        positive_probs = token_probs.index_select(dim = 1, index = positive_ids)
        negative_probs = token_probs.index_select(dim = 1, index = negative_ids)

        positive_prob = torch.sum(positive_probs, dim = 1)
        negative_prob = torch.sum(negative_probs, dim = 1)

        pred_labels = (positive_prob > negative_prob).int()

        return positive_probs, negative_probs, positive_prob, negative_prob, pred_labels


    def predict(self, input_list, template: SentenceTemplate, use_verbalizer = False):
        assert template.output_token == self.tokenizer.mask_token
        if not self.sentence_pair:
            if type(input_list) == list:
                x_prompt = self.preprocess_input(input_list, template)
            elif type(input_list) == str:
                x_prompt = self.preprocess_input([input_list], template)
            else:
                raise NotImplementedError
        else:
            if type(input_list[0]) == str:
                 x_prompt = self.preprocess_input([input_list], template)           
            elif type(input_list[0]) == list:
                x_prompt = self.preprocess_input(input_list, template)
            else:
                raise NotImplementedError
        
        tokenized = self.tokenizer(x_prompt, padding = 'longest', return_tensors = "pt", return_attention_mask = True, return_token_type_ids = True,
                                   truncation = True, max_length = 512)
        tokenized = tokenized.to(self.device)
        input_ids = tokenized['input_ids']
        batch_size, seq_len = input_ids.size()
        output = self.lm_model(**tokenized)

        logits = output.logits
        output_token_mask = self.locate_output_token(input_ids,)

        flat_logits = logits.view(batch_size * seq_len, -1)
        flat_mask = output_token_mask.view(-1)
        output_token_logits = flat_logits[flat_mask]
        output_token_probs = F.softmax(output_token_logits, dim = -1)
        assert output_token_mask.size(0) == batch_size, f"{output_token_mask.size(0)} -- {batch_size}"

        if use_verbalizer:
            positive_probs, negative_probs, positive_prob, negative_prob, pred_labes = self.verbalize(output_token_logits, )
            return VTuningOutput(positive_probs, negative_probs, positive_prob, negative_prob,pred_labes, output_token_probs, output_token_logits)
        else:
            return VTuningOutput(all_token_probs = output_token_probs, all_token_logits = output_token_logits)

    
   
    def predict_v2(self, input_list, template, use_verbalizer = False):
        if not self.sentence_pair:
            if type(input_list) == list:
                x_prompt = self.wrap(input_list, template)
            elif type(input_list) == str:
                x_prompt = self.wrap([input_list], template)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        tokenized = self.tokenizer(x_prompt, padding = 'longest', return_tensors = "pt", return_attention_mask = True, return_token_type_ids = True,
                                   truncation = True, max_length = 512)
        tokenized = tokenized.to(self.device)
        input_ids = tokenized['input_ids']
        batch_size, seq_len = input_ids.size()
        output = self.lm_model(**tokenized)

        logits = output.logits
        output_token_mask = self.locate_output_token(input_ids,)

        flat_logits = logits.view(batch_size * seq_len, -1)
        flat_mask = output_token_mask.view(-1)
        output_token_logits = flat_logits[flat_mask]
        output_token_probs = F.softmax(output_token_logits, dim = -1)
        assert output_token_mask.size(0) == batch_size, f"{output_token_mask.size(0)} -- {batch_size}"

        if use_verbalizer:
            positive_probs, negative_probs, positive_prob, negative_prob, pred_labes = self.verbalize(output_token_logits, )
            return VTuningOutput(positive_probs, negative_probs, positive_prob, negative_prob,pred_labes, output_token_probs, output_token_logits)
        else:
            return VTuningOutput(all_token_probs = output_token_probs, all_token_logits = output_token_logits)


class OPTVTuningClassification(BaseModel):
    def __init__(self, model_type, cache_dir = None, finetune_dir = None, num_labels = 2, max_length = 512, sentence_pair = False,
                device = torch.device('cuda'), verbalizer_dict = None,
                ):
        super().__init__(num_labels, max_length)
        self.model_type = model_type
        self.cache_dir = cache_dir
        self.finetune_dir = finetune_dir
        self.num_lables = num_labels
        self.max_length = max_length
        self.sentence_pair = sentence_pair
        if self.finetune_dir == None:
            lm_model = OPTForCausalLM.from_pretrained(cache_dir)
            self.tokenizer = GPT2Tokenizer.from_pretrained(cache_dir)
        else:
            lm_model = OPTForCausalLM.from_pretrained(self.finetune_dir)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.finetune_dir)

        self.tokenizer.mask_token = self.tokenizer.eos_token
        self.tokenizer.mask_token_id = self.tokenizer.eos_token_id

        self.device = device
        self.verbalizer_dict = verbalizer_dict

        self.lm_model = lm_model.to(device)
        self.lm_model.eval()

        self.word2idx = self.tokenizer.get_vocab()
        print("vocab size: ", len(self.word2idx))

        if self.verbalizer_dict is not None:
            self.validate_verbalizer()
        self.freeze_param()

    def freeze_param(self):
        print("freezing the parameters of language model....")
        for param in self.lm_model.parameters():
            param.requires_grad = False

    def validate_verbalizer(self):
        cp_verbalizer = copy.deepcopy(self.verbalizer_dict)
        positive_words = [self.tokenizer.tokenize(x)[0] for x in self.verbalizer_dict['pos']]
        negative_words = [self.tokenizer.tokenize(x)[0] for x in self.verbalizer_dict['neg']]
        cp_verbalizer['pos'] = positive_words
        cp_verbalizer['neg'] = negative_words
        for token in positive_words:
            assert token in self.tokenizer.encoder, f"{token} not in the vocabulary!"
        for token in negative_words:
            assert token in self.tokenizer.encoder, f"{token} not in the vocabulary!"
        
        self.verbalizer_dict = cp_verbalizer
        print(self.verbalizer_dict)

    def preprocess_input(self, input_list: List[str], template: SentenceTemplate):
        if self.sentence_pair:
            assert type(input_list[0]) == list
            text_a_list = [x[0] for x in input_list]
            text_b_list = [x[1] for x in input_list]
            x_prompt = template(text_a_list, text_b_list)
        else:
            x_prompt = template(input_list)
        return x_prompt

    def locate_output_token(self, input_ids: torch.LongTensor):
        pad_mask = input_ids.ne(self.tokenizer.pad_token_id)
        batch_length = torch.count_nonzero(pad_mask.int(), dim = 1)
        output_loc = batch_length - 1
        output_mask = torch.zeros_like(input_ids).bool().to(input_ids.device)
        for i in range(output_mask.size(0)):
            output_mask[i, output_loc[i]] = True
        return output_mask
    
    def verbalize(self, token_probs: torch.Tensor):
        positive_words = self.verbalizer_dict['pos']
        negative_words = self.verbalizer_dict['neg']
        
        positive_ids = self.tokenizer.convert_tokens_to_ids(positive_words)
        negative_ids = self.tokenizer.convert_tokens_to_ids(negative_words)

        positive_ids = torch.LongTensor(positive_ids).to(token_probs.device)
        negative_ids = torch.LongTensor(negative_ids).to(token_probs.device)

        positive_probs = token_probs.index_select(dim = 1, index = positive_ids)
        negative_probs = token_probs.index_select(dim = 1, index = negative_ids)

        positive_prob = torch.sum(positive_probs, dim = 1)
        negative_prob = torch.sum(negative_probs, dim = 1)

        pred_labels = (positive_prob > negative_prob).int()

        return positive_probs, negative_probs, positive_prob, negative_prob, pred_labels

    def predict(self, input_list, template: SentenceTemplate, use_verbalizer = False):
        assert template.output_token == self.tokenizer.mask_token
        if type(input_list) == list:
            x_prompt = self.preprocess_input(input_list, template)
        elif type(input_list) == str:
            x_prompt = self.preprocess_input([input_list], template)
        else:
            raise NotImplementedError
        tokenized = self.tokenizer(x_prompt, padding = 'longest', return_tensors = "pt", return_attention_mask = True, return_token_type_ids = False,
                                    truncation = True, max_length = 512,
                                    )
        tokenized = tokenized.to(self.device)
        input_ids = tokenized['input_ids']
        batch_size, seq_len = input_ids.size()

        with torch.no_grad():
            output = self.lm_model(**tokenized)

        logits = output.logits
        output_token_mask = self.locate_output_token(input_ids,)

        flat_logits = logits.view(batch_size * seq_len, -1)
        flat_mask = output_token_mask.view(-1)
        output_token_logits = flat_logits[flat_mask]
        output_token_probs = F.softmax(output_token_logits, dim = -1)
        assert output_token_mask.size(0) == batch_size, f"{output_token_mask.size(0)} -- {batch_size}"

        if use_verbalizer:
            positive_probs, negative_probs, positive_prob, negative_prob, pred_labes = self.verbalize(output_token_probs, )
            return VTuningOutput(positive_probs, negative_probs, positive_prob, negative_prob,pred_labes, output_token_probs, output_token_logits)
        else:
            return VTuningOutput(all_token_probs = output_token_probs, all_token_logits = output_token_logits)

class MLPClassificationHead(nn.Module):
    def __init__(self, mlp_layer_dim = 128, mlp_layer_num = 3, output_dim = 2, input_dim = 50000):
        super().__init__()
        self.mlp_layer_dim = mlp_layer_dim
        self.mlp_layer_num = mlp_layer_num
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.module_list = nn.ModuleList(
            [nn.Linear(in_features = input_dim if i == 0 else mlp_layer_dim, out_features = mlp_layer_dim)
            for i in range(self.mlp_layer_num)]
        )
        self.output_layer = nn.Linear(self.mlp_layer_dim, output_dim)
        self.dropout = nn.Dropout(p = 0.5)
        self.act_fn = nn.ReLU()
    
    def forward(self, input_x: torch.FloatTensor):
        for i in range(self.mlp_layer_num):
            hidden = self.module_list[i](input_x)
            hidden = self.act_fn(hidden)
            input_x = hidden
        output = self.output_layer(input_x)
        return output
