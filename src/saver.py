import numpy as np
import os
from .utils import ROOT_DIR
from .template import SentenceTemplate
import pickle
import torch

class PredictionSaver():
    '''
    PredictionSaver: We rely on the language model's output prediction over [MASK] token. Note that for the same tempalte, the output
    is always the same and we can reuse it. Therefore, this class is used to cache the output prediction of LMs for weak learner training.
    
    This saver only save train/validation prediction. For test set prediction, we use TestPredictionSaver.
    '''
    def __init__(self, save_dir = os.path.join(ROOT_DIR,'cached_preds/'), model_name = 'roberta', use_logits = False, fewshot = False, low = False, fewshot_k = 0, fewshot_seed = 0):
        assert not (fewshot and low), "fewshot and low resource can not be true simutaneously!"
        self.save_dir = save_dir
        self.model_name = model_name
        self.use_logits = use_logits
        self.fewshot = fewshot
        self.low = low
        self.fewshot_k = fewshot_k
        self.fewshot_seed = fewshot_seed
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def save_preds(self, template:SentenceTemplate, train_preds, valid_preds):
        template_name = template.template_name
        if self.model_name != 'roberta':
            template_name += f"_{self.model_name}"
        if self.fewshot:
            template_name += f'_fs_{self.fewshot_k}shot_seed{self.fewshot_seed}'
        elif self.low:
            template_name += f'_low{self.fewshot_k}_seed{self.fewshot_seed}'
        if os.path.exists(os.path.join(self.save_dir, f"{template_name}.pkl")):
            print("already exists! Will not save it")
        else:
            with open(os.path.join(self.save_dir, f"{template_name}.pkl"), 'wb') as f:
                pickle.dump((train_preds, valid_preds), f)
    
    def load_preds(self, template: SentenceTemplate):
        template_name = template.template_name
        if self.model_name != 'roberta':
            template_name += f"_{self.model_name}"
        if self.fewshot:
            template_name += f'_fs_{self.fewshot_k}shot_seed{self.fewshot_seed}'
        elif self.low:
            template_name += f'_low{self.fewshot_k}_seed{self.fewshot_seed}'
        if os.path.exists(os.path.join(self.save_dir, f"{template_name}.pkl")):
            with open(os.path.join(self.save_dir, f"{template_name}.pkl"), 'rb') as f:
                (train_preds, valid_preds) = pickle.load(f)            
            return (train_preds, valid_preds), True
        else:
            print(f"did not find file ", os.path.join(self.save_dir, f"{template_name}.pkl"))
            return (), False

class TestPredictionSaver():
    def __init__(self, save_dir = os.path.join(ROOT_DIR, 'cached_preds/'), model_name = 'roberta', use_logits = False, fewshot = False, fewshot_k = 0, fewshot_seed = 0):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok = True)
        self.model_name = model_name
        self.use_logits = use_logits
        self.fewshot = fewshot
        self.fewshot_k = fewshot_k
        self.fewshot_seed = fewshot_seed
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def save_preds(self, template:SentenceTemplate, test_preds):
        template_name = template.template_name
        if self.model_name != 'roberta':
            template_name += f"_{self.model_name}"
        if self.use_logits:
            template_name += '_logits'

        save_name = os.path.join(self.save_dir, f"{template_name}.pkl")
        with open(save_name, 'wb') as f:
            pickle.dump(test_preds, f)
        del test_preds
        torch.cuda.empty_cache() 
    
    def load_preds(self, template: SentenceTemplate):
        template_name = template.template_name
        if self.model_name != 'roberta':
            template_name += f"_{self.model_name}"
        if self.use_logits:
            template_name += '_logits'
        save_addr = os.path.join(self.save_dir, f"{template_name}.pkl")
        if not os.path.exists(save_addr):
            return [], False
        with open(save_addr, 'rb') as f:
            test_preds = pickle.load(f)
        return test_preds, True

