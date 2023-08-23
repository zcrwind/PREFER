import numpy as np
from typing import Dict, List
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import tqdm
import math
import pickle
import itertools
import time

from transformers import get_scheduler

from src.ptuning import BaseModel, MLPClassificationHead, RoBERTaVTuningClassification
from src.template import SentenceTemplate, TemplateManager, TemplateSaver
from src.saver import PredictionSaver, TestPredictionSaver
from src.label_set_util import generate_multicls_l1_label_set_with_cache
from src.utils import ROOT_DIR, BATCH_SIZE


class BaseMuticlsTrainer():
    def __init__(self, adaboost_lr = 1.0, num_classes = 2, use_logits = False):
        self.train_labels_by_model = []
        self.valid_labels_by_model = []
        self.test_labels_by_model = []

        self.dataset_weights = []
        self.model_weight_tensor = []

        self.best_ensemble_valid = 0
        self.best_epoch = -1
        self.adaboost_lr = adaboost_lr
        self.num_classes = num_classes
        self.use_logits = use_logits

        self.verbalizer_list = []
        self.template_name_list = []

    def save_prediction(self, pred_labels, split = 'train'):
        if split == 'train':
            if type(self.train_labels_by_model) == list:
                self.train_labels_by_model = pred_labels.unsqueeze(0)
            else:
                self.train_labels_by_model = torch.cat([self.train_labels_by_model, pred_labels.unsqueeze(0)])
        elif split == 'valid':
            if type(self.valid_labels_by_model) == list:
                self.valid_labels_by_model = pred_labels.unsqueeze(0)
            else:
                self.valid_labels_by_model = torch.cat([self.valid_labels_by_model, pred_labels.unsqueeze(0)])

        elif split == 'test':
            if type(self.test_labels_by_model) == list:
                self.test_labels_by_model = pred_labels.unsqueeze(0)
            else:
                self.test_labels_by_model = torch.cat([self.test_labels_by_model, pred_labels.unsqueeze(0)])
        
        else:
            raise NotImplementedError

    def ensemble_result(self, labels: torch.LongTensor, split = 'train', ensemble_num = 0):
        if ensemble_num <= 0:
            if split == 'train':
                labels_by_model = self.train_labels_by_model
            elif split == 'valid':
                labels_by_model = self.valid_labels_by_model
            elif split == 'test':
                labels_by_model = self.test_labels_by_model
            else:
                raise NotImplementedError
            model_weight_tensor = torch.tensor(self.model_weight_tensor).to(labels.device)

        else:
            if split == 'train':
                labels_by_model = self.train_labels_by_model[:ensemble_num]
            elif split == 'valid':
                labels_by_model = self.valid_labels_by_model[:ensemble_num]
            elif split == 'test':
                labels_by_model = self.test_labels_by_model[:ensemble_num]
            else:
                raise NotImplementedError
            model_weight_tensor = torch.tensor(self.model_weight_tensor[:ensemble_num]).to(labels.device)

        ## labels_by_model:  model_num * data_num
        ensemble_score = torch.zeros([labels_by_model.size(1), self.num_classes]).float().to(labels_by_model.device)
        model_weight_tensor = model_weight_tensor.view((-1,1))
        for i in range(self.num_classes):
            curr_class_score = torch.sum((labels_by_model == i).float() * model_weight_tensor, dim = 0)
            ensemble_score[:,i] = curr_class_score
        weighted_prediction = torch.argmax(ensemble_score, dim = 1)

        n_correct = torch.sum(weighted_prediction == labels)
        total = len(labels)
        acc = n_correct / total
        print(f"\tensemble: total {weighted_prediction.size(0)}, correct {n_correct}, accuracy {acc}")
        return acc

    '''
    在eval数据集上调用 vtuning_model.predict 进行预测, 逻辑简单清晰.
    template 只在 vtuning_model.predict 中使用到了.
    '''
    def pre_compute_logits(self, vtuning_model, template, eval_dataset, batch_size = None):
        sentence_list, label_list = eval_dataset
        if batch_size == None:
            print(f"using default batch size {BATCH_SIZE}")
            batch_size = BATCH_SIZE
        use_verbalizer = False
        num_batches = len(sentence_list) // batch_size

        all_probs = []

        for i in tqdm.tqdm(range(num_batches)):
            batch_input = sentence_list[i * batch_size: (i+1) * batch_size]
            model_output = vtuning_model.predict(batch_input, template, use_verbalizer)
            if self.use_logits:
                pred_probs = model_output.all_token_logits.detach().clone()
            else:
                pred_probs = model_output.all_token_probs.detach().clone()
            all_probs.append(pred_probs)
            del model_output
        
        if num_batches * batch_size < len(sentence_list):
            batch_input = sentence_list[num_batches * batch_size:]
            model_output = vtuning_model.predict(batch_input, template, use_verbalizer)
            if self.use_logits:
                pred_probs =  model_output.all_token_logits.detach().clone()
            else:
                pred_probs =  model_output.all_token_probs.detach().clone()
            all_probs.append(pred_probs)
            del model_output

        all_probs = torch.cat(all_probs, dim = 0)

        return all_probs

    def pre_compute_logits_v2(self, vtuning_model, template, eval_dataset, batch_size = None):
        sentence_list, label_list = eval_dataset
        if batch_size == None:
            print(f"using default batch size {BATCH_SIZE}")
            batch_size = BATCH_SIZE
        use_verbalizer = False
        num_batches = len(sentence_list) // batch_size

        all_probs = []

        for i in tqdm.tqdm(range(num_batches)):
            batch_input = sentence_list[i * batch_size: (i+1) * batch_size]
            model_output = vtuning_model.predict_v2(batch_input, template, use_verbalizer)
            if self.use_logits:
                pred_probs = model_output.all_token_logits.detach().clone()
            else:
                pred_probs = model_output.all_token_probs.detach().clone()
            all_probs.append(pred_probs)
            del model_output
        
        if num_batches * batch_size < len(sentence_list):
            batch_input = sentence_list[num_batches * batch_size:]
            model_output = vtuning_model.predict_v2(batch_input, template, use_verbalizer)
            if self.use_logits:
                pred_probs =  model_output.all_token_logits.detach().clone()
            else:
                pred_probs =  model_output.all_token_probs.detach().clone()
            all_probs.append(pred_probs)
            del model_output

        all_probs = torch.cat(all_probs, dim = 0)

        return all_probs

    def record_dataset_weights(self, weight_tensor: torch.FloatTensor):
        self.dataset_weights.append(weight_tensor.tolist())
    
    def adaboost_step(self, error, wrong_flags, weight_tensor):
        alpha = (math.log((1 - error)/error) + math.log(self.num_classes - 1)) * self.adaboost_lr
        weight_multiplier = torch.exp(alpha * wrong_flags)
        weight_tensor = weight_tensor * weight_multiplier
        weight_tensor = weight_tensor / torch.sum(weight_tensor)
        self.model_weight_tensor.append(alpha)
        return alpha, weight_tensor

    def save_dataset_weights(self):
        with open(ROOT_DIR + "dataset_weights/weight.pkl", 'wb') as f:
            pickle.dump(self.dataset_weights, f)

    def save_weak_learner(self, verbalizer, template_name):
        self.verbalizer_list.append(verbalizer)
        self.template_name_list.append(template_name)


    def analyze_acc_by_class(self, label_tensor, pred_tensor):
        for i in range(self.num_classes):
            class_mask = label_tensor == i
            corr_pred = pred_tensor[class_mask] == i
            total_curr_class = torch.sum(class_mask)
            total_corr = torch.sum(corr_pred)
            corr_acc = total_corr / total_curr_class
            print(f"class {i}: correct prediction: {total_corr}, wrong prediction: {total_curr_class - total_corr}, accuracy: {corr_acc}")

class PromptBoostingTrainer(BaseMuticlsTrainer):
    def __init__(self, adaboost_lr = 1.0, num_classes = 3, adaboost_maximum_epoch = 20000, use_logits = False):
        super().__init__(adaboost_lr, num_classes, use_logits)
        self.adaboost_maximum_epoch = adaboost_maximum_epoch

        self.template_list = []

    def train(self, dataset: List, vtuning_model: RoBERTaVTuningClassification,
              train_probs: torch.LongTensor, train_labels: torch.LongTensor, 
              weight_tensor: torch.FloatTensor, label_set_size: int, norm_class = False):
        label_map, token_scores = generate_multicls_l1_label_set_with_cache(
            dataset,
            vtuning_model,
            weight_list = weight_tensor.tolist(),
            cache_probs = train_probs,
            label_set_size = 0,
            num_classes = self.num_classes,
            norm_class = norm_class
        )
        for i in range(self.num_classes):
            class_mask = label_map == i
            token_scores[i,~class_mask] = -10000

        indices = torch.argsort(token_scores, dim = 1, descending = True)   ## num_classes, vocab_size
        class_token_indices = indices[:, :label_set_size]
        
        label_token_index_list = []
        label_token_list = []
        for i in range(self.num_classes):
            curr_token_index_list = class_token_indices[i].tolist()
            label_token_index_list.append(curr_token_index_list)
            label_tokens = vtuning_model.tokenizer.convert_ids_to_tokens(curr_token_index_list)
            label_token_list.append(label_tokens)

        verbalizer_pairs = list(itertools.product(*label_token_list))
        if self.num_classes == 2:  ## extend verbalizer
            extended_verbalizer_pairs = []
            extended_verbalizer_pairs += verbalizer_pairs
            for v_pair in verbalizer_pairs:
                reverse_pair = [v_pair[1], v_pair[0]]
                extended_verbalizer_pairs.append(reverse_pair)
            verbalizer_pairs = extended_verbalizer_pairs

        if self.adaboost_maximum_epoch > len(verbalizer_pairs):
            print(f"change maxmium epochs from {self.adaboost_maximum_epoch} to {len(verbalizer_pairs)}")
            candidate_size = len(verbalizer_pairs)
        else:
            candidate_size = self.adaboost_maximum_epoch
        selected_ids = np.random.choice(len(verbalizer_pairs), candidate_size, replace = False)
        
        best_error = 1
        worst_error = 0
        best_acc = 0
        best_verbalizer = None
        best_pred_labels = None
        best_wrong_flags = None

        word2idx = vtuning_model.tokenizer.get_vocab()
        for epoch in range(candidate_size):
            rand_verbalizer = verbalizer_pairs[selected_ids[epoch]]
            selected = [word2idx[rand_verbalizer[i]] for i in range(self.num_classes)]
            verbalizer = {i:rand_verbalizer[i] for i in range(self.num_classes)}
            wrong_flags, error, acc, pred_labels, train_logits = self.inference(train_probs, selected, train_labels, weight_tensor)

            if error < best_error:
                best_error = error
                best_acc = acc
                best_verbalizer = copy.deepcopy(verbalizer)
                best_selected = copy.deepcopy(selected)
                best_pred_labels = copy.deepcopy(pred_labels)
                best_wrong_flags = copy.deepcopy(wrong_flags)
            else:
                del train_logits
            if error > worst_error:
                worst_error = error
        print(f"error range: {best_error}-{worst_error}")
        return best_verbalizer, best_error, best_acc, best_wrong_flags, best_pred_labels

    def inference(self, eval_probs, verbalizer, eval_labels, weight_tensor):
        acc, pred_labels, logits = self.compute_acc(eval_probs, verbalizer, eval_labels, visualize = False)
        wrong_flags = (pred_labels != eval_labels).float()
        error = torch.sum(wrong_flags * weight_tensor).item()
        return wrong_flags, error, acc, pred_labels, logits

    def compute_acc(self, eval_probs, verbalizer: List[int], eval_labels, visualize = False):
        verbalizer_idxs = torch.LongTensor(verbalizer)
        logits = eval_probs[:, verbalizer_idxs]
        pred_labels = torch.argmax(logits, dim = 1).int()
        corr = (pred_labels == eval_labels).sum()
        acc = (corr / pred_labels.size(0)).item()
        if visualize:
            print(f"\ttotal {pred_labels.size(0)}, correct {corr}, accuracy {acc}")
        return acc, pred_labels, logits

    def evaluate(self, word2idx, eval_probs, verbalizer: Dict, eval_labels, visualize = True, analyze_pred = False):
        verbalizer_list = [word2idx[verbalizer[i]] for i in range(self.num_classes) ]
        acc, pred_labels, logits = self.compute_acc(eval_probs, verbalizer_list, eval_labels, visualize)
        if analyze_pred:
            self.analyze_acc_by_class(eval_labels, pred_labels)

        return acc, pred_labels, logits

    def final_eval(self, test_dataset: List, vtuning_model: RoBERTaVTuningClassification, template_list: List[SentenceTemplate],
                   saver: TestPredictionSaver):
        word2idx = vtuning_model.word2idx
        num_examples = len(test_dataset[0])
        test_labels = torch.LongTensor(test_dataset[1]).to(vtuning_model.device)
        all_pred_labels = torch.zeros([self.best_epoch, num_examples]).fill_(-1).long().to(vtuning_model.device)
        for template_idx in range(len(template_list)):
            curr_template = template_list[template_idx]
            template_name = curr_template.template_name
            model_ids = [x for x in range(self.best_epoch) if self.template_name_list[x] == template_name]
            if len(model_ids) == 0:
                continue
            num_weak_learner = len(model_ids)
            verbalizers = [self.verbalizer_list[x] for x in model_ids]
            label_token_list = [[word2idx[verbalizer[i]] for i in range(self.num_classes)] for verbalizer in verbalizers]
            label_token_tensor = torch.LongTensor(label_token_list).to(vtuning_model.device)
            cls_scores, flag = saver.load_preds(curr_template)
            # assert flag
            if not flag:
                print(f"Did not find LM's predictions on test set. Making forward passes on test set...")
                cls_scores = self.pre_compute_logits(vtuning_model, curr_template, test_dataset)
                saver.save_preds(curr_template, cls_scores)
            cls_predictions = cls_scores.index_select(dim = 1, index = label_token_tensor.view(-1))
            cls_predictions = cls_predictions.view(num_examples, num_weak_learner, self.num_classes)
            pred_labels = torch.argmax(cls_predictions, dim = -1).transpose(0, 1)
            del cls_scores

            model_ids = torch.LongTensor(model_ids).to(vtuning_model.device)
            all_pred_labels[model_ids,:] = pred_labels
        self.test_labels_by_model = all_pred_labels
        acc = self.ensemble_result(test_labels, split = 'test', ensemble_num = self.best_epoch)
        return acc

    '''zcr'''
    def final_eval_v2(
        self, test_dataset: List,
        vtuning_model: RoBERTaVTuningClassification,
        template_list: List[str],
        saver: TestPredictionSaver
    ):
        word2idx = vtuning_model.word2idx
        num_examples = len(test_dataset[0])
        test_labels = torch.LongTensor(test_dataset[1]).to(vtuning_model.device)
        all_pred_labels = torch.zeros([self.best_epoch, num_examples]).fill_(-1).long().to(vtuning_model.device)
        for template_idx in range(len(template_list)):
            curr_template = template_list[template_idx]
            model_ids = list(range(self.best_epoch))
            if len(model_ids) == 0:
                continue
            num_weak_learner = len(model_ids)
            verbalizers = [self.verbalizer_list[x] for x in model_ids]
            label_token_list = [[word2idx[verbalizer[i]] for i in range(self.num_classes)] for verbalizer in verbalizers]
            label_token_tensor = torch.LongTensor(label_token_list).to(vtuning_model.device)  ## num_weak_learner, num_classes

            cls_scores = self.pre_compute_logits_v2(vtuning_model, curr_template, test_dataset)

            cls_predictions = cls_scores.index_select(dim = 1, index = label_token_tensor.view(-1))
            cls_predictions = cls_predictions.view(num_examples, num_weak_learner, self.num_classes)
            pred_labels = torch.argmax(cls_predictions, dim = -1).transpose(0, 1) ## num_weak_learner, num_exmaples
            del cls_scores

            model_ids = torch.LongTensor(model_ids).to(vtuning_model.device)
            all_pred_labels[model_ids,:] = pred_labels
        self.test_labels_by_model = all_pred_labels
        acc = self.ensemble_result(test_labels, split = 'test', ensemble_num = self.best_epoch)
        return acc

    def save_weak_learner_residual(self, verbalizer, template):
        self.verbalizer_list.append(verbalizer)
        self.template_list.append(template)


class FeatureMLPTrainer():
    def __init__(self, mlp_layer_num, mlp_layer_dim, input_dim, output_dim, 
                lr, batch_size, num_epochs, num_examples, save_dir,
                device = torch.device("cuda")):
        self.mlp_layer_num = mlp_layer_num
        self.mlp_layer_dim = mlp_layer_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_examples = num_examples
        self.save_dir = save_dir
        self.save_path = self.save_dir + 'best_model.pt'

        self.device = device
        self.build_model()
        self.build_optim()

    def build_model(self,):
        self.mlp_model = MLPClassificationHead(mlp_layer_num = self.mlp_layer_num, mlp_layer_dim = self.mlp_layer_dim, 
                                                output_dim = self.output_dim, input_dim = self.input_dim, 
                                                ).to(self.device)

    def build_optim(self,):
        self.optimizer = torch.optim.AdamW(self.mlp_model.parameters(), lr = self.lr)
        # self.optimizer = torch.optim.SGD(soft_vt_model.parameters(), lr = lr)
        num_training_steps = self.num_examples * self.num_epochs / self.batch_size
        lr_scheduler = get_scheduler(
            'linear',
            optimizer = self.optimizer,
            num_warmup_steps = 0,
            num_training_steps = num_training_steps
        )
        self.lr_scheduler = lr_scheduler

    def train_epoch(self, train_probs: torch.FloatTensor, train_labels: torch.LongTensor):
        self.mlp_model.train()
        num_train = train_probs.size(0)
        num_batches = num_train // self.batch_size
        rand_idxs = np.random.choice(num_train, num_train, replace = False)

        loss_list = []
        pred_list = []
        total_correct = 0
        total_num = 0

        for i in range(num_batches):
            batch_idxs = torch.from_numpy(rand_idxs[i * self.batch_size: (i+1) * self.batch_size]).long().to(train_probs.device)
            batch_input = train_probs[batch_idxs]
            batch_labels = train_labels[batch_idxs]            
            pred_logits = self.mlp_model(batch_input)
            loss = F.cross_entropy(pred_logits, batch_labels, reduction = 'mean')
            loss_list.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            pred_labels = pred_logits.argmax(-1)
            corr = pred_labels.eq(batch_labels).sum()
            total_correct += corr
            total_num += batch_input.size(0)
            pred_list += pred_labels.tolist()
        if num_batches * self.batch_size < num_train:
            batch_idxs = torch.from_numpy(rand_idxs[num_batches * self.batch_size:]).long().to(train_probs.device)
            batch_input = train_probs[batch_idxs]
            batch_labels = train_labels[batch_idxs]
            
            pred_logits = self.mlp_model(batch_input)
            loss = F.cross_entropy(pred_logits, batch_labels, reduction = 'mean')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            pred_labels = pred_logits.argmax(-1)
            corr = pred_labels.eq(batch_labels).sum()
            total_correct += corr
            total_num += batch_input.size(0)
            pred_list += pred_labels.tolist()
            loss_list.append(loss.item())

        return np.mean(loss_list), total_correct / total_num

    def evaluate(self,train_probs: torch.FloatTensor, train_labels: torch.LongTensor,):
        self.mlp_model.eval()
        loss_list = []
        total_correct = 0
        total_num = 0
        num_eval = train_probs.size(0)
        num_batches = num_eval // self.batch_size
        all_preds = []
        for i in range(num_batches):
            batch_input = train_probs[i * self.batch_size: (i+1) * self.batch_size]
            batch_labels = train_labels[i * self.batch_size: (i+1) * self.batch_size]
            
            pred_logits = self.mlp_model(batch_input)
            loss = F.cross_entropy(pred_logits, batch_labels, reduction = 'mean')
            loss_list.append(loss.item())

            pred_labels = pred_logits.argmax(-1)
            corr = pred_labels.eq(batch_labels).sum()
            total_correct += corr
            total_num += batch_input.size(0)
            all_preds += pred_labels.tolist()
        if num_batches * self.batch_size < num_eval:
            batch_input = train_probs[num_batches * self.batch_size:]
            batch_labels = train_labels[num_batches * self.batch_size:]
            
            pred_logits = self.mlp_model(batch_input)
            loss = F.cross_entropy(pred_logits, batch_labels, reduction = 'mean')
            loss_list.append(loss.item())

            pred_labels = pred_logits.argmax(-1)
            corr = pred_labels.eq(batch_labels).sum()
            total_correct += corr
            total_num += batch_input.size(0)
            all_preds += pred_labels.tolist()
        all_preds = torch.LongTensor(all_preds).to(train_probs.device)
        return np.mean(loss_list), total_correct / total_num, all_preds

    def pre_compute_logits(self, vtuning_model, template, eval_dataset):
        sentence_list, label_list = eval_dataset
        batch_size = BATCH_SIZE
        num_batches = len(sentence_list) // batch_size

        all_probs = []

        for i in tqdm.tqdm(range(num_batches)):
            batch_input = sentence_list[i * batch_size: (i+1) * batch_size]
            model_output = vtuning_model.predict(batch_input, template)
            pred_probs =  model_output.all_token_probs.detach().clone()
            all_probs.append(pred_probs)
            del model_output
        
        if num_batches * batch_size < len(sentence_list):
            batch_input = sentence_list[num_batches * batch_size:]
            model_output = vtuning_model.predict(batch_input, template)
            pred_probs =  model_output.all_token_probs.detach().clone()
            all_probs.append(pred_probs)
            del model_output

        all_probs = torch.cat(all_probs, dim = 0)

        return all_probs

    def save_model(self):
        state_dict = self.mlp_model.state_dict()
        torch.save(state_dict, self.save_path)
    
    def load_model(self):
        best_model_state_dict = torch.load(self.save_path)
        self.mlp_model.load_state_dict(best_model_state_dict)
        return self.mlp_model

