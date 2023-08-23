'''
using chatgpt API for prompt boosting.

there are several kinds of prompts:
- solve_prompt: for solving the downstream task, e.g., news classification task (containing the initial prompt and generated prompts).
- feedback_prompt: for feedback


agnews dataset: see https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset
train: 64, valid: 64, test: 7600. tuple: (X, y)
e.g.,
X = [
    'Peace Doves Dropped on South Thailand PATTANI, Thailand, December 5 (IslamOnline.net  amp; News Agencies) - A fleet of military and civil aircraft and helicopters dropped Sunday, December 5, some 120 million paper birds across Thailands troubled Muslim-majority south, described by activists as ',
    'Israel Feuds With Agency Set Up to Aid Palestinians For years, Israel has feuded with the United Nations refugee agency for Palestinians over a wide range of issues, and recently Israel thought it had found a smoking gun to press its case.',
    'Saudi Police Kill Suspected Militant in Jeddah Saudi officials say security forces have killed a suspected militant in the western city of Jeddah after the man tried to use a hand grenade against them.',
    'Iraqi Militants Say They Shot Italian BAGHDAD, Iraq - Iraqi militants said they shot and killed an Italian citizen after he tried to break through a guerrilla roadblock on a highway outside the insurgent stronghold of Ramadi.',
    "Lockheed and the Future of Warfare In the post-9/11 world, Lockheed Martin has become more than just the nation's biggest military contractor. It is putting its stamp on military policies as well.",
    ...
]
y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
'''

import os
import re
import math
import random
import argparse
import requests
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd
from src.data_util import load_dataset, load_dataset_zcr
from utils import print_args, shuffle_dataset_binary, shuffle_dataset_triple, id2word_agnews, word2id_agnews, id2word_ethos_liar, word2id_ethos_liar, word2id_qnli_rte, get_num_classes
from template import solve_prompt_template, feedback_prompt_agnews, instruction0_agnews


APPID = "YOUR APPID HERE"
URL = "YOUR API URL HERE"


class ResPromptBooster():
    def __init__(self, args, train_dataset, valid_dataset, test_dataset):
        self.set_random_seed(args.fewshot_seed)

        self.args = args
        self.num_monte_carlo = args.num_monte_carlo
        self.num_feedbacks = args.num_feedbacks
        self.adaboost_lr = args.adaboost_lr
        self.max_error = args.max_error
        self.patience = args.patience

        self.k = args.fewshot_k
        self.train_X = train_dataset[0][:self.k]
        self.train_y = train_dataset[1][:self.k]
        self.num_train = len(train_dataset[0])
        self.num_test = args.num_test
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.average_mode = args.average_mode

        self.ins_weight_tensor = np.ones(self.k) / self.k
        self.used_solve_prompts = []
        self.model_weight_tensor = []
        self.used_train_y_pred = []
        self.instruction0_agnews = instruction0_agnews
        self.init_flag = True
        self.dataset_name = args.dataset
        self.num_classes = get_num_classes(self.dataset_name)
        self.timeout = args.timeout

        if self.dataset_name == 'agnews':
            self.word2id = word2id_agnews
        elif self.dataset_name in ('ethos', 'liar'):
            self.word2id = word2id_ethos_liar
        elif self.dataset_name in ('qnli', 'rte'):
            self.word2id = word2id_qnli_rte
        else:
            print(f'Unknown dataset: {self.dataset_name}')
            raise NotImplementedError
        self.target = list(self.word2id.keys())
        self.pattern = '|'.join(self.target)


    def solve_instance_single(self, solve_prompt, ins):
        '''Process single instance.
        return '-1' for failed status, other ids denote successed status.
        '''
        assert '{text}' in solve_prompt
        conversion = {"appId": APPID}
        try:
            messages = [
                {
                    "role": "user",
                    "content": solve_prompt.replace('{text}', ins)
                }
            ]
            conversion['messages'] = messages
            response = requests.post(URL, json=conversion, timeout=self.timeout)
            result = response.json()["data"]["result"]
            result = result.strip()

            find_list = re.findall(self.pattern, result)
            if len(find_list) == 0:
                return '-1'
            pred_cate = find_list[0]
            pred_id = self.word2id.get(pred_cate, '-1')
            return pred_id

        except Exception as e:
            print(f'error occurred in solve_instance_single: {e}')
            return '-1'


    def solve_task_single(self, solve_prompt, input_text_list, label_list, mode='train'):
        '''
        Use chatgpt for solving downstream tasks. e.g., news topic classification for AG's News.
        '''
        '''NOTE: failed/successed denotes the api calling whether successed, rather the classification task.'''
        y_pred = []
        random_guess_cnt = 0
        for ins_idx, ins in tqdm(enumerate(input_text_list)):
            pred_id = '-1'
            try_cnt = 0
            while pred_id == '-1':
                pred_id = self.solve_instance_single(solve_prompt, ins)
                try_cnt += 1
                if try_cnt >= self.patience:
                    '''random guess...'''
                    pred_id = '0'
                    random_guess_cnt += 1
            y_pred.append(int(pred_id))


        labels = np.array(label_list)
        y_pred = np.array(y_pred)
        assert len(labels) == len(y_pred)

        acc = np.sum(labels == y_pred) / len(labels)
        if self.num_classes == 2:
            precision = precision_score(labels, y_pred)
            recall = recall_score(labels, y_pred)
            f1 = f1_score(labels, y_pred)
        else:
            precision = precision_score(labels, y_pred, average=self.average_mode)
            recall = recall_score(labels, y_pred, average=self.average_mode)
            f1 = f1_score(labels, y_pred, average=self.average_mode)

        wrong_indices, wrong_flags_float, error = None, None, None
        if mode == 'train':
            wrong_flags = labels != y_pred
            wrong_indices = np.where(wrong_flags)[0]

            wrong_flags_float = wrong_flags.astype(float)
            assert self.ins_weight_tensor.shape == wrong_flags_float.shape
            error = np.sum(wrong_flags_float * self.ins_weight_tensor)

        return wrong_indices, wrong_flags_float, labels, y_pred, error, acc, precision, recall, f1


    def feedback_generate(self, wrong_indices, labels, y_pred, old_solve_prompt):
        '''
        - input_text: X
        - labels: y
        - C for category (i.e., text-style label)
        '''
        conversion = {"appId": APPID}

        def id2text(id):
            return id2word_agnews[str(id)]

        wrong_X = [self.train_X[idx]    for idx in list(wrong_indices)]
        wrong_C = [id2text(y_pred[idx]) for idx in list(wrong_indices)]
        right_C = [id2text(labels[idx]) for idx in list(wrong_indices)]
        assert len(wrong_X) == len(wrong_C) == len(right_C)

        error_string = '\n'.join([
            f'"{wrong_X[i]}" was wrongly classified as "{wrong_C[i]}" but should have been classified as "{right_C[i]}"' for i in range(len(wrong_X))
        ])
        
        feedback_message = feedback_prompt_agnews.replace('{prompt}', old_solve_prompt).replace('{error_string}', error_string).replace('{num_feedbacks}', str(self.num_feedbacks))


        generate_result = None
        try:
            messages = [
                {
                    "role": "user",
                    "content": feedback_message
                }
            ]

            conversion['messages'] = messages
            response = requests.post(URL, json=conversion, timeout=self.timeout)
            feedback_result = response.json()["data"]["result"]
            feedback_result = feedback_result.strip()
            print(f'feedback_result:\n{feedback_result}')

            system_info = {
                "role": "system",
                "content": feedback_result
            }
            generate_info = {
                "role": "user",
                "content": '''Based on the above reasons of reflection, please generation {} new prompts for this task. (Note that the prompt itself in the news classification task should not contain category information, such as Business or Sports.)'''.format(args.generate_cnt)
            }
            messages += [system_info, generate_info]

            conversion['messages'] = messages
            generated_prompts = []

            response = requests.post(URL, json=conversion, timeout=self.timeout)
            print(f'[generate] response: {response}')
            generate_result = response.json()["data"]["result"]
            print(f'generate_result: {generate_result}')
        except Exception as e:
            print(f'During generation, an error occurred: {e}')

        if not generate_result:
            return []
        generated_prompts = re.split(r'\n+', generate_result)
        generated_prompts = [re.sub(r"^\d+\.\s*", "", text) for text in generated_prompts]
        
        print(f'generated_prompts: {generated_prompts}')
        generated_prompts = [p.strip() for p in generated_prompts if len(p.strip()) > 0]
        new_instructions = random.sample(generated_prompts, self.num_monte_carlo)
        return new_instructions
    

    def do_feedback_generate(self, wrong_indices, labels, y_pred, old_solve_prompt):
        '''wrap feedback_generate func with timeout-retry mechanism.'''
        res = []
        try_cnt = 0
        while len(res) == 0:
            res = self.feedback_generate(wrong_indices, labels, y_pred, old_solve_prompt)
            try_cnt += 1
            if try_cnt >= self.patience:
                print('Tooooo many retry in do_feedback_generate!')
        return res


    def build_solve_prompt(self, instruction):
        '''build solve_prompt based on `instruction` and solve_prompt_template'''
        return solve_prompt_template.replace('{instruction}', instruction)


    def boost(self):
        prompt0_agnews = self.build_solve_prompt(self.instruction0_agnews)
        self.used_solve_prompts.append(prompt0_agnews)
        old_instruction = self.instruction0_agnews
        new_solve_prompt = None

        for weaker_id in tqdm(range(self.args.adaboost_weak_cls)):
            wrong_indices, wrong_flags_float, labels, y_pred, error, acc, precision, recall, f1 = self.solve_task_single(
                prompt0_agnews if self.init_flag else new_solve_prompt,
                self.train_X,
                self.train_y
            )
            print(f'weaker#[{weaker_id}] training acc: {acc} precision: {precision:.4f} recall: {recall:.4f} f1: {f1:.4f}')

            if not self.init_flag and len(wrong_indices) == 0:
                print(f'ALL training instances are solved! We have {len(self.model_weight_tensor)} weaker(s). Early Stop!')
                self.final_evaluate()
            else:
                '''Check the quality of the weaker: only good weakers will be ensembled.'''
                if self.init_flag or error < self.max_error:
                    print(f"{'*' * 10} Weaker [{weaker_id}] Adaboosting! {'*' * 10}")
                    alpha = self.adaboost_step(error, wrong_flags_float)
                    self.used_train_y_pred.append(y_pred)
                    if len(self.model_weight_tensor) > 1:
                        self.used_solve_prompts.append(new_solve_prompt)
                        ensemble_acc, precision, recall, f1 = self.ensemble_result(self.used_train_y_pred, labels)
                        print(f'[train] [ensemble] acc: {ensemble_acc:.4f} precision: {precision:.4f} recall: {recall:.4f} f1: {f1:.4f}')
                        if ensemble_acc > args.eval_trigger_threshold:
                            self.final_evaluate()

                new_instructions = self.do_feedback_generate(wrong_indices, labels, y_pred, old_instruction)
                new_instruction = new_instructions[0]
                new_solve_prompt = self.build_solve_prompt(new_instruction)
                old_instruction = new_instruction

            if self.init_flag:
                self.init_flag = False

        self.final_evaluate()


    def adaboost_step(self, error, wrong_flags_float):
        alpha = (math.log((1 - error) / error) + math.log(self.num_classes - 1)) * self.adaboost_lr
        weight_multiplier = np.exp(alpha * wrong_flags_float)
        self.ins_weight_tensor *= weight_multiplier
        self.ins_weight_tensor = self.ins_weight_tensor / np.sum(self.ins_weight_tensor)
        self.model_weight_tensor.append(alpha)
        return alpha


    def ensemble_result(self, y_pred_list, labels):
        '''
        Conduct ensemble for the final results.
        Ensemble training predictions.
        NOTE: this function is shared by training and testing.
        '''
        num_instance = len(labels)
        ensemble_score = np.zeros((num_instance, self.num_classes))
        assert len(y_pred_list) == len(self.model_weight_tensor) == len(self.used_solve_prompts)
        print(f'labels in ensemble_result: {labels}')

        y_pred_list = [np.reshape(arr, (-1, 1)) for arr in y_pred_list]
        y_pred_array = np.concatenate(y_pred_list, axis=1)

        for i in range(self.num_classes):
            curr_class_score = np.sum((y_pred_array == i).astype(float) * self.model_weight_tensor, axis=1)
            ensemble_score[:, i] = curr_class_score
        print(f'ensemble_score:\n{ensemble_score}')
        weighted_prediction = np.argmax(ensemble_score, axis=1)

        if self.num_classes == 2:
            precision = precision_score(labels, weighted_prediction)
            recall = recall_score(labels, weighted_prediction)
            f1 = f1_score(labels, weighted_prediction)
        else:
            precision = precision_score(labels, weighted_prediction, average=self.average_mode)
            recall = recall_score(labels, weighted_prediction, average=self.average_mode)
            f1 = f1_score(labels, weighted_prediction, average=self.average_mode)

        n_correct = np.sum(weighted_prediction == labels)
        ensemble_acc = n_correct / len(labels)
    
        return ensemble_acc, precision, recall, f1


    def final_evaluate(self):
        for p in self.used_solve_prompts:
            print(p)

        test_X = self.test_dataset[0][:self.num_test]
        test_y = self.test_dataset[1][:self.num_test]

        test_pred = []
        for p in self.used_solve_prompts:
            _, _, labels, y_pred, _, acc, precision, recall, f1 = self.solve_task_single(p, test_X, test_y, mode='test')
            test_pred.append(y_pred)

        ensemble_acc, precision, recall, f1 = self.ensemble_result(test_pred, labels)
        print(f'[test] [ensemble] acc: {ensemble_acc:.4f} precision: {precision:.4f} recall: {recall:.4f} f1: {f1:.4f}')

    def set_random_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaboost_lr", type=float, default=1.0)
    parser.add_argument("--max_error", type=float, default=0.8)
    parser.add_argument("--eval_trigger_threshold", type=float, default=0.9)
    parser.add_argument("--adaboost_weak_cls", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_feedbacks", type=int, default=8)
    parser.add_argument("--generate_cnt", type=int, default=4)
    parser.add_argument("--num_monte_carlo", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--dataset", type=str, default='rte', choices=['agnews', 'ethos', 'liar', 'snli', 'mnli', 'rte'])
    parser.add_argument("--sort_dataset", action='store_true')
    parser.add_argument("--fewshot", action='store_false')
    parser.add_argument("--fewshot_k", type=int, default=64)
    parser.add_argument("--low", action='store_true')
    parser.add_argument("--fewshot_seed", type=int, default=100, choices=[100, 13, 21, 42, 87])
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--num_test", type=int, default=20)
    # default: macro, only for multi-class datasets.
    parser.add_argument("--average_mode", type=str, default='macro', choices=['binary', 'micro', 'macro', 'weighted'])

    args = parser.parse_args()
    '''
    ------------------------------
    adaboost_lr: 1.0
    adaboost_weak_cls: 200
    dataset: agnews
    sort_dataset: False
    fewshot: True
    fewshot_k: 16
    low: False
    fewshot_seed: 100
    ------------------------------
    '''
    print_args(args)

    if args.dataset in ('agnews'):
        train_dataset, valid_dataset, test_dataset = load_dataset(
            dataset_name=args.dataset,
            sort_dataset=True,
            fewshot=args.fewshot,
            k=args.fewshot_k,
            rand_seed=args.fewshot_seed,
            low_resource=args.low
        )
    elif args.dataset in ('ethos', 'liar', 'rte'):
        dataset_dir = 'YOUR DATASET DIR HERE'
        dataset_dir = os.path.join(dataset_dir, args.dataset)
        train_path = os.path.join(dataset_dir, 'train.csv')
        valid_path = os.path.join(dataset_dir, 'valid.csv')
        test_path  = os.path.join(dataset_dir, 'test.csv')

        train_set = pd.read_csv(train_path)
        valid_set = pd.read_csv(valid_path)
        test_set  = pd.read_csv(test_path)

        print(f'#train_set: {len(train_set)}, #valid_set: {len(valid_set)}, #test_set: {len(test_set)}')

        train_dataset = tuple(train_set.values.transpose().tolist())
        valid_dataset = tuple(valid_set.values.transpose().tolist())
        test_dataset  = tuple(test_set.values.transpose().tolist())
    else:
        print(f'Unknown dataset: {args.dataset}')
        raise NotImplementedError

    if args.dataset in ('snli', 'mnli', 'qnli', 'rte'):
        train_dataset = shuffle_dataset_triple(train_dataset)
        valid_dataset = shuffle_dataset_triple(valid_dataset)
        test_dataset  = shuffle_dataset_triple(test_dataset)
    else:
        train_dataset = shuffle_dataset_binary(train_dataset)
        valid_dataset = shuffle_dataset_binary(valid_dataset)
        test_dataset  = shuffle_dataset_binary(test_dataset)
    print('The dataset is ready!')

    booster = ResPromptBooster(
        args,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset
    )

    booster.boost()