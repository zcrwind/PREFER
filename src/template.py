from typing import List, Optional, Union

from transformers import PreTrainedTokenizer
from .utils import ROOT_DIR
import json
import copy
import os
import numpy as np
import random
import string

import requests
import re


class SentenceTemplateResidual():
    def __init__(self, output_token='[MASK]'):
        self.output_token = output_token

    def wrap(self, template_str, text_a, text_b=None):
        pass


class SentenceTemplate():
    def __init__(self, template_path, template_json_string = None, output_token = '[MASK]', read_from_raw_file = True):
        self.template_name = ''
        self.read_from_raw_file = read_from_raw_file
        if self.read_from_raw_file:
            self.template_path = template_path
            self.template_content, self.input_positions, self.output_position = self.parse_template_file(self.template_path)
        else:
            self.template_path = None
            self.template_content, self.input_positions, self.output_position = self.parse_json_str(template_json_string)
        if len(self.input_positions) == 2:
            self.sentence_pair = True
        elif len(self.input_positions) == 1:
            self.sentence_pair = False
        else:
            raise NotImplementedError
        self.output_token = output_token

    def parse_template_file(self, path):
        with open(path, 'r', encoding = 'utf-8') as f:
            content = f.read().strip()
            template_info = json.loads(content)
        template_content = []
        input_positions = []
        output_position = -1
        self.template_name = template_info['name']
        self.reverse_order = False
        if 'reverse_order' in template_info:
            self.reverse_order = template_info['reverse_order']
            print(f"reverse the order of sentence pairs: {self.reverse_order}")

        for i, desc_dict in enumerate(template_info['template']):
            meta = desc_dict['meta']
            if 'text' in meta:
                input_positions.append(i)
                template_content.append("{" + meta + "}")
            elif "output_token" in meta:
                output_position = i
                template_content.append("[P]")
            elif "prompt_segment" in meta:
                segment = desc_dict['content']
                template_content.append(segment)
            else:
                raise NotImplementedError
        return template_content, input_positions, output_position
    
    def parse_json_str(self, json_str):
        assert not self.read_from_raw_file
        template_info = json.loads(json_str)
        template_content = []
        input_positions = []
        output_position = -1
        self.template_name = template_info['name']
        self.reverse_order = False

        for i, desc_dict in enumerate(template_info['template']):
            meta = desc_dict['meta']
            if 'text' in meta:
                input_positions.append(i)
                template_content.append("{" + meta + "}")
            elif "output_token" in meta:
                output_position = i
                template_content.append("[P]")
            elif "prompt_segment" in meta:
                segment = desc_dict['content']
                template_content.append(segment)
            else:
                raise NotImplementedError
        return template_content, input_positions, output_position

    def visualize(self):
        print("template: ", ''.join(self.template_content))
        return ''.join(self.template_content)

    def format_sp_input(self, text_a, text_b, prompt_before_texta, prompt_after_texta, prompt_before_textb, prompt_after_textb):
        if prompt_before_texta != None:
            if prompt_before_texta[-1] in ['.','!','?','...']:
                text_a = " " + text_a
            else:
                text_a = text_a[0].lower() + text_a[1:]
                text_a = " " + text_a

        if prompt_after_texta[0] in string.punctuation:
            if text_a[-1] in string.punctuation:
                if text_a[-2] == " ":
                    text_a = text_a[:-2]    
                else:
                    text_a = text_a[:-1]
        
        if prompt_before_textb[-1] in ['.','!','?','...']:
            text_b = " " + text_b
        else:
            text_b = text_b[0].lower() + text_b[1:]
            text_b = " " + text_b
        
        if prompt_after_textb != None:
            if prompt_after_textb[0] in string.punctuation:
                if text_b[-1] in string.punctuation:
                    if text_b[-2] == " ":
                        text_b = text_b[:-2]    
                    else:
                        text_b = text_b[:-1]

        return text_a, text_b

    def format_input(self, text_a: str, prompt_after_texta):
        if text_a[-1] in string.punctuation:
            if text_a[-2] == " ":
                text_a = text_a[:-2] + text_a[-1]
        if prompt_after_texta[0] in string.punctuation:
            if text_a[-1] in string.punctuation:
                if text_a[-2] == " ":
                    text_a = text_a[:-2]    
                else:
                    text_a = text_a[:-1]
        return text_a

    def truncate(self, output_list, tokenizer: PreTrainedTokenizer, orig_length: int):
        max_length = tokenizer.model_max_length
        num_delete = orig_length - max_length + 20 
        if not self.sentence_pair:
            orig_sentence = output_list[self.input_positions[0]]
            token_list = tokenizer.tokenize(orig_sentence)
            shortened_token_list = token_list[:orig_length - num_delete]
            new_sentence = tokenizer.convert_tokens_to_string(shortened_token_list)
            output_list[self.input_positions[0]] = new_sentence
        else:
            sen1 = output_list[self.input_positions[0]]
            sen2 = output_list[self.input_positions[1]]
            token_list1 = tokenizer.tokenize(sen1)
            token_list2 = tokenizer.tokenize(sen2)
            for _ in range(num_delete):
                if len(token_list1) > len(token_list2):
                    token_list1.pop()
                else:
                    token_list2.pop()
            new_sen1 = tokenizer.convert_tokens_to_string(token_list1)
            new_sen2 = tokenizer.convert_tokens_to_string(token_list2)
            output_list[self.input_positions[0]] = new_sen1
            output_list[self.input_positions[1]] = new_sen2
        return output_list


    def get_output_list(self, text_a, text_b = None, tokenizer: PreTrainedTokenizer = None):
        output_list = copy.deepcopy(self.template_content)

        if self.sentence_pair:
            if self.input_positions[0] >= 1:
                prompt_before_texta = self.template_content[self.input_positions[0] - 1]
            else:
                prompt_before_texta = None
            if self.input_positions[1] < len(self.template_content) - 1:
                prompt_after_textb = self.template_content[self.input_positions[1] + 1]
                if len(prompt_after_textb) == 0:
                    prompt_after_textb = None
            else:
                prompt_after_textb = None
            if self.reverse_order:
                text_a, text_b = self.format_sp_input(text_b, text_a, prompt_before_texta, self.template_content[self.input_positions[0] + 1], self.template_content[self.input_positions[1] - 1], prompt_after_textb)            
            else:
                text_a, text_b = self.format_sp_input(text_a, text_b, prompt_before_texta, self.template_content[self.input_positions[0] + 1], self.template_content[self.input_positions[1] - 1], prompt_after_textb)
        else:
            if self.input_positions[0] < len(self.template_content) - 1:
                text_a = self.format_input(text_a, self.template_content[self.input_positions[0] + 1])
        output_list[self.input_positions[0]] = text_a
        if self.sentence_pair:
            if text_b == None:
                raise NotImplementedError
            output_list[self.input_positions[1]] = text_b
        '''For CausalLM, there is no output token (we will take the output on the last token) and the output_position is -1'''
        if self.output_position >= 0:
            output_list[self.output_position] = self.output_token
        output_sequence = ''.join(output_list)

        if tokenizer is not None:
            max_length = tokenizer.model_max_length - 2
            tokenized_sequence = tokenizer.tokenize(output_sequence)
            num_tokens = len(tokenized_sequence)
            if num_tokens > max_length:
                truncated_output_list = self.truncate(output_list, tokenizer, num_tokens)
                output_list = truncated_output_list
        return output_list


    def transform_input(self, text_a, text_b = None, tokenizer: PreTrainedTokenizer = None):
        output_list = self.get_output_list(text_a, text_b, tokenizer)
        output_sequence = ''.join(output_list)
        return output_sequence

    def __call__(self, text_a, text_b = None, tokenizer = None):
        if type(text_a) == list:
            if text_b == None:
                return [self.transform_input(text_a[i], tokenizer = tokenizer) for i in range(len(text_a))]
            else:
                return [self.transform_input(text_a[i], text_b[i], tokenizer = tokenizer) for i in range(len(text_a))]
        elif type(text_a) == str:
            return self.transform_input(text_a, text_b)
        else:
            raise NotImplementedError


class RandomSentenceTemplate():
    def __init__(self, output_token = '[MASK]', tokenizer: PreTrainedTokenizer = None, prompt_loc = 'end', candidate_length = [10, 20, 50, 100,],
                        rand_prompt_length = False, rand_mask_loc = False, prompt_length = 10, mask_loc = 0,
                        sentence_pair = False):
        self.template_name = ''
        self.tokenizer = tokenizer
        word2idx = self.tokenizer.get_vocab()
        idx2word = {v:k for k,v in word2idx.items()}
        self.vocab_list = [idx2word[idx] for idx in range(len(word2idx))]
        self.prompt_loc = prompt_loc
        self.candidate_length = candidate_length

        self.rand_prompt_length = rand_prompt_length
        self.rand_mask_loc = rand_mask_loc
        self.prompt_length = prompt_length
        self.mask_loc = mask_loc
        self.sentence_pair = sentence_pair

        self.template_content, self.input_positions, self.output_position = self.generate_template()
        self.output_token = output_token


    def generate_template(self,):
        rand_idx = np.random.choice(len(self.candidate_length))
        if not self.rand_prompt_length:
            rand_length = self.prompt_length
        else:
            rand_length = self.candidate_length[rand_idx]
        rand_token_list = []
        while True:
            if len(rand_token_list) >= rand_length:
                break
            rand_token_id = np.random.choice(len(self.vocab_list))
            token = self.vocab_list[rand_token_id]
            if not token.startswith("Ġ"):
                continue
            rand_token_list.append(token)

        if not self.rand_mask_loc:
            if self.mask_loc == -1:
                mask_token_pos = rand_length
            elif self.mask_loc == 0:
                mask_token_pos = 0
            else:
                raise NotImplementedError
        else: 
            mask_token_pos = np.random.choice(rand_length + 1)
        template_content = []
        input_position = []
        output_position = -1
        curr_loc = 0
        if self.prompt_loc == 'end':
            template_content.append("text_a")
            input_position.append(curr_loc)
            curr_loc += 1

        if mask_token_pos == 0:
            template_content.append("[P]")
            output_position = curr_loc
            curr_loc += 1
            template_segment = self.tokenizer.convert_tokens_to_string(rand_token_list)
            template_content.append(template_segment)
            curr_loc += 1
        else:
            template_segment = self.tokenizer.convert_tokens_to_string(rand_token_list[:mask_token_pos])
            template_content.append(template_segment)
            curr_loc += 1
            if mask_token_pos != rand_length:
                template_content.append("[P]")
                output_position = curr_loc
                curr_loc += 1
                template_content.append(self.tokenizer.convert_tokens_to_string(rand_token_list[mask_token_pos:]))
                curr_loc += 1
            else:
                template_content.append("[P]")
                output_position = curr_loc
                curr_loc += 1
        if self.prompt_loc == 'begin':
            template_content.append("text_a")
            input_position.append(curr_loc)
            curr_loc += 1
        if self.sentence_pair:
            template_content.append("text_b")
            input_position.append(curr_loc)
            curr_loc += 1
        print("template: ", ' '.join(template_content))
        print("input position", input_position)
        print("output position: ", output_position)
        return template_content, input_position, output_position


    def transform_input(self, input_sentence):
        output_sequence = copy.deepcopy(self.template_content)
        output_sequence[self.input_positions[0]] = input_sentence
        output_sequence[self.output_position] = self.output_token
        output_sequence = ' '.join(output_sequence)
        return output_sequence
    
    def __call__(self, input_sentence):
        if type(input_sentence) == list:
            return [self.transform_input(x) for x in input_sentence]
        elif type(input_sentence) == str:
            return self.transform_input(input_sentence)

class TemplateSaver():
    def __init__(self, template_path, template_suffix = ''):
        self.template_path = template_path
        self.template_suffix = template_suffix
        if not os.path.exists(self.template_path):
            os.makedirs(self.template_path)
        self.count_template()
        
    def count_template(self,):
        filenames = os.listdir(self.template_path)
        print(filenames)
        num_templates = len(filenames)
        self.num_templates = num_templates

    def save(self, template:SentenceTemplate):
        self.count_template()
        template_name = f"{self.template_suffix}_{self.num_templates + 1}"
        json_list = []
        segment_id = 1
        for i, content in enumerate(template.template_content):
            if i in template.input_positions:
                desc_dict = {"meta": "text_a"}
            elif i == template.output_position:
                desc_dict = {"meta": "output_token"}
            else:
                desc_dict = {"meta": f"prompt_segment{segment_id}", "content": content}
                segment_id += 1
            json_list.append(desc_dict)
        json_dict = {"name": template_name,"template": json_list}
        with open(os.path.join(self.template_path, f'{template_name}.json'), 'w', encoding = 'utf-8') as f:
            json.dump(json_dict, f, indent = 4)

    def save_template(self, template: Union[SentenceTemplate, RandomSentenceTemplate]):
        self.save(template)


class TemplateManager():
    def __init__(self, template_dir_list, dataset='agnews', output_token = '<mask>', max_template_num = 0, 
                 use_part_templates = False, start_idx = 0, end_idx = 10, rand_order = True,
                 single_template_file = False, filtered_template_ids = None,):
        self.template_dir_list = template_dir_list
        self.output_token = output_token
        self.max_template_num = max_template_num
        self.rand_order = rand_order
        self.single_template_file = single_template_file
        self.filtered_template_ids = filtered_template_ids

        if self.single_template_file:
            self.template_list = self.load_single_template_file()
        else:
            self.template_list = self.load_templates()
        print(f"{len(self.template_list)} templates loaded...")

        self.use_part_templates = use_part_templates
        self.start_idx = start_idx
        self.end_idx = end_idx

        if self.filtered_template_ids != None:
            self.template_list = [self.template_list[x] for x in self.filtered_template_ids]
            self.end_idx = len(self.template_list)

        if not self.use_part_templates:
            if self.rand_order:
                self.random_indices = np.random.choice(len(self.template_list), 100)
            else:
                self.random_indices = np.arange(len(self.template_list))
            self.curr_index = 0
        else:
            assert self.start_idx >= 0
            assert self.end_idx <= len(self.template_list), f"{self.end_idx}, {len(self.template_list)}"
            print(f"using templates from {self.start_idx} to {self.end_idx}")
            self.random_indices = np.arange(self.start_idx, self.end_idx)
            if self.rand_order:
                random.shuffle(self.random_indices)
            self.curr_index = 0

        self.dataset = dataset
        self.prompt0 = self.get_prompt0()
        self.template_list_residual = []
        self.template_list_residual.append(self.prompt0)


    def update_template_list(self, template_idxs: np.ndarray):
        self.random_indices = copy.deepcopy(template_idxs)
        if self.rand_order:
            random.shuffle(self.random_indices)
        self.curr_index = 0
    
    def infer_template_file_name(self, filenames: List[str]):
        
        first_template_name = filenames[0]
        basename = first_template_name[:-6]
        return basename

    def load_templates(self,) -> List[SentenceTemplate]:
        template_list = []
        for template_dir in self.template_dir_list:
            filenames = os.listdir(template_dir)
            base_filename = self.infer_template_file_name(filenames)
            for idx in range(len(filenames)):
                filename = f"{base_filename}{idx + 1}.json"
                file_addr = os.path.join(template_dir, filename)
                template = SentenceTemplate(template_path = file_addr, output_token = self.output_token)
                template_list.append(template)
        if self.max_template_num > 0:
            template_list = template_list[:self.max_template_num]
        return template_list

    def load_single_template_file(self, ) -> List[SentenceTemplate]:
        assert self.single_template_file
        assert type(self.template_dir_list) == str
        with open(self.template_dir_list, 'r', encoding = 'utf-8') as f:
            raw_templates = json.load(f)
        template_list = []
        for raw_template in raw_templates:
            json_str_template = json.dumps(raw_template)
            template = SentenceTemplate(template_json_string = json_str_template, output_token = self.output_token, read_from_raw_file = False)
            template_list.append(template)
        if self.max_template_num > 0:
            if self.rand_order:
                rand_template_idxs = np.random.choice(len(template_list), self.max_template_num)
                template_list = [template_list[x] for x in rand_template_idxs]
            else:
                template_list = template_list[:self.max_template_num]
        return template_list

        
    def change_rand_indices(self):
        if self.use_part_templates:
            if self.rand_order:
                random.shuffle(self.random_indices)
            self.curr_index = 0
        else:
            if self.rand_order:
                self.random_indices = np.random.choice(len(self.template_list), 100)
            self.curr_index = 0

    def get_template(self, index = 0):
        return self.template_list[index]

    def change_template(self, prev_template = None) -> SentenceTemplate:
        if not prev_template == None:
            del prev_template
        if self.curr_index >= len(self.random_indices):
            self.change_rand_indices()
        template = self.template_list[self.random_indices[self.curr_index]]
        self.curr_index += 1
        template.visualize()
        return template

    def get_all_template(self):
        if not self.use_part_templates:
            return self.template_list
        else:
            indices = np.arange(self.start_idx, self.end_idx)
            return [self.get_template(x) for x in indices]

    def get_all_template_v2(self):
        return self.template_list_residual

    def get_prompt0(self):
        if self.dataset == 'agnews':
            return '[INPUT] This entry was posted in <mask>.'
        elif self.dataset == 'trec':
            return '[INPUT] What is <mask>?'
        else:
            raise NotImplementedError


    def change_template_residual(self, last_prompt_template, input_data, labels, preds=None, wrong_flags=None):
        try:
            url = "YOUR API URL HERE"
            conversion = {"appId": "YOUR APPID HERE"}

            target_map = {'World': 0, 'Sports': 1, 'Business': 2, 'Science': 3}
            target_map_reverse = {'0': 'World', '1': 'Sports', '2': 'Business', '3': 'Science'}

            wrong_indices = (wrong_flags == 1.0).nonzero()
            wrong_indices = wrong_indices.squeeze().tolist()
            wrong_data = [input_data[i] for i in wrong_indices]
            wrong_pred_category_list = []
            for wrong_idx in wrong_indices:
                wrong_pred = str(int(preds[wrong_idx]))
                wrong_pred_category = target_map_reverse[wrong_pred]
                wrong_pred_category_list.append(wrong_pred_category)

            wrong_batch = [wrong_data[i] + " This example is mistakenly classified as \"{}\".".format(wrong_pred_category_list[i]) for i in range(len(wrong_pred_category_list))]

            messages = [
                {
                    "role": "user",
                    "content": '''I'm trying to write a news topic classifer prompt. My current prompt is: "[LAST_PROMPT]"
But this prompt gets the following examples wrong:
WRONGDATA
Please analyze the reasons for this prompt obtaining incorrect answers and indicate effective strategies for designing prompts for the current task.'''
                }
            ]
            messages[0]['content'] = messages[0]['content'].replace("WRONGDATA", "\n".join(wrong_batch)).replace('[LAST_PROMPT]', last_prompt_template)
            conversion['messages'] = messages
            response = requests.post(url, json=conversion)
            result = response.json()["data"]["result"]

            system_info = {
                "role": "system",
                "content": "SYSTEMRESPONSE"
            }
            user_info = {
                "role": "user",
                "content": '''Based on the above information, please generation 10 new prompts  (MUST contain "[INPUT]" and "<mask>") for this classification task. (Note that the prompt itself in the news classification task should not contain category information, such as Business, Sports, etc.)'''
            }
            messages = messages + [system_info, user_info]
            messages[1]['content'] = messages[1]['content'].replace("SYSTEMRESPONSE", result)
            
            conversion['messages'] = messages
            response = requests.post(url, json=conversion)
            result = response.json()["data"]["result"]

            new_prompt_list = result.split('\n')
            new_prompt_list = [re.sub(r'^\d+\.\s*', '', x).strip() for x in new_prompt_list]
            new_prompt_list = [x for x in new_prompt_list if '[INPUT]' in x and '<mask>' in x]
            return new_prompt_list[0]
        except Exception as e:
            print('Use old prompt template!')
            print(f'exception: {e}')
            return self.template_list_residual[0]




class RandomTemplateManager(SentenceTemplate):
    def __init__(self, init_template_path: str, output_token = '[MASK]', tokenizer: PreTrainedTokenizer = None, 
                prompt_length = 10,):
        super().__init__([], no_init = True)
        self.tokenizer = tokenizer
        word2idx = self.tokenizer.get_vocab()
        idx2word = {v:k for k,v in word2idx.items()}
        self.vocab_list = [idx2word[idx] for idx in range(len(word2idx))]

        self.init_template_path = init_template_path
        self.prompt_length = prompt_length
        self.output_token = output_token

        self.template_content, self.input_positions, self.output_position = self.parse_template_file(self.init_template_path)

    def parse_template_file(self, path):
        with open(path, 'r', encoding = 'utf-8') as f:
            content = f.read().strip()
            template_info = json.loads(content)
        template_content = []
        input_positions = []
        output_position = -1
        self.template_name = template_info['name']

        for i, desc_dict in enumerate(template_info['template']):
            meta = desc_dict['meta']
            if 'text' in meta:
                input_positions.append(i)
                template_content.append("{" + meta + "}")
            elif "output_token" in meta:
                output_position = i
                template_content.append("[P]")
            elif "prompt_segment" in meta:
                if "rand" in meta:
                    rand_length = self.prompt_length
                    rand_token_list = []
                    while True:
                        if len(rand_token_list) >= rand_length:
                            break
                        rand_token_id = np.random.choice(len(self.vocab_list))
                        token = self.vocab_list[rand_token_id]
                        if not token.startswith("Ġ"):
                            continue
                        rand_token_list.append(token)
                    template_segment = self.tokenizer.convert_tokens_to_string(rand_token_list)   
                    template_segment += '. '
                    template_content.append(template_segment)
                else:
                    segment = desc_dict['content']
                    template_content.append(segment)
            else:
                raise NotImplementedError
        return template_content, input_positions, output_position
    