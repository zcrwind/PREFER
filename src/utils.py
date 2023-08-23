import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
ROOT_DIR = parentdir
MODEL_CACHE_DIR = os.path.join(ROOT_DIR, 'model_cache/')
FEWSHOT_PATH = os.path.join(ROOT_DIR, 'fewshot_id/')
BATCH_SIZE = 12


import logging
import time
import os
from typing import List

def create_logger(logger_name = "log", root_path = './logs/', filename = ''):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    curr_time =  time.localtime(time.time())
    log_dir = time.strftime('%Y-%m-%d', curr_time)
    time_dir = time.strftime('%H-%M-%S', curr_time)

    if filename != '':
        save_dir = root_path + log_dir + '/' + filename + '_' + time_dir
    else:
        save_dir = root_path + log_dir + "/" + time_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_file_dir = save_dir + "/" + "%slog.txt"%(filename)
    fh = logging.FileHandler(log_file_dir, mode='w')
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger, save_dir

def write_performance(file_addr: str, data_dict, required_keys: List[str]):
    assert file_addr.endswith('.csv')
    if not os.path.exists(file_addr):        
        f = open(file_addr, 'w', encoding = 'utf-8')
        to_write = ','.join(required_keys)
        f.write(to_write + '\n')    
    else:
        f = open(file_addr, 'a', encoding = 'utf-8')
    items = [data_dict[x] for x in required_keys]
    items = [str(x) if type(x) != str else x for x in items]
    to_write = ','.join(items)
    f.write(to_write + '\n')
    f.close()