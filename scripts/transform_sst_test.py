import numpy as np
import pandas as pd
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from src.utils import ROOT_DIR


new_lines = []
with open(ROOT_DIR + "datasets/original/SST-2/test.tsv", 'r', encoding = 'utf-8') as f:
    for line in f:
        label = line[0]
        sentence = line[2:].strip()
        new_lines.append(sentence + '\t' + label + '\n')

with open(ROOT_DIR + "datasets/full_dataset/SST-2/test.tsv", 'w', encoding = 'utf-8') as f:
    f.write('sentence\tlabel\n')
    for line in new_lines:
        f.write(line)


