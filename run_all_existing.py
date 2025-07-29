# -*- coding: utf-8 -*-
# +
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(16)
# -

from os import environ
import argparse
import subprocess
import random
import numpy as np
import seaborn as sns
import os

# Settings
num_runs = 5
datasets = ["German", "Credit", "NBA", "syn-1", "syn-2", "sport", "occupation"]
failed_runs = []

for i in range(num_runs):    
    
    for dataset in datasets:
            
        print("\t", "dataset: ", dataset)
        print("\t", "teacher")
        script_to_run = 'train_teacher.py'
        script_arguments = ['--exp_setting', 'tran', '--teacher', 'GCN', '--dataset', dataset, '--seed', str(i)]
        try:
            result = subprocess.run(['python', script_to_run] + script_arguments, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print("ERROR", e.stdout)

            continue # skip student if teacher fails

        print("student")
        script_to_run = 'train_student.py'
        script_arguments = ['--exp_setting', 'tran', '--teacher', 'GCN', '--student', 'MLP', '--dataset', dataset, '--out_t_path', 'outputs', '--seed', str(i)]
        try:
            result = subprocess.run(['python', script_to_run] + script_arguments, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print("ERROR", e.stdout)

