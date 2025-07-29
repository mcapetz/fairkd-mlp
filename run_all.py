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
dataset = "sbm"
failed_runs = []

for i in range(num_runs):    
    for c in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print("c: ", c)
    
        for p in [0.1, 0.2]:
            q = 0.5
            # Construct the filename
            filename = f"trad_metrics_student_{i}_p={p}_q={q}_c={c}.npy"
            file_path = os.path.join("saved_arrays", filename)
            
            # check if it is already done
            if os.path.exists(file_path): 
                print("\t", dataset, class_weights, file_path, "already done")
                continue
            else:
                print("not done")
            
            print("\t", "dataset: ", dataset)
            print("\t", "teacher")
            script_to_run = 'train_teacher.py'
            script_arguments = ['--exp_setting', 'tran', '--teacher', 'GCN', '--dataset', dataset, '--p', str(p), '--q', str(q), '--c', str(c), '--seed', str(i)]
            try:
                result = subprocess.run(['python', script_to_run] + script_arguments, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                print("\t", "\t", dataset, class_weights,"error")
                num_failures = e.stdout.count("✗")
                entry_line = f"{dataset}_{class_weights}_num_failures:{num_failures}\n"
                
                # Check if the line already exists
                already_logged = False
                if os.path.exists("error_log_1.txt"):
                    with open("error_log_1.txt", "r") as f:
                        for line in f:
                            if line.startswith(f"{dataset}_{class_weights}_num_failures:"):
                                already_logged = True
                                break

                # Only write if not already logged
                if not already_logged:
                    with open("error_log_1.txt", "a") as f:
                        f.write(entry_line)
                        f.write(f"{e.stdout}\n{e.stderr}")
                        
                continue # skip student if teacher fails
            
            print("student")
            script_to_run = 'train_student.py'
            script_arguments = ['--exp_setting', 'tran', '--teacher', 'GCN', '--student', 'MLP', '--dataset', dataset, '--p', str(p), '--q', str(q), '--c', str(c), '--out_t_path', 'outputs', '--seed', str(i)]
            try:
                result = subprocess.run(['python', script_to_run] + script_arguments, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                print("\t", "\t", dataset, class_weights,"error")
                num_failures = e.stdout.count("✗")
                entry_line = f"{dataset}_{class_weights}_num_failures:{num_failures}\n"
                
                # Check if the line already exists
                already_logged = False
                if os.path.exists("error_log_1.txt"):
                    with open("error_log_1.txt", "r") as f:
                        for line in f:
                            if line.startswith(f"{dataset}_{class_weights}_num_failures:"):
                                already_logged = True
                                break

                # Only write if not already logged
                if not already_logged:
                    with open("error_log_1.txt", "a") as f:
                        f.write(entry_line)
                        f.write(f"{e.stdout}\n{e.stderr}")

