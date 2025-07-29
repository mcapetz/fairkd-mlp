# Fair Knowledge Distillation from GNNs to MLPs

This repository contains the code and data for the paper.

## Code
- `data_preprocess.py`: data preprocessing, mainly relevant for data in previous experiments
- `dataloader.py`: data loading for existing fairness datasets and code for creating custom synthetic datasets (stochastic block model)
- `models.py`: model specifications for GNN teacher and MLP student
- `plot.py`, `plot_existing.py`: plotting code for custom synthetic SBM datasets and existing (real-world and synthetic) fairness datasets
- `run_all.py`, `run_all_existing.py`: main runner scripts that run experiments for both custom and existing datasets
- `train.conf.yaml`, `train_and_eval.py`, `train_student.py`, `train_teacher.py`, `utils.py`: scripts related to training teacher and student
