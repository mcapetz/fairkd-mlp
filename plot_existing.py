# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy import stats as scipy_stats
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import pandas as pd

seeds = ["0", "1", "2", "3", "4"]

figs_folder = "figs_final"
saved_arrays_folder = "saved_arrays"
datasets = ["German", "Credit", "NBA", "syn-1", "syn-2", "sport", "occupation"]

# +
auc_dict = dict()

for dataset in datasets:

    auc_dict[dataset] = dict()

    teacher_auc_diffs = []
    student_auc_diffs = []
    auc_diffs = []

    teacher_acc_diffs = []
    student_acc_diffs = []
    acc_diffs = []

    teacher_dp_diffs = []
    student_dp_diffs = []
    dp_diffs = []

    teacher_eo_diffs = []
    student_eo_diffs = []
    eo_diffs = []

    # avg and std for seeds
    for seed in seeds: 
        teacher_auc_diff = abs(np.load(f'{saved_arrays_folder}/auc_ovr_diff_teacher_{seed}_{dataset}.npy'))
        student_auc_diff = abs(np.load(f'{saved_arrays_folder}/auc_ovr_diff_student_{seed}_{dataset}.npy'))
        auc_diff = teacher_auc_diff - student_auc_diff

        dp_0, dp_1, eo_0, eo_1, dp, eo, teacher_acc_diff = np.load(f'{saved_arrays_folder}/trad_metrics_teacher_{seed}_{dataset}.npy')
        dp_0_, dp_1_, eo_0_, eo_1_, dp_, eo_, student_acc_diff = np.load(f'{saved_arrays_folder}/trad_metrics_student_{seed}_{dataset}.npy')
        
        teacher_acc_diff = abs(teacher_acc_diff)
        student_acc_diff = abs(student_acc_diff)
        acc_diff = teacher_acc_diff - student_acc_diff

        teacher_dp_diff = abs(dp_0 - dp_1)
        student_dp_diff = abs(dp_0_ - dp_1_)
        dp_diff = teacher_dp_diff - student_dp_diff

        teacher_eo_diff = abs(eo_0 - eo_1)
        student_eo_diff = abs(eo_0_ - eo_1_)
        eo_diff = teacher_eo_diff - student_eo_diff

        teacher_auc_diffs.append(teacher_auc_diff)
        student_auc_diffs.append(student_auc_diff)
        auc_diffs.append(auc_diff)

        teacher_acc_diffs.append(teacher_acc_diff)
        student_acc_diffs.append(student_acc_diff)
        acc_diffs.append(acc_diff)

        teacher_dp_diffs.append(teacher_dp_diff)
        student_dp_diffs.append(student_dp_diff)
        dp_diffs.append(dp_diff)

        teacher_eo_diffs.append(teacher_eo_diff)
        student_eo_diffs.append(student_eo_diff)
        eo_diffs.append(eo_diff)

    auc_dict[dataset] = {
        "avg_teacher_auc_diffs": np.mean(teacher_auc_diffs, axis=0),
        "std_teacher_auc_diffs": np.std(teacher_auc_diffs, axis=0),
        "avg_student_auc_diffs": np.mean(student_auc_diffs, axis=0),
        "std_student_auc_diffs": np.std(student_auc_diffs, axis=0),
        "avg_auc_diff": np.mean(auc_diffs, axis=0),
        "std_auc_diff": np.std(auc_diffs, axis=0),

        "avg_teacher_acc_diffs": np.mean(teacher_acc_diffs, axis=0),
        "std_teacher_acc_diffs": np.std(teacher_acc_diffs, axis=0),
        "avg_student_acc_diffs": np.mean(student_acc_diffs, axis=0),
        "std_student_acc_diffs": np.std(student_acc_diffs, axis=0),
        "avg_acc_diff": np.mean(acc_diffs, axis=0),
        "std_acc_diff": np.std(acc_diffs, axis=0),

        "avg_teacher_dp_diffs": np.mean(teacher_dp_diffs, axis=0),
        "std_teacher_dp_diffs": np.std(teacher_dp_diffs, axis=0),
        "avg_student_dp_diffs": np.mean(student_dp_diffs, axis=0),
        "std_student_dp_diffs": np.std(student_dp_diffs, axis=0),
        "avg_dp_diff": np.mean(dp_diffs, axis=0),
        "std_dp_diff": np.std(dp_diffs, axis=0),

        "avg_teacher_eo_diffs": np.mean(teacher_eo_diffs, axis=0),
        "std_teacher_eo_diffs": np.std(teacher_eo_diffs, axis=0),
        "avg_student_eo_diffs": np.mean(student_eo_diffs, axis=0),
        "std_student_eo_diffs": np.std(student_eo_diffs, axis=0),
        "avg_eo_diff": np.mean(eo_diffs, axis=0),
        "std_eo_diff": np.std(eo_diffs, axis=0),
    }

# -

def plot_heatmap_comparison_acc_auc(auc_dict, datasets):
    # Create data matrix
    metrics = ['AUC Diff', 'Acc Diff', 'Teacher AUC', 'Student AUC', 'Teacher Acc', 'Student Acc']
    data_matrix = []
    
    for dataset in datasets:
        row = [
            auc_dict[dataset]['avg_auc_diff'],
            auc_dict[dataset]['avg_acc_diff'],
            auc_dict[dataset]['avg_teacher_auc_diffs'],
            auc_dict[dataset]['avg_student_auc_diffs'],
            auc_dict[dataset]['avg_teacher_acc_diffs'],
            auc_dict[dataset]['avg_student_acc_diffs']
        ]
        data_matrix.append(row)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(data_matrix, 
                xticklabels=metrics,
                yticklabels=datasets,
                annot=True, 
                fmt='.3f',
                cmap='viridis',
                center=0,
                cbar_kws={'label': 'Difference Value'})
    plt.title('Model Performance Heatmap')
    plt.xlabel('Metrics')
    plt.ylabel('Datasets')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{figs_folder}/heatmap_acc_auc.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_heatmap_comparison_dp_eo(auc_dict, datasets):
    # Create data matrix with all metrics
    metrics = ['DP Diff', 'EO Diff', 'Teacher DP', 'Student DP', 'Teacher EO', 'Student EO']
    data_matrix = []
    
    for dataset in datasets:
        row = [
            auc_dict[dataset]['avg_dp_diff'],
            auc_dict[dataset]['avg_eo_diff'],
            auc_dict[dataset]['avg_teacher_dp_diffs'],
            auc_dict[dataset]['avg_student_dp_diffs'],
            auc_dict[dataset]['avg_teacher_eo_diffs'],
            auc_dict[dataset]['avg_student_eo_diffs']
        ]
        data_matrix.append(row)
    
    # Create enhanced heatmap
    plt.figure(figsize=(16, 10))
    sns.heatmap(data_matrix, 
                xticklabels=metrics,
                yticklabels=datasets,
                annot=True, 
                fmt='.3f',
                cmap='viridis',
                center=0,
                cbar_kws={'label': 'Difference Value'})
    plt.title('Traditional Fairness Metrics Heatmap')
    plt.xlabel('Metrics')
    plt.ylabel('Datasets')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{figs_folder}/heatmap_dp_eo.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_heatmap_comparison_metrics(auc_dict, datasets):
    # Create data matrix with all metrics
    metrics = ['DP Diff', 'EO Diff', 'AUC Diff', 'Acc Diff']
    data_matrix = []
    
    for dataset in datasets:
        row = [
            auc_dict[dataset]['avg_dp_diff'],
            auc_dict[dataset]['avg_eo_diff'],
            auc_dict[dataset]['avg_auc_diff'],
            auc_dict[dataset]['avg_acc_diff']
        ]
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    # Create enhanced heatmap
    plt.figure(figsize=(16, 10))
    ax = sns.heatmap(data_matrix, 
                     xticklabels=metrics,
                     yticklabels=datasets,
                     annot=True, 
                     fmt='.3f',
                     cmap='viridis',
                     center=0,
                     cbar_kws={'label': 'Difference Value'})

    # Add vertical line to separate fairness (DP/EO) from performance (AUC/Acc)
    ax.axvline(x=2, color='white', linewidth=3)  # x=2 is between column 1 and 2 (0-indexed)

    # Optionally enhance tick appearance
    plt.title('Metrics Comparison Heatmap')
    plt.xlabel('Metrics')
    plt.ylabel('Datasets')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{figs_folder}/heatmap_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_heatmap_comparison_metrics_abs(auc_dict, datasets):
    # Create data matrix with all metrics
    metrics = ['DP Diff', 'EO Diff', 'AUC Diff', 'Acc Diff']
    data_matrix = []
    
    for dataset in datasets:
        row = [
            abs(auc_dict[dataset]['avg_dp_diff']),
            abs(auc_dict[dataset]['avg_eo_diff']),
            abs(auc_dict[dataset]['avg_auc_diff']),
            abs(auc_dict[dataset]['avg_acc_diff'])
        ]
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    # Create enhanced heatmap
    plt.figure(figsize=(16, 10))
    ax = sns.heatmap(data_matrix, 
                     xticklabels=metrics,
                     yticklabels=datasets,
                     annot=True, 
                     fmt='.3f',
                     cmap='viridis',
                     center=0,
                     cbar_kws={'label': 'Difference Value'})

    # Add vertical line to separate fairness (DP/EO) from performance (AUC/Acc)
    ax.axvline(x=2, color='white', linewidth=3)  # x=2 is between column 1 and 2 (0-indexed)

    # Optionally enhance tick appearance
    plt.title('Metrics Comparison Heatmap')
    plt.xlabel('Metrics')
    plt.ylabel('Datasets')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{figs_folder}/heatmap_metrics_abs.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


plot_heatmap_comparison_acc_auc(auc_dict, datasets)
plot_heatmap_comparison_dp_eo(auc_dict, datasets)
plot_heatmap_comparison_metrics(auc_dict, datasets)
plot_heatmap_comparison_metrics_abs(auc_dict, datasets)
