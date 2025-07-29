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

seeds = ["0", "1", "2", "3", "4"]

auc_dict_pxc = dict()
auc_dict_qxc = dict()
auc_dict_qxp = dict()

figs_folder = "figs_final"
saved_arrays_folder = "saved_arrays"

# load in the auc values
for p in [0.1,0.2,0.3,0.4,0.5]:  
    q = 0.5
    
    auc_dict_pxc[p] = dict()
    
    teacher_auc_diffs = []
    student_auc_diffs = []
    auc_diffs = []
    
    teacher_acc_diffs = []
    student_acc_diffs = []
    acc_diffs = []
    
    for c in [0.1,0.2,0.3,0.4,0.5]:

        # avg and std for seeds
        for seed in seeds: 
            teacher_auc_diff = abs(np.load(f'{saved_arrays_folder}/auc_ovr_diff_teacher_{seed}_p={p}_q={q}_c={c}.npy'))
            student_auc_diff = abs(np.load(f'{saved_arrays_folder}/auc_ovr_diff_student_{seed}_p={p}_q={q}_c={c}.npy'))
            auc_diff = teacher_auc_diff - student_auc_diff
            
            dp_0, dp_1, eo_0, eo_1, dp, eo, teacher_acc_diff = np.load(f'{saved_arrays_folder}/trad_metrics_teacher_{seed}_p={p}_q={q}_c={c}.npy')
            dp_0_, dp_1_, eo_0_, eo_1_, dp_, eo_, student_acc_diff = np.load(f'{saved_arrays_folder}/trad_metrics_student_{seed}_p={p}_q={q}_c={c}.npy')
            teacher_acc_diff = abs(teacher_acc_diff)
            student_acc_diff = abs(student_acc_diff)
            acc_diff = teacher_acc_diff - student_acc_diff

            teacher_auc_diffs.append(teacher_auc_diff)
            student_auc_diffs.append(student_auc_diff)
            auc_diffs.append(auc_diff)
            
            teacher_acc_diffs.append(teacher_acc_diff)
            student_acc_diffs.append(student_acc_diff)
            acc_diffs.append(acc_diff)
            
        
        auc_dict_pxc[p][c] = {
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
            
        }

# load in the auc values
for q in [0.05,0.25,0.50,0.75,1.00]:  
    p = 0.3
    
    auc_dict_qxc[q] = dict()
    
    teacher_auc_diffs = []
    student_auc_diffs = []
    auc_diffs = []
    
    teacher_acc_diffs = []
    student_acc_diffs = []
    acc_diffs = []
    
    for c in [0.1,0.2,0.3,0.4,0.5]:

        # avg and std for seeds
        for seed in seeds: 
            teacher_auc_diff = abs(np.load(f'{saved_arrays_folder}/auc_ovr_diff_teacher_{seed}_p={p}_q={q}_c={c}.npy'))
            student_auc_diff = abs(np.load(f'{saved_arrays_folder}/auc_ovr_diff_student_{seed}_p={p}_q={q}_c={c}.npy'))
            auc_diff = teacher_auc_diff - student_auc_diff
            
            dp_0, dp_1, eo_0, eo_1, dp, eo, teacher_acc_diff = np.load(f'{saved_arrays_folder}/trad_metrics_teacher_{seed}_p={p}_q={q}_c={c}.npy')
            dp_0_, dp_1_, eo_0_, eo_1_, dp_, eo_, student_acc_diff = np.load(f'{saved_arrays_folder}/trad_metrics_student_{seed}_p={p}_q={q}_c={c}.npy')
            teacher_acc_diff = abs(teacher_acc_diff)
            student_acc_diff = abs(student_acc_diff)
            acc_diff = teacher_acc_diff - student_acc_diff

            teacher_auc_diffs.append(teacher_auc_diff)
            student_auc_diffs.append(student_auc_diff)
            auc_diffs.append(auc_diff)
            
            teacher_acc_diffs.append(teacher_acc_diff)
            student_acc_diffs.append(student_acc_diff)
            acc_diffs.append(acc_diff)
            
        
        auc_dict_qxc[q][c] = {
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
            
        }

# load in the auc values
for q in [0.05,0.25,0.50,0.75,1.00]:  
    c = 0.3
    
    auc_dict_qxp[q] = dict()
    
    teacher_auc_diffs = []
    student_auc_diffs = []
    auc_diffs = []
    
    teacher_acc_diffs = []
    student_acc_diffs = []
    acc_diffs = []
    
    for p in [0.1,0.2,0.3,0.4,0.5]:

        # avg and std for seeds
        for seed in seeds: 
            teacher_auc_diff = abs(np.load(f'{saved_arrays_folder}/auc_ovr_diff_teacher_{seed}_p={p}_q={q}_c={c}.npy'))
            student_auc_diff = abs(np.load(f'{saved_arrays_folder}/auc_ovr_diff_student_{seed}_p={p}_q={q}_c={c}.npy'))
            auc_diff = teacher_auc_diff - student_auc_diff
            
            dp_0, dp_1, eo_0, eo_1, dp, eo, teacher_acc_diff = np.load(f'{saved_arrays_folder}/trad_metrics_teacher_{seed}_p={p}_q={q}_c={c}.npy')
            dp_0_, dp_1_, eo_0_, eo_1_, dp_, eo_, student_acc_diff = np.load(f'{saved_arrays_folder}/trad_metrics_student_{seed}_p={p}_q={q}_c={c}.npy')
            teacher_acc_diff = abs(teacher_acc_diff)
            student_acc_diff = abs(student_acc_diff)
            acc_diff = teacher_acc_diff - student_acc_diff

            teacher_auc_diffs.append(teacher_auc_diff)
            student_auc_diffs.append(student_auc_diff)
            auc_diffs.append(auc_diff)
            
            teacher_acc_diffs.append(teacher_acc_diff)
            student_acc_diffs.append(student_acc_diff)
            acc_diffs.append(acc_diff)
            
        
        auc_dict_qxp[q][p] = {
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
            
        }


def create_combined_heatmap_grid(auc_dict_pxc, auc_dict_qxc, auc_dict_qxp, 
                               metric_type='acc', save_path=None):
    """
    Create a single 3x3 grid showing all parameter combinations.
    
    Parameters:
    - auc_dict_pxc: Dictionary for p vs c (q fixed at 0.5 the median)
    - auc_dict_qxc: Dictionary for q vs c (p fixed at 0.3 the median)  
    - auc_dict_qxp: Dictionary for q vs p (c fixed at 0.3 the median)
    - metric_type: 'acc' or 'auc'
    - save_path: Path to save the figure
    """

    # Define parameter values
    p_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    q_values = [0.05, 0.25, 0.50, 0.75, 1.00]
    c_values = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Increase figure size and spacing
    fig = plt.figure(figsize=(18, 14), constrained_layout=False)
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.35, hspace=0.45)

    # Define row configurations: (dict_name, x_values, y_values, x_label, y_label, title_suffix)
    row_configs = [
        (auc_dict_pxc, p_values, c_values, 'Group Balance (p)', 'Class Balance (c)', '(q=0.5)'),
        (auc_dict_qxc, q_values, c_values, 'Edge Ratio (q)', 'Class Balance (c)', '(p=0.3)'),
        (auc_dict_qxp, q_values, p_values, 'Edge Ratio (q)', 'Group Balance (p)', '(c=0.3)')
    ]
    
    col_metrics = ['teacher', 'student', 'diff']
    col_titles = ['Teacher Fairness', 'Student Fairness', 'Teacher - Student Fairness']

    # Precompute common vmin/vmax for all teacher and student metrics
    teacher_vals = []
    student_vals = []
    teacher_student_vals = []

    # Collect all values for consistent scaling
    for data_dict, x_vals, y_vals, _, _, _ in row_configs:
        for x_val in x_vals:
            for y_val in y_vals:
                try:
                    teacher_vals.append(data_dict[x_val][y_val][f"avg_teacher_{metric_type}_diffs"])
                    student_vals.append(data_dict[x_val][y_val][f"avg_student_{metric_type}_diffs"])
                    diff = data_dict[x_val][y_val][f"avg_teacher_{metric_type}_diffs"] - data_dict[x_val][y_val][f"avg_student_{metric_type}_diffs"]
                    teacher_student_vals.append(diff)
                except KeyError:
                    continue

    shared_vmin = min(teacher_vals + student_vals) if teacher_vals and student_vals else 0
    shared_vmax = max(teacher_vals + student_vals) if teacher_vals and student_vals else 1

    teacher_student_vmin = min(teacher_student_vals) if teacher_student_vals else -0.1
    teacher_student_vmax = max(teacher_student_vals) if teacher_student_vals else 0.1

    # Create the 3x3 grid of heatmaps
    for row_idx, (data_dict, x_values, y_values, x_label, y_label, title_suffix) in enumerate(row_configs):
        for col_idx, metric in enumerate(col_metrics):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            # Create data matrix
            data_matrix = np.full((len(y_values), len(x_values)), np.nan)
            
            for i, y_val in enumerate(y_values):
                for j, x_val in enumerate(x_values):
                    try:
                        if metric == 'teacher':
                            data_matrix[i, j] = data_dict[x_val][y_val][f"avg_teacher_{metric_type}_diffs"]
                        elif metric == 'student':
                            data_matrix[i, j] = data_dict[x_val][y_val][f"avg_student_{metric_type}_diffs"]
                        else:  # diff
                            teacher_val = data_dict[x_val][y_val][f"avg_teacher_{metric_type}_diffs"]
                            student_val = data_dict[x_val][y_val][f"avg_student_{metric_type}_diffs"]
                            data_matrix[i, j] = teacher_val - student_val
                    except KeyError as e:
                        print(f"KeyError at row {row_idx}, col {col_idx}: {e}")

            # Set vmin and vmax based on metric
            if metric in ['teacher', 'student']:
                vmin = shared_vmin
                vmax = shared_vmax
                cmap = 'viridis'
                norm = None
                viridis_ticks = np.round(np.linspace(vmin, vmax, 7), 3)
                cbar_kws = {'label': 'Difference', 'ticks': viridis_ticks}
            else:  # diff heatmap
                abs_max = max(abs(teacher_student_vmin), abs(teacher_student_vmax))
                vmin = -abs_max
                vmax = abs_max
                cmap = 'RdBu'
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                ticks = np.linspace(vmin, vmax, 7)
                cbar_kws = {'label': 'Difference', 'ticks': ticks}
                
            # Reverse y-axis to make it ascending from bottom to top
            reversed_data_matrix = np.flipud(data_matrix)
            reversed_y_labels = [f"{y:.2f}" if y < 1 else f"{y:.1f}" for y in reversed(y_values)]

            sns.heatmap(
                reversed_data_matrix, annot=True, cmap=cmap, norm=norm, ax=ax, fmt='.3f',
                xticklabels=[f"{x:.2f}" if x < 1 else f"{x:.1f}" for x in x_values], 
                yticklabels=reversed_y_labels,
                cbar_kws=cbar_kws,
                vmin=vmin, vmax=vmax,
                annot_kws={'fontsize': 8}
            )
            
            ax.set_title(f"{col_titles[col_idx]}", fontsize=13, pad=20)
            
            # Add dispersion annotation (standard deviation of the matrix)
            dispersion = np.nanstd(data_matrix)
            ax.text(1.02, 1.05, f"stdev: {dispersion:.3f}", 
                horizontalalignment='left',
                verticalalignment='bottom', 
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(facecolor='lightblue', alpha=0.7, edgecolor='lightgray', boxstyle='round,pad=0.3'))

            # Set labels and title
            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)

            # Set title - add suffix only for first column to avoid repetition
            if col_idx == 0:
                full_title = f"{col_titles[col_idx]} {title_suffix}"
            else:
                full_title = col_titles[col_idx]
            ax.set_title(full_title, fontsize=11, pad=20)

    # Set main title
    metric_name = 'Accuracy' if metric_type == 'acc' else 'AUC'
    plt.suptitle(f'{metric_name} Fairness Differences (0-1)', fontsize=16, y=0.96)

    # Use tight_layout with padding
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig


# +
fig_combined_acc = create_combined_heatmap_grid(auc_dict_pxc, auc_dict_qxc, auc_dict_qxp, 
                                           metric_type='acc', 
                                           save_path=f'{figs_folder}/combined_heatmap_acc.png')

fig_combined_auc = create_combined_heatmap_grid(auc_dict_pxc, auc_dict_qxc, auc_dict_qxp, 
                                           metric_type='auc', 
                                           save_path=f'{figs_folder}/combined_heatmap_auc.png')


# -

def plot_fairness_subplots_updated(auc_dict_pxc, auc_dict_qxc, auc_dict_qxp, 
                                   metric_type='acc', save_path=None, sharey=True):
    """
    Create line plots showing fairness differences across different parameters.
    
    Parameters:
    - auc_dict_pxc: Dictionary for p vs c (q fixed at 0.5)
    - auc_dict_qxc: Dictionary for q vs c (p fixed at 0.3)  
    - auc_dict_qxp: Dictionary for q vs p (c fixed at 0.3)
    - metric_type: 'acc' or 'auc'
    - save_path: Path to save the figure
    - sharey: Whether to share y-axis across subplots
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=sharey)
    
    # Define parameter values
    p_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    q_values = [0.05, 0.25, 0.50, 0.75, 1.00]
    c_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Colors
    teacher_color = "#332288"  # Dark purple
    student_color = "#EE7733"  # Vermilion
    
    # 1. Class balance plot (averaging across p and q)
    ax = axes[0]
    teacher_means, teacher_stds = [], []
    student_means, student_stds = [], []
    
    for c_val in c_values:
        teacher_vals, student_vals = [], []
        
        # Collect from p vs c dict (q=0.5)
        for p_val in p_values:
            try:
                teacher_vals.append(auc_dict_pxc[p_val][c_val][f"avg_teacher_{metric_type}_diffs"])
                student_vals.append(auc_dict_pxc[p_val][c_val][f"avg_student_{metric_type}_diffs"])
            except KeyError:
                continue
        
        # Collect from q vs c dict (p=0.3)
        for q_val in q_values:
            try:
                teacher_vals.append(auc_dict_qxc[q_val][c_val][f"avg_teacher_{metric_type}_diffs"])
                student_vals.append(auc_dict_qxc[q_val][c_val][f"avg_student_{metric_type}_diffs"])
            except KeyError:
                continue
        
        # Calculate statistics
        if teacher_vals and student_vals:
            teacher_means.append(np.mean(teacher_vals))
            teacher_stds.append(np.std(teacher_vals))
            student_means.append(np.mean(student_vals))
            student_stds.append(np.std(student_vals))
        else:
            teacher_means.append(np.nan)
            teacher_stds.append(0)
            student_means.append(np.nan)
            student_stds.append(0)
    
    # Plot class balance
    x_pos = np.arange(len(c_values))
    valid = np.isfinite(teacher_means) & np.isfinite(student_means)
    
    ax.plot(x_pos[valid], np.array(teacher_means)[valid], '-o', color=teacher_color, label='Teacher')
    ax.fill_between(x_pos[valid], 
                    np.array(teacher_means)[valid] - np.array(teacher_stds)[valid],
                    np.array(teacher_means)[valid] + np.array(teacher_stds)[valid],
                    color=teacher_color, alpha=0.2)
    
    ax.plot(x_pos[valid], np.array(student_means)[valid], '--s', color=student_color, label='Student')
    ax.fill_between(x_pos[valid],
                    np.array(student_means)[valid] - np.array(student_stds)[valid],
                    np.array(student_means)[valid] + np.array(student_stds)[valid],
                    color=student_color, alpha=0.2)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{c:.1f}" for c in c_values])
    ax.set_xlabel("Class Balance (c)")
    ax.set_title(f"{'Acc' if metric_type == 'acc' else 'AUC'} vs Class Balance (c)")
#     ax.grid(True)
    ax.set_ylabel(f"{'Acc' if metric_type == 'acc' else 'AUC'} Difference")
    ax.legend()
    
    # 2. Group balance plot (p) - average across c values
    ax = axes[1]
    teacher_means, teacher_stds = [], []
    student_means, student_stds = [], []
    
    for p_val in p_values:
        teacher_vals, student_vals = [], []
        
        # Collect from p vs c dict
        for c_val in c_values:
            try:
                teacher_vals.append(auc_dict_pxc[p_val][c_val][f"avg_teacher_{metric_type}_diffs"])
                student_vals.append(auc_dict_pxc[p_val][c_val][f"avg_student_{metric_type}_diffs"])
            except KeyError:
                continue
        
        # Collect from q vs p dict (average across q values for this p)
        for q_val in q_values:
            try:
                teacher_vals.append(auc_dict_qxp[q_val][p_val][f"avg_teacher_{metric_type}_diffs"])
                student_vals.append(auc_dict_qxp[q_val][p_val][f"avg_student_{metric_type}_diffs"])
            except KeyError:
                continue
        
        # Calculate statistics
        if teacher_vals and student_vals:
            teacher_means.append(np.mean(teacher_vals))
            teacher_stds.append(np.std(teacher_vals))
            student_means.append(np.mean(student_vals))
            student_stds.append(np.std(student_vals))
        else:
            teacher_means.append(np.nan)
            teacher_stds.append(0)
            student_means.append(np.nan)
            student_stds.append(0)
    
    # Plot group balance
    x_pos = np.arange(len(p_values))
    valid = np.isfinite(teacher_means) & np.isfinite(student_means)
    
    ax.plot(x_pos[valid], np.array(teacher_means)[valid], '-o', color=teacher_color, label='Teacher')
    ax.fill_between(x_pos[valid], 
                    np.array(teacher_means)[valid] - np.array(teacher_stds)[valid],
                    np.array(teacher_means)[valid] + np.array(teacher_stds)[valid],
                    color=teacher_color, alpha=0.2)
    
    ax.plot(x_pos[valid], np.array(student_means)[valid], '--s', color=student_color, label='Student')
    ax.fill_between(x_pos[valid],
                    np.array(student_means)[valid] - np.array(student_stds)[valid],
                    np.array(student_means)[valid] + np.array(student_stds)[valid],
                    color=student_color, alpha=0.2)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{p:.1f}" for p in p_values])
    ax.set_xlabel("Group Balance (p)")
    ax.set_title(f"{'Acc' if metric_type == 'acc' else 'AUC'} vs Group Balance (p)")
    ax.legend()
    
    # 3. Edge probability plot (q) - average across c values
    ax = axes[2]
    teacher_means, teacher_stds = [], []
    student_means, student_stds = [], []
    
    for q_val in q_values:
        teacher_vals, student_vals = [], []
        
        # Collect from q vs c dict
        for c_val in c_values:
            try:
                teacher_vals.append(auc_dict_qxc[q_val][c_val][f"avg_teacher_{metric_type}_diffs"])
                student_vals.append(auc_dict_qxc[q_val][c_val][f"avg_student_{metric_type}_diffs"])
            except KeyError:
                continue
        
        # Collect from q vs p dict (average across p values for this q)
        for p_val in p_values:
            try:
                teacher_vals.append(auc_dict_qxp[q_val][p_val][f"avg_teacher_{metric_type}_diffs"])
                student_vals.append(auc_dict_qxp[q_val][p_val][f"avg_student_{metric_type}_diffs"])
            except KeyError:
                continue
        
        # Calculate statistics
        if teacher_vals and student_vals:
            teacher_means.append(np.mean(teacher_vals))
            teacher_stds.append(np.std(teacher_vals))
            student_means.append(np.mean(student_vals))
            student_stds.append(np.std(student_vals))
        else:
            teacher_means.append(np.nan)
            teacher_stds.append(0)
            student_means.append(np.nan)
            student_stds.append(0)
    
    # Plot edge probability
    x_pos = np.arange(len(q_values))
    valid = np.isfinite(teacher_means) & np.isfinite(student_means)
    
    ax.plot(x_pos[valid], np.array(teacher_means)[valid], '-o', color=teacher_color, label='Teacher')
    ax.fill_between(x_pos[valid], 
                    np.array(teacher_means)[valid] - np.array(teacher_stds)[valid],
                    np.array(teacher_means)[valid] + np.array(teacher_stds)[valid],
                    color=teacher_color, alpha=0.2)
    
    ax.plot(x_pos[valid], np.array(student_means)[valid], '--s', color=student_color, label='Student')
    ax.fill_between(x_pos[valid],
                    np.array(student_means)[valid] - np.array(student_stds)[valid],
                    np.array(student_means)[valid] + np.array(student_stds)[valid],
                    color=student_color, alpha=0.2)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{q:.2f}" for q in q_values])
    ax.set_xlabel("Edge Probability (q)")
    ax.set_title(f"{'Acc' if metric_type == 'acc' else 'AUC'} vs Edge Probability (q)")
    ax.legend()
    
    # Set the main title
    fig.suptitle(f"{'Acc' if metric_type == 'acc' else 'AUC'} One-factor Plots", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


# +
fig_acc = plot_fairness_subplots_updated(auc_dict_pxc, auc_dict_qxc, auc_dict_qxp, 
                                     metric_type='acc', 
                                     save_path=f'{figs_folder}/fairness_lines_acc.png')

fig_auc = plot_fairness_subplots_updated(auc_dict_pxc, auc_dict_qxc, auc_dict_qxp, 
                                     metric_type='auc', 
                                     save_path=f'{figs_folder}/fairness_lines_auc.png')


# +
def plot_teacher_vs_student_scatter(auc_dict_pxc, auc_dict_qxc, auc_dict_qxp, 
                                   metric_type='acc', save_path=None, color_by='p'):
    """
    Create scatter plots showing teacher vs student fairness metrics with color-coding by balance parameters.
    
    Parameters:
    - auc_dict_pxc: Dictionary for p vs c (q fixed at 0.5)
    - auc_dict_qxc: Dictionary for q vs c (p fixed at 0.3)  
    - auc_dict_qxp: Dictionary for q vs p (c fixed at 0.3)
    - metric_type: 'acc' or 'auc'
    - save_path: Path to save the figure
    - color_by: Parameter to color by ('p', 'c', 'q', or 'combined_balance')
    """
    
    # Collect all teacher and student fairness values with parameters
    teacher_values = []
    student_values = []
    parameter_labels = []
    p_values = []
    c_values = []
    q_values = []
    
    # From p vs c dict (q=0.5)
    for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for c in [0.1, 0.2, 0.3, 0.4, 0.5]:
            try:
                teacher_val = auc_dict_pxc[p][c][f"avg_teacher_{metric_type}_diffs"]
                student_val = auc_dict_pxc[p][c][f"avg_student_{metric_type}_diffs"]
                teacher_values.append(teacher_val)
                student_values.append(student_val)
                parameter_labels.append(f"p={p}, c={c}, q=0.5")
                p_values.append(p)
                c_values.append(c)
                q_values.append(0.5)
            except KeyError:
                continue
    
    # From q vs c dict (p=0.3)
    for q in [0.05, 0.25, 0.50, 0.75, 1.00]:
        for c in [0.1, 0.2, 0.3, 0.4, 0.5]:
            try:
                teacher_val = auc_dict_qxc[q][c][f"avg_teacher_{metric_type}_diffs"]
                student_val = auc_dict_qxc[q][c][f"avg_student_{metric_type}_diffs"]
                teacher_values.append(teacher_val)
                student_values.append(student_val)
                parameter_labels.append(f"p=0.3, c={c}, q={q}")
                p_values.append(0.3)
                c_values.append(c)
                q_values.append(q)
            except KeyError:
                continue
    
    # From q vs p dict (c=0.3)
    for q in [0.05, 0.25, 0.50, 0.75, 1.00]:
        for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
            try:
                teacher_val = auc_dict_qxp[q][p][f"avg_teacher_{metric_type}_diffs"]
                student_val = auc_dict_qxp[q][p][f"avg_student_{metric_type}_diffs"]
                teacher_values.append(teacher_val)
                student_values.append(student_val)
                parameter_labels.append(f"p={p}, c=0.3, q={q}")
                p_values.append(p)
                c_values.append(0.3)
                q_values.append(q)
            except KeyError:
                continue
    
    # Convert to numpy arrays
    teacher_values = np.array(teacher_values)
    student_values = np.array(student_values)
    p_values = np.array(p_values)
    c_values = np.array(c_values)
    q_values = np.array(q_values)
    
    # Remove any NaN values
    valid_mask = np.isfinite(teacher_values) & np.isfinite(student_values)
    teacher_values = teacher_values[valid_mask]
    student_values = student_values[valid_mask]
    p_values = p_values[valid_mask]
    c_values = c_values[valid_mask]
    q_values = q_values[valid_mask]
    parameter_labels = [parameter_labels[i] for i in range(len(parameter_labels)) if valid_mask[i]]
    
    # Determine color values based on color_by parameter
    if color_by == 'p':
        color_values = p_values
        color_label = 'Group Balance (p)'
        cmap = 'viridis'
    elif color_by == 'c':
        color_values = c_values
        color_label = 'Class Balance (c)'
        cmap = 'plasma'
    elif color_by == 'q':
        color_values = q_values
        color_label = 'Group Connectivity (q)'
        cmap = 'coolwarm'
    elif color_by == 'combined_balance':
        # Create a combined balance metric
        color_values = (np.abs(p_values - 0.5) + np.abs(c_values - 0.5) + np.abs(q_values - 1.0))/3
        color_label = 'Combined Imbalance'
        cmap = 'RdYlBu_r'
    else:
        color_values = p_values  # Default to p
        color_label = 'Group Balance (p)'
        cmap = 'viridis'
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create scatter plot with color coding
    scatter = ax.scatter(teacher_values, student_values, 
                        c=color_values, cmap=cmap,
                        alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(color_label, fontsize=11)
    
    # Calculate correlation coefficients
    pearson_r, pearson_p = pearsonr(teacher_values, student_values)
    spearman_r, spearman_p = spearmanr(teacher_values, student_values)
    
    # Fit regression line
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(teacher_values, student_values)
    line_x = np.array([teacher_values.min(), teacher_values.max()])
    line_y = slope * line_x + intercept
    
    # Plot regression line
    ax.plot(line_x, line_y, 'r--', linewidth=2, alpha=0.8, label=f'Linear fit (R²={r_value**2:.3f})')
    
    # Add diagonal line (perfect correlation)
    min_val = min(teacher_values.min(), student_values.min())
    max_val = max(teacher_values.max(), student_values.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k:', linewidth=2, alpha=0.5, label='Perfect correlation')
    
    # Labels and title
    metric_name = 'Accuracy' if metric_type == 'acc' else 'AUC'
    ax.set_xlabel(f'Teacher {metric_name} Fairness Difference', fontsize=12)
    ax.set_ylabel(f'Student {metric_name} Fairness Difference', fontsize=12)
    ax.set_title(f'Teacher vs Student {metric_name} Fairness (Colored by {color_label})', fontsize=14, pad=20)
    
    # Add correlation statistics as text box
    stats_text = f"""Correlation Statistics:
Pearson r = {pearson_r:.3f} (p = {pearson_p:.3f})
Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.3f})
R² = {r_value**2:.3f}
n = {len(teacher_values)} points"""

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add interpretation text
    if pearson_r > 0.7:
        interpretation = "Strong positive correlation"
    elif pearson_r > 0.5:
        interpretation = "Moderate positive correlation"  
    elif pearson_r > 0.3:
        interpretation = "Weak positive correlation"
    elif pearson_r > -0.3:
        interpretation = "No clear correlation"
    elif pearson_r > -0.5:
        interpretation = "Weak negative correlation"
    elif pearson_r > -0.7:
        interpretation = "Moderate negative correlation"
    else:
        interpretation = "Strong negative correlation"
        
    print(f"{interpretation} - Colored by {color_label}")
    
    # Legend
    ax.legend(loc='upper right')
    
    # Set aspect ratio for better comparison
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig, (pearson_r, pearson_p, spearman_r, spearman_p, r_value**2)


# Alternative version with discrete color coding for clearer visualization
def plot_teacher_vs_student_scatter_discrete(pxc, qxc, qxp, metric_type='acc', ax=None, save_path=None):
    
    """
    Create scatter plots with discrete color coding for different balance categories.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    # [Same data collection code as above...]
    teacher_values = []
    student_values = []
    parameter_labels = []
    p_values = []
    c_values = []
    q_values = []
    
    # From p vs c dict (q=0.5)
    for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for c in [0.1, 0.2, 0.3, 0.4, 0.5]:
            try:
                teacher_val = auc_dict_pxc[p][c][f"avg_teacher_{metric_type}_diffs"]
                student_val = auc_dict_pxc[p][c][f"avg_student_{metric_type}_diffs"]
                teacher_values.append(teacher_val)
                student_values.append(student_val)
                parameter_labels.append(f"p={p}, c={c}, q=0.5")
                p_values.append(p)
                c_values.append(c)
                q_values.append(0.5)
            except KeyError:
                continue
    
    # From q vs c dict (p=0.3)
    for q in [0.05, 0.25, 0.50, 0.75, 1.00]:
        for c in [0.1, 0.2, 0.3, 0.4, 0.5]:
            try:
                teacher_val = auc_dict_qxc[q][c][f"avg_teacher_{metric_type}_diffs"]
                student_val = auc_dict_qxc[q][c][f"avg_student_{metric_type}_diffs"]
                teacher_values.append(teacher_val)
                student_values.append(student_val)
                parameter_labels.append(f"p=0.3, c={c}, q={q}")
                p_values.append(0.3)
                c_values.append(c)
                q_values.append(q)
            except KeyError:
                continue
    
    # From q vs p dict (c=0.3)
    for q in [0.05, 0.25, 0.50, 0.75, 1.00]:
        for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
            try:
                teacher_val = auc_dict_qxp[q][p][f"avg_teacher_{metric_type}_diffs"]
                student_val = auc_dict_qxp[q][p][f"avg_student_{metric_type}_diffs"]
                teacher_values.append(teacher_val)
                student_values.append(student_val)
                parameter_labels.append(f"p={p}, c=0.3, q={q}")
                p_values.append(p)
                c_values.append(0.3)
                q_values.append(q)
            except KeyError:
                continue
    
    # Convert to numpy arrays and clean
    teacher_values = np.array(teacher_values)
    student_values = np.array(student_values)
    p_values = np.array(p_values)
    c_values = np.array(c_values)
    q_values = np.array(q_values)
    
    valid_mask = np.isfinite(teacher_values) & np.isfinite(student_values)
    teacher_values = teacher_values[valid_mask]
    student_values = student_values[valid_mask]
    p_values = p_values[valid_mask]
    c_values = c_values[valid_mask]
    q_values = q_values[valid_mask]
    
    # Create balance categories
    balance_categories = []
    colors = []
    color_map = {
        'High Balance': '#2E8B57',      # Sea Green
        'Medium Balance': '#FFA500',     # Orange
        'Low Balance': '#DC143C',        # Crimson
        'Very Low Balance': '#8B0000'    # Dark Red
    }
    
    for i in range(len(p_values)):
        p, c, q = p_values[i], c_values[i], q_values[i]
        
        # Calculate balance score (closer to 0.5 is more balanced)
        balance_score = (abs(p - 0.5) + abs(c - 0.5) + abs(q - 1.0)) / 3
        
        if balance_score < 0.15:
            category = 'High Balance'
        elif balance_score < 0.25:
            category = 'Medium Balance'
        elif balance_score < 0.35:
            category = 'Low Balance'
        else:
            category = 'Very Low Balance'
            
        balance_categories.append(category)
        colors.append(color_map[category])
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot points by category
    for category in color_map.keys():
        mask = [cat == category for cat in balance_categories]
        if any(mask):
            ax.scatter(teacher_values[mask], student_values[mask], 
                      c=color_map[category], label=category,
                      alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
    
    # Add regression line and diagonal
    pearson_r, pearson_p = pearsonr(teacher_values, student_values)
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(teacher_values, student_values)
    line_x = np.array([teacher_values.min(), teacher_values.max()])
    line_y = slope * line_x + intercept
    
    ax.plot(line_x, line_y, 'r--', linewidth=2, alpha=0.8, label=f'Linear fit (R²={r_value**2:.3f})')
    
    min_val = min(teacher_values.min(), student_values.min())
    max_val = max(teacher_values.max(), student_values.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k:', linewidth=2, alpha=0.5, label='Perfect correlation')
    
    # Labels and formatting
    metric_name = 'Accuracy' if metric_type == 'acc' else 'AUC'
    ax.set_xlabel(f'Teacher {metric_name} Fairness Difference', fontsize=12)
    ax.set_ylabel(f'Student {metric_name} Fairness Difference', fontsize=12)
    ax.set_title(f'Teacher vs Student {metric_name} Fairness (by Balance Level)', fontsize=14, pad=20)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig, balance_categories


# +
# Color by p/q/c
fig, stats = plot_teacher_vs_student_scatter(
    auc_dict_pxc, auc_dict_qxc, auc_dict_qxp, 
    metric_type='acc', color_by='p', save_path=f'{figs_folder}/acc_scatter_by_p.png'
)

fig, stats = plot_teacher_vs_student_scatter(
    auc_dict_pxc, auc_dict_qxc, auc_dict_qxp, 
    metric_type='acc', color_by='q', save_path=f'{figs_folder}/acc_scatter_by_q.png'
)

fig, stats = plot_teacher_vs_student_scatter(
    auc_dict_pxc, auc_dict_qxc, auc_dict_qxp, 
    metric_type='acc', color_by='c', save_path=f'{figs_folder}/acc_scatter_by_c.png'
)

fig, stats = plot_teacher_vs_student_scatter(
    auc_dict_pxc, auc_dict_qxc, auc_dict_qxp, 
    metric_type='acc', color_by='combined_balance', save_path=f'{figs_folder}/acc_scatter_by_combined.png'
)

fig, stats = plot_teacher_vs_student_scatter(
    auc_dict_pxc, auc_dict_qxc, auc_dict_qxp, 
    metric_type='auc', color_by='p', save_path=f'{figs_folder}/auc_scatter_by_p.png'
)

fig, stats = plot_teacher_vs_student_scatter(
    auc_dict_pxc, auc_dict_qxc, auc_dict_qxp, 
    metric_type='auc', color_by='q', save_path=f'{figs_folder}/auc_scatter_by_q.png'
)

fig, stats = plot_teacher_vs_student_scatter(
    auc_dict_pxc, auc_dict_qxc, auc_dict_qxp, 
    metric_type='auc', color_by='c', save_path=f'{figs_folder}/auc_scatter_by_c.png'
)

fig, stats = plot_teacher_vs_student_scatter(
    auc_dict_pxc, auc_dict_qxc, auc_dict_qxp, 
    metric_type='auc', color_by='combined_balance', save_path=f'{figs_folder}/auc_scatter_by_combined.png'
)

# Use discrete categories
fig, categories = plot_teacher_vs_student_scatter_discrete(
    auc_dict_pxc, auc_dict_qxc, auc_dict_qxp, 
    metric_type='acc', save_path=f'{figs_folder}/acc_scatter_discrete.png'
)

fig, categories = plot_teacher_vs_student_scatter_discrete(
    auc_dict_pxc, auc_dict_qxc, auc_dict_qxp, 
    metric_type='auc', save_path=f'{figs_folder}/auc_scatter_discrete.png'
)
