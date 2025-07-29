# +
import argparse
import math
import os
import shutil
from os import environ
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.optim as optim
from scipy.special import softmax
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    homogeneity_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from dataloader import check_sbm_edge_probabilities, load_data
from models import Model
from train_and_eval import run_inductive, run_transductive
from utils import (
    check_writable,
    compute_min_cut_loss,
    feature_prop,
    get_evaluator,
    get_logger,
    get_training_config,
    graph_split,
    set_seed,
)


# -

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--device", type=int, default=-1, help="CUDA device, -1 means CPU")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
        help="Logger levels for run {10: DEBUG, 20: INFO, 30: WARNING}",
    )
    parser.add_argument(
        "--console_log",
        action="store_true",
        help="Set to True to display log info in console",
    )
    parser.add_argument(
        "--output_path", type=str, default="outputs", help="Path to save outputs"
    )
    parser.add_argument(
        "--num_exp", type=int, default=1, help="Repeat how many experiments"
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default="tran",
        help="Experiment setting, one of [tran, ind]",
    )
    parser.add_argument(
        "--eval_interval", type=int, default=1, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Set to True to save the loss curves, trained model, and min-cut loss for the transductive setting",
    )

    """Dataset"""
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data")
    parser.add_argument(
        "--labelrate_train",
        type=int,
        default=20,
        help="How many labeled data per class as train set",
    )
    parser.add_argument(
        "--labelrate_val",
        type=int,
        default=30,
        help="How many labeled data per class in valid set",
    )
    parser.add_argument(
        "--split_idx",
        type=int,
        default=0,
        help="For Non-Homo datasets only, one of [0,1,2,3,4]",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.3,
        help="For sbm datasets only, group balance ratio",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=0.50,
        help="For sbm datasets only, group edge probability",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=0.3,
        help="For sbm datasets only, class balance ratio",
    )

    """Model"""
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="./train.conf.yaml",
        help="Path to model configeration",
    )
    parser.add_argument("--teacher", type=str, default="SAGE", help="Teacher model")
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Model number of layers"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Model hidden layer dimensions"
    )
    parser.add_argument("--dropout_ratio", type=float, default=0)
    parser.add_argument(
        "--norm_type", type=str, default="none", help="One of [none, batch, layer]"
    )

    """SAGE Specific"""
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--fan_out",
        type=str,
        default="5,5",
        help="Number of samples for each layer in SAGE. Length = num_layers",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for sampler"
    )

    """Optimization"""
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument(
        "--max_epoch", type=int, default=500, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stop is the score on validation set does not improve for how many epochs",
    )

    """Ablation"""
    parser.add_argument(
        "--feature_noise",
        type=float,
        default=0,
        help="add white noise to features for analysis, value in [0, 1] for noise level",
    )
    parser.add_argument(
        "--split_rate",
        type=float,
        default=0.2,
        help="Rate for graph split, see comment of graph_split for more details",
    )
    parser.add_argument(
        "--compute_min_cut",
        action="store_true",
        help="Set to True to compute and store the min-cut loss",
    )
    parser.add_argument(
        "--feature_aug_k",
        type=int,
        default=0,
        help="Augment node futures by aggregating feature_aug_k-hop neighbor features",
    )

    args = parser.parse_args()
    return args


def run(args):
    """
    Returns:
    score_lst: a list of evaluation results on test set.
    len(score_lst) = 1 for the transductive setting.
    len(score_lst) = 2 for the inductive/production setting.
    """

    """ Set seed, device, and logger """
    set_seed(args.seed)
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = "cpu"

    if args.feature_noise != 0:
        args.output_path = Path.cwd().joinpath(
            args.output_path, "noisy_features", f"noise_{args.feature_noise}"
        )

    if args.feature_aug_k > 0:
        args.output_path = Path.cwd().joinpath(
            args.output_path, "aug_features", f"aug_hop_{args.feature_aug_k}"
        )
        args.teacher = f"GA{args.feature_aug_k}{args.teacher}"

    if args.exp_setting == "tran":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "transductive",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )
    elif args.exp_setting == "ind":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "inductive",
            f"split_rate_{args.split_rate}",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )
    else:
        raise ValueError(f"Unknown experiment setting! {args.exp_setting}")
    args.output_dir = output_dir

    check_writable(output_dir, overwrite=False)
    logger = get_logger(output_dir.joinpath("log"), args.console_log, args.log_level)
    logger.info(f"output_dir: {output_dir}")

    """ Load data """
    
    g, labels, idx_train, idx_val, idx_test, sens = load_data(
        args.dataset,
        args.data_path,
        split_idx=args.split_idx,
        seed=args.seed,
        labelrate_train=args.labelrate_train,
        labelrate_val=args.labelrate_val,
        p=args.p,
        q=args.q,
        c=args.c
    )
    logger.info(f"Total {g.number_of_nodes()} nodes.")
    logger.info(f"Total {g.number_of_edges()} edges.")

    # extract features
    feats = g.ndata["feat"]
    args.feat_dim = g.ndata["feat"].shape[1]
    args.label_dim = labels.int().max().item() + 1
    
    
    if 0 < args.feature_noise <= 1:
        feats = (
            1 - args.feature_noise
        ) * feats + args.feature_noise * torch.randn_like(feats)

    """ Model config """
    conf = {}
    if args.model_config_path is not None:
        conf = get_training_config(args.model_config_path, args.teacher, args.dataset)
    conf = dict(args.__dict__, **conf)
    conf["device"] = device
    logger.info(f"conf: {conf}")

    """ Model init """
    model = Model(conf)
    optimizer = optim.Adam(
        model.parameters(), lr=conf["learning_rate"], weight_decay=conf["weight_decay"]
    )
    criterion = torch.nn.NLLLoss()
    evaluator = get_evaluator(conf["dataset"])

    """ Data split and run """
    loss_and_score = []
    if args.exp_setting == "tran":
        indices = (idx_train, idx_val, idx_test)

        # propagate node feature
        if args.feature_aug_k > 0:
            feats = feature_prop(feats, g, args.feature_aug_k)

        out, score_val, score_test = run_transductive(
            conf,
            model,
            g,
            feats,
            labels,
            indices,
            criterion,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
        )
        score_lst = [score_test]

    elif args.exp_setting == "ind":
        indices = graph_split(idx_train, idx_val, idx_test, args.split_rate, args.seed)

        # propagate node feature. The propagation for the observed graph only happens within the subgraph obs_g
        if args.feature_aug_k > 0:
            idx_obs = indices[3]
            obs_g = g.subgraph(idx_obs)
            obs_feats = feature_prop(feats[idx_obs], obs_g, args.feature_aug_k)
            feats = feature_prop(feats, g, args.feature_aug_k)
            feats[idx_obs] = obs_feats

        out, score_val, score_test_tran, score_test_ind = run_inductive(
            conf,
            model,
            g,
            feats,
            labels,
            indices,
            criterion,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
        )
        score_lst = [score_test_tran, score_test_ind]

    logger.info(
        f"num_layers: {conf['num_layers']}. hidden_dim: {conf['hidden_dim']}. dropout_ratio: {conf['dropout_ratio']}"
    )
    logger.info(f"# params {sum(p.numel() for p in model.parameters())}")

    """ Saving teacher outputs """
#     out_np = out.detach().cpu().numpy()
    out_np = out.detach().cpu().tolist()
    out_np = np.array(out_np)
    np.savez(output_dir.joinpath("out"), out_np)
    
    """ Calculating and outputting fairness metrics """
    # convert tensors to lists to np
    sens = sens.tolist()
    sens = np.array(sens)
    idx_test = idx_test.tolist()
    idx_test = np.array(idx_test)
    labels = labels.tolist()
    labels = np.array(labels)
    
    # convert model out to preds via argmax
    pred = out_np.argmax(axis=1)
    
    if args.dataset == "sbm":
     
        """ Checking if class and group balance are correct """
        # Subset for test set
        test_labels = labels[idx_test]
        test_sens = sens[idx_test]
        test_samples = len(idx_test)

        # Group sizes
        s0 = test_sens == 0
        s1 = test_sens == 1

        # Class counts per group
        y1_s0 = test_labels[s0].sum()
        y0_s0 = len(test_labels[s0]) - y1_s0

        y1_s1 = test_labels[s1].sum()
        y0_s1 = len(test_labels[s1]) - y1_s1

        # Sanity checks
        total_samples = len(test_labels)
        total_y1 = test_labels.sum()
        total_y0 = total_samples - total_y1

        # 1. Assert class balance
        class_1_ratio = float(total_y1) / total_samples
        expected_class_1 = 1-args.c
        std_error_class = math.sqrt(expected_class_1 * (1-expected_class_1) / test_samples)
        threshold_class = 3 * std_error_class
        assert abs(class_1_ratio - expected_class_1) < threshold_class, f"Class 1 ratio {class_1_ratio:.2f} doesn't match expected {expected_class_1:.2f}"

        # 2. Assert group balance
        s1_ratio = float(s1.sum()) / total_samples
        expected_group_0 = 1-args.p # default for p when varying q
        std_error_group = math.sqrt(expected_group_0 * (1-expected_group_0) / test_samples)
        threshold_group = 3 * std_error_group
        assert abs(s1_ratio - expected_group_0) < threshold_group, f"Group balance {s1_ratio:.2f} doesn't match expected {expected_group_0:.2f} since it is not within threshold {threshold_group:.2f}"

        """ Checking if edge probabilities are correct """
        src_nodes, tgt_nodes = g.edges(order='eid')
        num_edges = g.num_edges()

        check_sbm_edge_probabilities(torch.stack([src_nodes, tgt_nodes]),labels,sens,args.q)

    """ Calculating traditional fairness metrics """
    # calculate and output demographic parity and equal opportunity (original fairness metrics)
    dp_0 = pred[idx_test][sens[idx_test] == 0].mean()
    dp_1 = pred[idx_test][sens[idx_test] == 1].mean()
    eo_0 = pred[idx_test][(sens[idx_test] == 0) & (labels[idx_test] == 1)].mean()
    eo_1 = pred[idx_test][(sens[idx_test] == 1) & (labels[idx_test] == 1)].mean()
    print("dp 0: ", pred[idx_test][sens[idx_test] == 0].mean())
    print("dp 1: ", pred[idx_test][sens[idx_test] == 1].mean())
    print("eo 0: ", pred[idx_test][(sens[idx_test] == 0) & (labels[idx_test] == 1)].mean())
    print("eo 1: ", pred[idx_test][(sens[idx_test] == 1) & (labels[idx_test] == 1)].mean())
    
    dp = abs(pred[idx_test][sens[idx_test] == 0].mean() - pred[idx_test][sens[idx_test] == 1].mean())
    print("dp: ", dp)
    eo = abs(pred[idx_test][(sens[idx_test] == 0) & (labels[idx_test] == 1)].mean() - pred[idx_test][(sens[idx_test] == 1) & (labels[idx_test] == 1)].mean())
    print("eo: ", eo)
    
    print("acc:", (pred[idx_test] == labels[idx_test]).mean()) # overall accuracy
    print("acc diff:", ((pred[idx_test][(sens[idx_test] == 0)] == labels[idx_test][(sens[idx_test] == 0)]).mean()) - (pred[idx_test][(sens[idx_test] == 1)] == labels[idx_test][(sens[idx_test] == 1)]).mean())
    acc_diff = ((pred[idx_test][(sens[idx_test] == 0)] == labels[idx_test][(sens[idx_test] == 0)]).mean()) - (pred[idx_test][(sens[idx_test] == 1)] == labels[idx_test][(sens[idx_test] == 1)]).mean()
    
    trad_metrics = [dp_0, dp_1, eo_0, eo_1, dp, eo, acc_diff]
    
    if args.dataset == "sbm":
        np.save(f'saved_arrays/trad_metrics_teacher_{args.seed}_p={args.p}_q={args.q}_c={args.c}.npy', trad_metrics, allow_pickle=True)
    else:
        np.save(f'saved_arrays/trad_metrics_teacher_{args.seed}_{args.dataset}.npy', trad_metrics, allow_pickle=True)
    
    
    """ Calculating confusion matrices """
    overall_cm = confusion_matrix(labels[idx_test], pred[idx_test])
    group_0_cm = confusion_matrix(labels[idx_test][sens[idx_test] == 0], pred[idx_test][sens[idx_test] == 0])
    group_1_cm = confusion_matrix(labels[idx_test][sens[idx_test] == 1], pred[idx_test][sens[idx_test] == 1])
        
    if args.dataset == "sbm":
        np.save(f'saved_arrays/overall_cm_teacher_{args.seed}_p={args.p}_q={args.q}_c={args.c}.npy', overall_cm, allow_pickle=True)
        np.save(f'saved_arrays/group_0_cm_teacher_{args.seed}_p={args.p}_q={args.q}_c={args.c}.npy', group_0_cm, allow_pickle=True)
        np.save(f'saved_arrays/group_1_cm_teacher_{args.seed}_p={args.p}_q={args.q}_c={args.c}.npy', group_1_cm, allow_pickle=True)    
    else:
        np.save(f'saved_arrays/overall_cm_teacher_{args.seed}_{args.dataset}.npy', overall_cm, allow_pickle=True)
        np.save(f'saved_arrays/group_0_cm_teacher_{args.seed}_{args.dataset}.npy', group_0_cm, allow_pickle=True)
        np.save(f'saved_arrays/group_1_cm_teacher_{args.seed}_{args.dataset}.npy', group_1_cm, allow_pickle=True)
    
    """ Calculating AUC """
    # out_np is the logits array, use softmax to get probabilities
    probabilities = softmax(out_np, axis=1)
    assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities do not sum to 1 for each sample."
    
    labels = labels.ravel()
    
    if args.dataset not in ["cora", "citeseer", "pubmed"]:
        probabilities = probabilities[:, 1] # only use from the pos class
        
    macro_roc_auc_ovr = roc_auc_score(
        labels[idx_test],
        probabilities[idx_test],
        multi_class="ovr",
        average="macro",
    )

    # Check labels for each sensitive group
    labels_sens_0 = labels[idx_test][sens[idx_test] == 0]
    labels_sens_1 = labels[idx_test][sens[idx_test] == 1]
    
    # Assert that we have data for both sensitive groups
    assert len(labels_sens_0) > 0, "No samples found for sensitive group 0"
    assert len(labels_sens_1) > 0, "No samples found for sensitive group 1"

    # Assert that we have the expected number of classes in each group
    unique_classes = np.unique(labels)
    assert all(cls in np.unique(labels_sens_0) for cls in unique_classes) or len(labels_sens_0) < 10, \
        f"Sensitive group 0 missing some classes. Found: {np.unique(labels_sens_0)}"
    assert all(cls in np.unique(labels_sens_1) for cls in unique_classes) or len(labels_sens_1) < 10, \
        f"Sensitive group 1 missing some classes. Found: {np.unique(labels_sens_1)}"

    # Assert shapes are consistent for probabilities
    probs_sens_0 = probabilities[idx_test][sens[idx_test] == 0]
    probs_sens_1 = probabilities[idx_test][sens[idx_test] == 1]
    assert probs_sens_0.shape[0] == labels_sens_0.shape[0], "Mismatch between probabilities and labels for group 0"
    assert probs_sens_1.shape[0] == labels_sens_1.shape[0], "Mismatch between probabilities and labels for group 1"
    
    # Calculate the AUC for group 0 and group 1
    try:
        macro_roc_auc_ovr_0 = roc_auc_score(
            labels[idx_test][sens[idx_test] == 0],
            probabilities[idx_test][sens[idx_test] == 0],
            multi_class="ovr",
            average="macro",
        )
    except:
        print("Exception calculating AUC for sensitive group 0")
        macro_roc_auc_ovr_0 = 0.5 # neutral score
        
    try:
    
        macro_roc_auc_ovr_1 = roc_auc_score(
            labels[idx_test][sens[idx_test] == 1],
            probabilities[idx_test][sens[idx_test] == 1],
            multi_class="ovr",
            average="macro",
        )
    except:
        print("Exception calculating AUC for sensitive group 1")
        macro_roc_auc_ovr_1 = 0.5 # neutral score
        
    macro_roc_auc_ovr_diff = macro_roc_auc_ovr_0 - macro_roc_auc_ovr_1

    print(f"auc diff: {macro_roc_auc_ovr_diff}")
    if args.dataset == "sbm":
        np.save(f'saved_arrays/auc_ovr_diff_teacher_{args.seed}_p={args.p}_q={args.q}_c={args.c}.npy', macro_roc_auc_ovr_diff, allow_pickle=True)
        np.save(f'saved_arrays/auc_ovr_overall_teacher_{args.seed}_p={args.p}_q={args.q}_c={args.c}.npy', macro_roc_auc_ovr, allow_pickle=True)
    else:
        np.save(f'saved_arrays/auc_ovr_diff_teacher_{args.seed}_{args.dataset}.npy', macro_roc_auc_ovr_diff, allow_pickle=True)
        np.save(f'saved_arrays/auc_ovr_overall_teacher_{args.seed}_{args.dataset}.npy', macro_roc_auc_ovr, allow_pickle=True)

    """ Saving loss curve and model """
    if args.save_results:
        # Loss curves
        loss_and_score = np.array(loss_and_score)
        np.savez(output_dir.joinpath("loss_and_score"), loss_and_score)

        # Model
        torch.save(model.state_dict(), output_dir.joinpath("model.pth"))

    """ Saving min-cut loss """
    if args.exp_setting == "tran" and args.compute_min_cut:
        min_cut = compute_min_cut_loss(g, out)
        with open(output_dir.parent.joinpath("min_cut_loss"), "a+") as f:
            f.write(f"{min_cut :.4f}\n")

    score = score_lst
    score_str = "".join([f"{s : .4f}\t" for s in score])
    
    if args.dataset == "sbm":
        np.save(f'saved_arrays/acc_teacher_{args.seed}_p={args.p}_q={args.q}_c={args.c}.npy', float(score_str), allow_pickle=True)
    else:
        np.save(f'saved_arrays/acc_teacher_{args.seed}_{args.dataset}.npy', float(score_str), allow_pickle=True)
    
    return score_lst


def repeat_run(args):
    scores = []
    for seed in range(args.num_exp):
        args.seed = seed
        scores.append(run(args))
    scores_np = np.array(scores)
    return scores_np.mean(axis=0), scores_np.std(axis=0)


def main():
    args = get_args()
    if args.num_exp == 1:
        score = run(args)
        score_str = "".join([f"{s : .4f}\t" for s in score])

    elif args.num_exp > 1:
        score_mean, score_std = repeat_run(args)
        score_str = "".join(
            [f"{s : .4f}\t" for s in score_mean] + [f"{s : .4f}\t" for s in score_std]
        )

    with open(args.output_dir.parent.joinpath("exp_results"), "a+") as f:
        f.write(f"{score_str[0]}\n")

    # for collecting aggregated results
    print("test acc: ", score_str)
        
    if args.dataset == "sbm":
        np.save(f'saved_arrays/acc_teacher_{args.seed}_p={args.p}_q={args.q}_c={args.c}.npy', float(score_str), allow_pickle=True)
    else:
        np.save(f'saved_arrays/acc_teacher_{args.seed}_{args.dataset}.npy', float(score_str), allow_pickle=True)


if __name__ == "__main__":
    main()
