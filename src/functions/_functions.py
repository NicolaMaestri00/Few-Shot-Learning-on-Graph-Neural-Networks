"""    Experiment functions    """

import pathlib

import concurrent.futures
import numpy as np
import pandas as pd
import torch

import models
import utils


def baseline(config: dict, data, shots: list, device: torch.device, results_path: str='./results/') -> None:
    '''    Baseline function for node classification tasks    '''

    num_classes = int(data.y.max().item()) + 1
    in_channels = data.num_features

    # Load prevoious results if they exist or create new DataFrames.
    if pathlib.Path(results_path).exists() and any(pathlib.Path(results_path).iterdir()):
        avg_acc_df = pd.read_csv(f'{results_path}/avg_acc.csv')
        std_acc_df = pd.read_csv(f'{results_path}/std_acc.csv')
        avg_f1_df = pd.read_csv(f'{results_path}/avg_f1.csv')
        std_f1_df = pd.read_csv(f'{results_path}/std_f1.csv')
    else:
        results_path = pathlib.Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        avg_acc_df, std_acc_df, avg_f1_df, std_f1_df = utils.metric_dfs(shots)

    avg_accuracies, std_accuracies, avg_f1_scores, std_f1_scores = [], [], [], []

    def run_experiment(train_mask):
        # Instantiate the model
        model = models.GCN(in_channels, config["model"]["hidden_channels"], num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), **config["optimizer"])
        # Train and evaluate the model
        train_df = utils.training(model, data, train_mask, data.val_mask, optimizer, **config["training"])
        acc, f1, _, _ = utils.test(model, data, data.test_mask)
        return acc, f1, train_df

    print(f'\n\n{"-"*20} Baseline {"-"*20}')

    # Loop over different training masks
    for n_shots, train_mask in shots:
        print(f'\nShots: {n_shots}')
        accuracies = []
        f1_scores = []
        train_dfs = []

        # Parallelize the independent runs
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_experiment, train_mask) for _ in range(config["n_runs"])]
            for future in concurrent.futures.as_completed(futures):
                acc, f1, train_df = future.result()
                accuracies.append(acc)
                f1_scores.append(f1)
                train_dfs.append(train_df)

        # Compute statistics
        accuracies = np.array(accuracies)
        f1_scores = np.array(f1_scores)
        avg_acc = round(accuracies.mean() * 100, 2)
        std_acc = round(accuracies.std() * 100, 2)
        avg_f1 = round(f1_scores.mean() * 100, 2)
        std_f1 = round(f1_scores.std() * 100, 2)

        avg_accuracies.append(avg_acc)
        std_accuracies.append(std_acc)
        avg_f1_scores.append(avg_f1)
        std_f1_scores.append(std_f1)

        print(f"\tAvg. Accuracy: {avg_acc:.2f} ± {std_acc:.2f} %\t\tAccuracy: [{', '.join([f'{acc*100:.2f}' for acc in accuracies])}]")
        print(f"\tAvg. F1 Score: {avg_f1:.2f} ± {std_f1:.2f} %\t\tF1 Score: [{', '.join([f'{f1*100:.2f}' for f1 in f1_scores])}]")

    # Log the results to the DataFrames
    avg_acc_df.loc[len(avg_acc_df)] = ["Baseline"] + avg_accuracies
    std_acc_df.loc[len(std_acc_df)] = ["Baseline"] + std_accuracies
    avg_f1_df.loc[len(avg_f1_df)] = ["Baseline"] + avg_f1_scores
    std_f1_df.loc[len(std_f1_df)] = ["Baseline"] + std_f1_scores

    # Save the DataFrames to CSV files
    dfs = [avg_acc_df, std_acc_df, avg_f1_df, std_f1_df]
    file_names = ["avg_acc", "std_acc", "avg_f1", "std_f1"]
    utils.save_dfs(dfs=dfs, file_path=results_path, file_names=file_names)


def reg_baseline(config: dict, data, shots: list, device: torch.device, results_path: str='./results/', csr=False, er=False) -> None:
    '''    Regularized baseline for node classification tasks     '''

    num_classes = int(data.y.max().item()) + 1
    in_channels = data.num_features

    # Load prevoious results if they exist or create new DataFrames.
    if pathlib.Path(results_path).exists() and any(pathlib.Path(results_path).iterdir()):
        avg_acc_df = pd.read_csv(f'{results_path}/avg_acc.csv')
        std_acc_df = pd.read_csv(f'{results_path}/std_acc.csv')
        avg_f1_df = pd.read_csv(f'{results_path}/avg_f1.csv')
        std_f1_df = pd.read_csv(f'{results_path}/std_f1.csv')
    else:
        results_path = pathlib.Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        avg_acc_df, std_acc_df, avg_f1_df, std_f1_df = utils.metric_dfs(shots)

    avg_accuracies, std_accuracies, avg_f1_scores, std_f1_scores = [], [], [], []

    def run_experiment(run, train_mask):
        # Instantiate the model
        model = models.GCN(in_channels, config["model"]["hidden_channels"], num_classes, cs_reg=csr).to(device)     # Cosine Similarity Regularization
        optimizer = torch.optim.Adam(model.parameters(), **config["optimizer"])
        # Train and evaluate the model
        if er:    # Entropy Regularization
            train_df = utils.training(model, data, train_mask, data.val_mask, optimizer, lambda_entropy=-0.5, **config["training"])
        else:
            train_df = utils.training(model, data, train_mask, data.val_mask, optimizer, **config["training"])
        acc, f1, _, _ = utils.test(model, data, data.test_mask)
        return acc, f1, train_df

    if csr:
        print(f'\n\n{"-"*20} Cosine Similarity Regularization {"-"*20}')
    elif er:
        print(f'\n\n{"-"*20} Entropy Regularization {"-"*20}')

    # Loop over different training masks
    for n_shots, train_mask in shots:
        print(f'\nShots: {n_shots}')
        accuracies = []
        f1_scores = []
        train_dfs = []

        # Parallelize the independent runs
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_experiment, run, train_mask) for run in range(config["n_runs"])]
            for future in concurrent.futures.as_completed(futures):
                acc, f1, train_df = future.result()
                accuracies.append(acc)
                f1_scores.append(f1)
                train_dfs.append(train_df)

        # Compute statistics
        accuracies = np.array(accuracies)
        f1_scores = np.array(f1_scores)
        avg_acc = round(accuracies.mean() * 100, 2)
        std_acc = round(accuracies.std() * 100, 2)
        avg_f1 = round(f1_scores.mean() * 100, 2)
        std_f1 = round(f1_scores.std() * 100, 2)

        avg_accuracies.append(avg_acc)
        std_accuracies.append(std_acc)
        avg_f1_scores.append(avg_f1)
        std_f1_scores.append(std_f1)

        print(f"\tAvg. Accuracy: {avg_acc:.2f} ± {std_acc:.2f} %\t\tAccuracy: [{', '.join([f'{acc*100:.2f}' for acc in accuracies])}]")
        print(f"\tAvg. F1 Score: {avg_f1:.2f} ± {std_f1:.2f} %\t\tF1 Score: [{', '.join([f'{f1*100:.2f}' for f1 in f1_scores])}]")

    # Log the results to the DataFrames
    if csr:
        avg_acc_df.loc[len(avg_acc_df)] = ["CSR"] + avg_accuracies
        std_acc_df.loc[len(std_acc_df)] = ["CSR"] + std_accuracies
        avg_f1_df.loc[len(avg_f1_df)] = ["CSR"] + avg_f1_scores
        std_f1_df.loc[len(std_f1_df)] = ["CSR"] + std_f1_scores
    elif er:
        avg_acc_df.loc[len(avg_acc_df)] = ["ER"] + avg_accuracies
        std_acc_df.loc[len(std_acc_df)] = ["ER"] + std_accuracies
        avg_f1_df.loc[len(avg_f1_df)] = ["ER"] + avg_f1_scores
        std_f1_df.loc[len(std_f1_df)] = ["ER"] + std_f1_scores

    # Save the DataFrames to CSV files
    dfs = [avg_acc_df, std_acc_df, avg_f1_df, std_f1_df]
    file_names = ["avg_acc", "std_acc", "avg_f1", "std_f1"]
    utils.save_dfs(dfs=dfs, file_path=results_path, file_names=file_names)


def drop_message(config: dict, data, shots: list, device: torch.device, results_path: str='./results/') -> None:
    '''    DropMessage    '''

    num_classes = int(data.y.max().item()) + 1
    in_channels = data.num_features

    # Load prevoious results if they exist or create new DataFrames.
    if pathlib.Path(results_path).exists() and any(pathlib.Path(results_path).iterdir()):
        avg_acc_df = pd.read_csv(f'{results_path}/avg_acc.csv')
        std_acc_df = pd.read_csv(f'{results_path}/std_acc.csv')
        avg_f1_df = pd.read_csv(f'{results_path}/avg_f1.csv')
        std_f1_df = pd.read_csv(f'{results_path}/std_f1.csv')
    else:
        results_path = pathlib.Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        avg_acc_df, std_acc_df, avg_f1_df, std_f1_df = utils.metric_dfs(shots)

    avg_accuracies, std_accuracies, avg_f1_scores, std_f1_scores = [], [], [], []

    def run_experiment(train_mask):
        # Instantiate the model
        model = models.GNN_DropMessage(in_channels, config["model"]["hidden_channels"], num_classes, drop_rate=config["model"]["drop_msg"]).to(device)  # DropMessage
        optimizer = torch.optim.Adam(model.parameters(), **config["optimizer"])
        # Train and evaluate the model
        train_df = utils.training(model, data, train_mask, data.val_mask, optimizer, lambda_entropy=-0.5, **config["training"])
        acc, f1, _, _ = utils.test(model, data, data.test_mask)
        return acc, f1, train_df

    print(f'\n\n{"-"*20} DropMessage {"-"*20}')

    # Loop over different training masks
    for n_shots, train_mask in shots:
        print(f'\nShots: {n_shots}')
        accuracies = []
        f1_scores = []
        train_dfs = []

        # Parallelize the independent runs
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_experiment, train_mask) for _ in range(config["n_runs"])]
            for future in concurrent.futures.as_completed(futures):
                acc, f1, train_df = future.result()
                accuracies.append(acc)
                f1_scores.append(f1)
                train_dfs.append(train_df)

        # Compute statistics
        accuracies = np.array(accuracies)
        f1_scores = np.array(f1_scores)
        avg_acc = round(accuracies.mean() * 100, 2)
        std_acc = round(accuracies.std() * 100, 2)
        avg_f1 = round(f1_scores.mean() * 100, 2)
        std_f1 = round(f1_scores.std() * 100, 2)

        avg_accuracies.append(avg_acc)
        std_accuracies.append(std_acc)
        avg_f1_scores.append(avg_f1)
        std_f1_scores.append(std_f1)

        print(f"\tAvg. Accuracy: {avg_acc:.2f} ± {std_acc:.2f} %\t\tAccuracy: [{', '.join([f'{acc*100:.2f}' for acc in accuracies])}]")
        print(f"\tAvg. F1 Score: {avg_f1:.2f} ± {std_f1:.2f} %\t\tF1 Score: [{', '.join([f'{f1*100:.2f}' for f1 in f1_scores])}]")

    # Log the results to the DataFrames
    avg_acc_df.loc[len(avg_acc_df)] = ["DropMessage"] + avg_accuracies
    std_acc_df.loc[len(std_acc_df)] = ["DropMessage"] + std_accuracies
    avg_f1_df.loc[len(avg_f1_df)] = ["DropMessage"] + avg_f1_scores
    std_f1_df.loc[len(std_f1_df)] = ["DropMessage"] + std_f1_scores

    # Save the DataFrames to CSV files
    dfs = [avg_acc_df, std_acc_df, avg_f1_df, std_f1_df]
    file_names = ["avg_acc", "std_acc", "avg_f1", "std_f1"]
    utils.save_dfs(dfs=dfs, file_path=results_path, file_names=file_names)


def dropout(config: dict, data, shots: list, device: torch.device, drop_strategy="DropNode", results_path: str='./results/') -> None:
    '''    Dropout    '''

    num_classes = int(data.y.max().item()) + 1
    in_channels = data.num_features

    # Load prevoious results if they exist or create new DataFrames.
    if pathlib.Path(results_path).exists() and any(pathlib.Path(results_path).iterdir()):
        avg_acc_df = pd.read_csv(f'{results_path}/avg_acc.csv')
        std_acc_df = pd.read_csv(f'{results_path}/std_acc.csv')
        avg_f1_df = pd.read_csv(f'{results_path}/avg_f1.csv')
        std_f1_df = pd.read_csv(f'{results_path}/std_f1.csv')
    else:
        results_path = pathlib.Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        avg_acc_df, std_acc_df, avg_f1_df, std_f1_df = utils.metric_dfs(shots)

    avg_accuracies, std_accuracies, avg_f1_scores, std_f1_scores = [], [], [], []

    def run_experiment(train_mask):
        # Instantiate the model
        model = models.GNN(in_channels, config["model"]["hidden_channels"], num_classes, drop_strategy, drop_rate=config["model"][drop_strategy]).to(device)  # DropNode
        optimizer = torch.optim.Adam(model.parameters(), **config["optimizer"])
        # Train and evaluate the model
        train_df = utils.training(model, data, train_mask, data.val_mask, optimizer, lambda_entropy=-0.5, **config["training"])
        acc, f1, _, _ = utils.test(model, data, data.test_mask)
        return acc, f1, train_df

    print(f'\n\n{"-"*20} {drop_strategy} {"-"*20}')

    # Loop over different training masks
    for n_shots, train_mask in shots:
        print(f'\nShots: {n_shots}')
        accuracies = []
        f1_scores = []
        train_dfs = []

        # Parallelize the independent runs
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_experiment, train_mask) for _ in range(config["n_runs"])]
            for future in concurrent.futures.as_completed(futures):
                acc, f1, train_df = future.result()
                accuracies.append(acc)
                f1_scores.append(f1)
                train_dfs.append(train_df)

        # Compute statistics
        accuracies = np.array(accuracies)
        f1_scores = np.array(f1_scores)
        avg_acc = round(accuracies.mean() * 100, 2)
        std_acc = round(accuracies.std() * 100, 2)
        avg_f1 = round(f1_scores.mean() * 100, 2)
        std_f1 = round(f1_scores.std() * 100, 2)

        avg_accuracies.append(avg_acc)
        std_accuracies.append(std_acc)
        avg_f1_scores.append(avg_f1)
        std_f1_scores.append(std_f1)

        print(f"\tAvg. Accuracy: {avg_acc:.2f} ± {std_acc:.2f} %\t\tAccuracy: [{', '.join([f'{acc*100:.2f}' for acc in accuracies])}]")
        print(f"\tAvg. F1 Score: {avg_f1:.2f} ± {std_f1:.2f} %\t\tF1 Score: [{', '.join([f'{f1*100:.2f}' for f1 in f1_scores])}]")

    # Log the results to the DataFrames
    avg_acc_df.loc[len(avg_acc_df)] = [drop_strategy] + avg_accuracies
    std_acc_df.loc[len(std_acc_df)] = [drop_strategy] + std_accuracies
    avg_f1_df.loc[len(avg_f1_df)] = [drop_strategy] + avg_f1_scores
    std_f1_df.loc[len(std_f1_df)] = [drop_strategy] + std_f1_scores

    # Save the DataFrames to CSV files
    dfs = [avg_acc_df, std_acc_df, avg_f1_df, std_f1_df]
    file_names = ["avg_acc", "std_acc", "avg_f1", "std_f1"]
    utils.save_dfs(dfs=dfs, file_path=results_path, file_names=file_names)


def augmentation(config: dict, data, shots: list, device: torch.device, results_path: str='./results/') -> None:
    '''   Augmentation    '''

    num_classes = int(data.y.max().item()) + 1
    in_channels = data.num_features

    # Load prevoious results if they exist or create new DataFrames.
    if pathlib.Path(results_path).exists() and any(pathlib.Path(results_path).iterdir()):
        avg_acc_df = pd.read_csv(f'{results_path}/avg_acc.csv')
        std_acc_df = pd.read_csv(f'{results_path}/std_acc.csv')
        avg_f1_df = pd.read_csv(f'{results_path}/avg_f1.csv')
        std_f1_df = pd.read_csv(f'{results_path}/std_f1.csv')
    else:
        results_path = pathlib.Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        avg_acc_df, std_acc_df, avg_f1_df, std_f1_df = utils.metric_dfs(shots)

    avg_accuracies, std_accuracies, avg_f1_scores, std_f1_scores = [], [], [], []

    def run_experiment(train_mask):
        # Instantiate the model
        model = models.GCN(in_channels, config["model"]["hidden_channels"], num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), **config["optimizer"])
        # Train and evaluate the model
        train_df = utils.training_augm(model, data, train_mask, data.val_mask, optimizer, device=device, lambda_entropy=-0.5, augm=config["model"]["augm"], **config["training"])
        acc, f1, _, _ = utils.test(model, data, data.test_mask)
        return acc, f1, train_df

    print(f'\n\n{"-"*20} Augmentation {"-"*20}')

    # Loop over different training masks
    for n_shots, train_mask in shots:
        print(f'\nShots: {n_shots}')
        accuracies = []
        f1_scores = []
        train_dfs = []

        # Parallelize the independent runs
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_experiment, train_mask) for _ in range(config["n_runs"])]
            for future in concurrent.futures.as_completed(futures):
                acc, f1, train_df = future.result()
                accuracies.append(acc)
                f1_scores.append(f1)
                train_dfs.append(train_df)

        # Compute statistics
        accuracies = np.array(accuracies)
        f1_scores = np.array(f1_scores)
        avg_acc = round(accuracies.mean() * 100, 2)
        std_acc = round(accuracies.std() * 100, 2)
        avg_f1 = round(f1_scores.mean() * 100, 2)
        std_f1 = round(f1_scores.std() * 100, 2)

        avg_accuracies.append(avg_acc)
        std_accuracies.append(std_acc)
        avg_f1_scores.append(avg_f1)
        std_f1_scores.append(std_f1)

        print(f"\tAvg. Accuracy: {avg_acc:.2f} ± {std_acc:.2f} %\t\tAccuracy: [{', '.join([f'{acc*100:.2f}' for acc in accuracies])}]")
        print(f"\tAvg. F1 Score: {avg_f1:.2f} ± {std_f1:.2f} %\t\tF1 Score: [{', '.join([f'{f1*100:.2f}' for f1 in f1_scores])}]")

    # Log the results to the DataFrames
    avg_acc_df.loc[len(avg_acc_df)] = ["Augmentation"] + avg_accuracies
    std_acc_df.loc[len(std_acc_df)] = ["Augmentation"] + std_accuracies
    avg_f1_df.loc[len(avg_f1_df)] = ["Augmentation"] + avg_f1_scores
    std_f1_df.loc[len(std_f1_df)] = ["Augmentation"] + std_f1_scores

    # Save the DataFrames to CSV files
    dfs = [avg_acc_df, std_acc_df, avg_f1_df, std_f1_df]
    file_names = ["avg_acc", "std_acc", "avg_f1", "std_f1"]
    utils.save_dfs(dfs=dfs, file_path=results_path, file_names=file_names)


def rewiring(config: dict, data, shots: list, device: torch.device, results_path: str='./results/') -> None:
    '''   Rewiring    '''

    num_classes = int(data.y.max().item()) + 1
    in_channels = data.num_features

    # Load prevoious results if they exist or create new DataFrames.
    if pathlib.Path(results_path).exists() and any(pathlib.Path(results_path).iterdir()):
        avg_acc_df = pd.read_csv(f'{results_path}/avg_acc.csv')
        std_acc_df = pd.read_csv(f'{results_path}/std_acc.csv')
        avg_f1_df = pd.read_csv(f'{results_path}/avg_f1.csv')
        std_f1_df = pd.read_csv(f'{results_path}/std_f1.csv')
    else:
        results_path = pathlib.Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        avg_acc_df, std_acc_df, avg_f1_df, std_f1_df = utils.metric_dfs(shots)

    avg_accuracies, std_accuracies, avg_f1_scores, std_f1_scores = [], [], [], []

    def run_experiment(train_mask):
        # Instantiate the model
        model = models.GCN(in_channels, config["model"]["hidden_channels"], num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), **config["optimizer"])
        # Train and evaluate the model
        train_df = utils.training_rew(model, data, train_mask, data.val_mask, optimizer, lambda_entropy=-0.5, rew=config["model"]["rew"], **config["training"])
        acc, f1, _, _ = utils.test(model, data, data.test_mask)
        return acc, f1, train_df

    print(f'\n\n{"-"*20} Rewiring {"-"*20}')

    # Loop over different training masks
    for n_shots, train_mask in shots:
        print(f'\nShots: {n_shots}')
        accuracies = []
        f1_scores = []
        train_dfs = []

        # Parallelize the independent runs
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_experiment, train_mask) for _ in range(config["n_runs"])]
            for future in concurrent.futures.as_completed(futures):
                acc, f1, train_df = future.result()
                accuracies.append(acc)
                f1_scores.append(f1)
                train_dfs.append(train_df)

        # Compute statistics
        accuracies = np.array(accuracies)
        f1_scores = np.array(f1_scores)
        avg_acc = round(accuracies.mean() * 100, 2)
        std_acc = round(accuracies.std() * 100, 2)
        avg_f1 = round(f1_scores.mean() * 100, 2)
        std_f1 = round(f1_scores.std() * 100, 2)

        avg_accuracies.append(avg_acc)
        std_accuracies.append(std_acc)
        avg_f1_scores.append(avg_f1)
        std_f1_scores.append(std_f1)

        print(f"\tAvg. Accuracy: {avg_acc:.2f} ± {std_acc:.2f} %\t\tAccuracy: [{', '.join([f'{acc*100:.2f}' for acc in accuracies])}]")
        print(f"\tAvg. F1 Score: {avg_f1:.2f} ± {std_f1:.2f} %\t\tF1 Score: [{', '.join([f'{f1*100:.2f}' for f1 in f1_scores])}]")

    # Log the results to the DataFrames
    avg_acc_df.loc[len(avg_acc_df)] = ["Rewiring"] + avg_accuracies
    std_acc_df.loc[len(std_acc_df)] = ["Rewiring"] + std_accuracies
    avg_f1_df.loc[len(avg_f1_df)] = ["Rewiring"] + avg_f1_scores
    std_f1_df.loc[len(std_f1_df)] = ["Rewiring"] + std_f1_scores

    # Save the DataFrames to CSV files
    dfs = [avg_acc_df, std_acc_df, avg_f1_df, std_f1_df]
    file_names = ["avg_acc", "std_acc", "avg_f1", "std_f1"]
    utils.save_dfs(dfs=dfs, file_path=results_path, file_names=file_names)


def protonet(config: dict, data, shots: list, device: torch.device, results_path: str='./results/') -> None:
    '''   ProtoNet    '''

    num_classes = int(data.y.max().item()) + 1
    in_channels = data.num_features

    # Load prevoious results if they exist or create new DataFrames.
    if pathlib.Path(results_path).exists() and any(pathlib.Path(results_path).iterdir()):
        avg_acc_df = pd.read_csv(f'{results_path}/avg_acc.csv')
        std_acc_df = pd.read_csv(f'{results_path}/std_acc.csv')
        avg_f1_df = pd.read_csv(f'{results_path}/avg_f1.csv')
        std_f1_df = pd.read_csv(f'{results_path}/std_f1.csv')
    else:
        results_path = pathlib.Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        avg_acc_df, std_acc_df, avg_f1_df, std_f1_df = utils.metric_dfs(shots)

    avg_accuracies, std_accuracies, avg_f1_scores, std_f1_scores = [], [], [], []

    def run_experiment(train_mask):
        # Instantiate the model
        model = models.GCN(in_channels, config["model"]["hidden_channels"], num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), **config["optimizer"])
        # Train and evaluate the model
        train_df = utils.training_protonet(model, data, train_mask, data.val_mask, optimizer, **config["training"])
        prototypes = utils.get_prototypes(model, data, data.test_mask)
        acc, f1, _, _ = utils.test_protonet(model, data, data.test_mask, prototypes)
        return acc, f1, train_df

    print(f'\n\n{"-"*20} ProtoNet {"-"*20}')

    # Loop over different training masks
    for n_shots, train_mask in shots:
        print(f'\nShots: {n_shots}')
        accuracies = []
        f1_scores = []
        train_dfs = []

        # Parallelize the independent runs
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_experiment, train_mask) for _ in range(config["n_runs"])]
            for future in concurrent.futures.as_completed(futures):
                acc, f1, train_df = future.result()
                accuracies.append(acc)
                f1_scores.append(f1)
                train_dfs.append(train_df)

        # Compute statistics
        accuracies = np.array(accuracies)
        f1_scores = np.array(f1_scores)
        avg_acc = round(accuracies.mean() * 100, 2)
        std_acc = round(accuracies.std() * 100, 2)
        avg_f1 = round(f1_scores.mean() * 100, 2)
        std_f1 = round(f1_scores.std() * 100, 2)

        avg_accuracies.append(avg_acc)
        std_accuracies.append(std_acc)
        avg_f1_scores.append(avg_f1)
        std_f1_scores.append(std_f1)

        print(f"\tAvg. Accuracy: {avg_acc:.2f} ± {std_acc:.2f} %\t\tAccuracy: [{', '.join([f'{acc*100:.2f}' for acc in accuracies])}]")
        print(f"\tAvg. F1 Score: {avg_f1:.2f} ± {std_f1:.2f} %\t\tF1 Score: [{', '.join([f'{f1*100:.2f}' for f1 in f1_scores])}]")

    # Log the results to the DataFrames
    avg_acc_df.loc[len(avg_acc_df)] = ["ProtoNet"] + avg_accuracies
    std_acc_df.loc[len(std_acc_df)] = ["ProtoNet"] + std_accuracies
    avg_f1_df.loc[len(avg_f1_df)] = ["ProtoNet"] + avg_f1_scores
    std_f1_df.loc[len(std_f1_df)] = ["ProtoNet"] + std_f1_scores

    # Save the DataFrames to CSV files
    dfs = [avg_acc_df, std_acc_df, avg_f1_df, std_f1_df]
    file_names = ["avg_acc", "std_acc", "avg_f1", "std_f1"]
    utils.save_dfs(dfs=dfs, file_path=results_path, file_names=file_names)


def siamesenet(config: dict, data, shots: list, device: torch.device, results_path: str='./results/', CL_Loss='TPL') -> None:
    '''    Siamese Network    '''

    num_classes = int(data.y.max().item()) + 1
    in_channels = data.num_features

    # Load prevoious results if they exist or create new DataFrames.
    if pathlib.Path(results_path).exists() and any(pathlib.Path(results_path).iterdir()):
        avg_acc_df = pd.read_csv(f'{results_path}/avg_acc.csv')
        std_acc_df = pd.read_csv(f'{results_path}/std_acc.csv')
        avg_f1_df = pd.read_csv(f'{results_path}/avg_f1.csv')
        std_f1_df = pd.read_csv(f'{results_path}/std_f1.csv')
    else:
        results_path = pathlib.Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        avg_acc_df, std_acc_df, avg_f1_df, std_f1_df = utils.metric_dfs(shots)

    avg_accuracies, std_accuracies, avg_f1_scores, std_f1_scores = [], [], [], []

    def run_experiment(train_mask):
        # Instantiate the model
        model = models.GCN(in_channels, config["model"]["hidden_channels"], num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), **config["optimizer"])
        # Train and evaluate the model
        train_df = utils.training_siamesenet(model, data, train_mask, data.val_mask, optimizer, CL_Loss=CL_Loss, lambda_entropy=-0.5, device=device, **config["training"])
        acc, f1, _, _ = utils.test(model, data, data.test_mask)
        return acc, f1, train_df

    print(f'\n\n{"-"*20} {"Siamese_"+CL_Loss} {"-"*20}')

    # Loop over different training masks
    for n_shots, train_mask in shots:
        print(f'\nShots: {n_shots}')
        accuracies = []
        f1_scores = []
        train_dfs = []

        # Parallelize the independent runs
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_experiment, train_mask) for _ in range(config["n_runs"])]
            for future in concurrent.futures.as_completed(futures):
                acc, f1, train_df = future.result()
                accuracies.append(acc)
                f1_scores.append(f1)
                train_dfs.append(train_df)

        # Compute statistics
        accuracies = np.array(accuracies)
        f1_scores = np.array(f1_scores)
        avg_acc = round(accuracies.mean() * 100, 2)
        std_acc = round(accuracies.std() * 100, 2)
        avg_f1 = round(f1_scores.mean() * 100, 2)
        std_f1 = round(f1_scores.std() * 100, 2)

        avg_accuracies.append(avg_acc)
        std_accuracies.append(std_acc)
        avg_f1_scores.append(avg_f1)
        std_f1_scores.append(std_f1)

        print(f"\tAvg. Accuracy: {avg_acc:.2f} ± {std_acc:.2f} %\t\tAccuracy: [{', '.join([f'{acc*100:.2f}' for acc in accuracies])}]")
        print(f"\tAvg. F1 Score: {avg_f1:.2f} ± {std_f1:.2f} %\t\tF1 Score: [{', '.join([f'{f1*100:.2f}' for f1 in f1_scores])}]")

    # Log the results to the DataFrames
    avg_acc_df.loc[len(avg_acc_df)] = ["Siamese_"+CL_Loss] + avg_accuracies
    std_acc_df.loc[len(std_acc_df)] = ["Siamese_"+CL_Loss] + std_accuracies
    avg_f1_df.loc[len(avg_f1_df)] = ["Siamese_"+CL_Loss] + avg_f1_scores
    std_f1_df.loc[len(std_f1_df)] = ["Siamese_"+CL_Loss] + std_f1_scores

    # Save the DataFrames to CSV files
    dfs = [avg_acc_df, std_acc_df, avg_f1_df, std_f1_df]
    file_names = ["avg_acc", "std_acc", "avg_f1", "std_f1"]
    utils.save_dfs(dfs=dfs, file_path=results_path, file_names=file_names)


def graphcl(config: dict, data, shots: list, device: torch.device, results_path: str='./results/') -> None:
    '''    GraphCL    '''

    num_classes = int(data.y.max().item()) + 1
    in_channels = data.num_features

    # Load prevoious results if they exist or create new DataFrames.
    if pathlib.Path(results_path).exists() and any(pathlib.Path(results_path).iterdir()):
        avg_acc_df = pd.read_csv(f'{results_path}/avg_acc.csv')
        std_acc_df = pd.read_csv(f'{results_path}/std_acc.csv')
        avg_f1_df = pd.read_csv(f'{results_path}/avg_f1.csv')
        std_f1_df = pd.read_csv(f'{results_path}/std_f1.csv')
    else:
        results_path = pathlib.Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        avg_acc_df, std_acc_df, avg_f1_df, std_f1_df = utils.metric_dfs(shots)

    avg_accuracies, std_accuracies, avg_f1_scores, std_f1_scores = [], [], [], []

    def run_experiment(train_mask):
        # Instantiate the model
        model = models.GCN(in_channels, config["model"]["hidden_channels"], num_classes).to(device)
        pretraining_opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        optimizer = torch.optim.Adam(model.parameters(), **config["optimizer"])
        # Train and evaluate the model
        utils.pretrain_graphcl(model, data, train_mask, pretraining_opt, epochs=100)
        train_df = utils.training(model, data, train_mask, data.val_mask, optimizer, lambda_entropy=-0.5, **config["training"])
        acc, f1, _, _ = utils.test(model, data, data.test_mask)
        return acc, f1, train_df

    print(f'\n\n{"-"*20} GraphCL {"-"*20}')

    # Loop over different training masks
    for n_shots, train_mask in shots:
        print(f'\nShots: {n_shots}')
        accuracies = []
        f1_scores = []
        train_dfs = []

        # Parallelize the independent runs
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_experiment, train_mask) for _ in range(config["n_runs"])]
            for future in concurrent.futures.as_completed(futures):
                acc, f1, train_df = future.result()
                accuracies.append(acc)
                f1_scores.append(f1)
                train_dfs.append(train_df)

        # Compute statistics
        accuracies = np.array(accuracies)
        f1_scores = np.array(f1_scores)
        avg_acc = round(accuracies.mean() * 100, 2)
        std_acc = round(accuracies.std() * 100, 2)
        avg_f1 = round(f1_scores.mean() * 100, 2)
        std_f1 = round(f1_scores.std() * 100, 2)

        avg_accuracies.append(avg_acc)
        std_accuracies.append(std_acc)
        avg_f1_scores.append(avg_f1)
        std_f1_scores.append(std_f1)

        print(f"\tAvg. Accuracy: {avg_acc:.2f} ± {std_acc:.2f} %\t\tAccuracy: [{', '.join([f'{acc*100:.2f}' for acc in accuracies])}]")
        print(f"\tAvg. F1 Score: {avg_f1:.2f} ± {std_f1:.2f} %\t\tF1 Score: [{', '.join([f'{f1*100:.2f}' for f1 in f1_scores])}]")

    # Log the results to the DataFrames
    avg_acc_df.loc[len(avg_acc_df)] = ["GraphCL"] + avg_accuracies
    std_acc_df.loc[len(std_acc_df)] = ["GraphCL"] + std_accuracies
    avg_f1_df.loc[len(avg_f1_df)] = ["GraphCL"] + avg_f1_scores
    std_f1_df.loc[len(std_f1_df)] = ["GraphCL"] + std_f1_scores

    # Save the DataFrames to CSV files
    dfs = [avg_acc_df, std_acc_df, avg_f1_df, std_f1_df]
    file_names = ["avg_acc", "std_acc", "avg_f1", "std_f1"]
    utils.save_dfs(dfs=dfs, file_path=results_path, file_names=file_names)
