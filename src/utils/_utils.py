'''    Utility functions    '''

import argparse
import copy
import pathlib
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score

import losses


def is_valid_file(path: str) -> str:
    """Check if the given path is a valid file"""
    file = pathlib.Path(path)
    if not file.is_file():
        raise argparse.ArgumentTypeError(f"{path} does not exist")
    return path


def get_parser(description: str) -> argparse.ArgumentParser:
    """Get the argument parser"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--conf",
        "-c",
        dest="config_file_path",
        required=True,
        metavar="FILE",
        type=lambda x: is_valid_file(x),  # type: ignore
        help="The YAML configuration file.",
    )
    return parser


def get_config(description: str="MNIST GAN") -> dict:
    """Get the configuration from the YAML file"""
    parser = get_parser(description)
    args = parser.parse_args()
    config_file = args.config_file_path
    with open(config_file) as yaml_file:
        config = yaml.full_load(yaml_file)
    return config


def metric_dfs(shots:list) -> tuple:
    avg_acc_df = pd.DataFrame(columns=['Method'] + [f'FS{s[0]}' for s in shots])
    std_acc_df = pd.DataFrame(columns=['Method'] + [f'FS{s[0]}' for s in shots])
    avg_f1_df = pd.DataFrame(columns=['Method'] + [f'FS{s[0]}' for s in shots])
    std_f1_df = pd.DataFrame(columns=['Method'] + [f'FS{s[0]}' for s in shots])
    return avg_acc_df, std_acc_df, avg_f1_df, std_f1_df


def save_dfs(dfs:list=[], file_path:str='./', file_names:list=[]) -> None:
    for df, file_name in zip(dfs, file_names):
        df.to_csv(f'{file_path}/{file_name}.csv', index=False)


def train_step(model, data, train_mask, optimizer, lambda_entropy=0):
    ''' Training step: yields average loss over batch. '''
    model.train()
    optimizer.zero_grad()
    _, out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    if lambda_entropy:
        loss += lambda_entropy * losses.entropy_regularization(out[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data, test_mask):
    ''' Evaluation step: yields test accuracy and F1 score. '''
    model.eval()
    _, out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=1)[test_mask]
    y_true = data.y[test_mask]
    test_correct = y_pred == y_true
    test_acc = int(test_correct.sum()) / int(test_mask.sum())
    test_f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')
    return test_acc, test_f1, y_pred, y_true


def training(model, data, train_mask, val_mask, optimizer, epochs=100, patience=50, lambda_entropy=0):
    ''' Training function '''
    rec_df = pd.DataFrame(columns=['train_loss', 'val_acc', 'val_f1'])
    best_score = 0
    no_improvements = 0
    best_model_state = None

    for epoch in range(1, epochs+1):
        train_loss = train_step(model, data, train_mask, optimizer, lambda_entropy)
        val_acc, val_f1, _, _ = test(model, data, val_mask)
        rec_df.loc[len(rec_df)] = [train_loss, val_acc, val_f1]
        if val_acc + val_f1 > best_score:
            best_score = val_acc + val_f1
            no_improvements = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            no_improvements += 1
            if no_improvements == patience:  # Early Stopping
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return rec_df


def augment_graph(graph, n=1, device='cpu'):
    graph_clone = graph.clone()
    node_as = random.choices(range(graph_clone.num_nodes), k=n)
    node_bs = random.choices(range(graph_clone.num_nodes), k=n)
    graph_clone.edge_index = torch.cat((graph_clone.edge_index, torch.tensor([node_as, node_bs]).to(device)), dim=1)
    graph_clone.edge_index = graph_clone.edge_index[:, graph_clone.edge_index[0, :].argsort()]
    return graph_clone


def train_step_augm(model, data, train_mask, optimizer, device, lambda_entropy=0, augm=0.1):
    ''' Training step: yields average loss over batch. '''
    model.train()
    optimizer.zero_grad()
    data = augment_graph(data, int(data.num_edges*augm), device)
    _, out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    if lambda_entropy:
        loss += lambda_entropy * losses.entropy_regularization(out[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def training_augm(model, data, train_mask, val_mask, optimizer, epochs=100, patience=50, device='cpu', lambda_entropy=0, augm=0.1):
    ''' Training function '''
    rec_df = pd.DataFrame(columns=['train_loss', 'val_acc', 'val_f1'])
    best_score = 0
    no_improvements = 0
    best_model_state = None

    for epoch in range(1, epochs+1):
        train_loss = train_step_augm(model, data, train_mask, optimizer, device, lambda_entropy, augm)
        val_acc, val_f1, _, _ = test(model, data, val_mask)
        rec_df.loc[len(rec_df)] = [train_loss, val_acc, val_f1]
        if val_acc + val_f1 > best_score:
            best_score = val_acc + val_f1
            no_improvements = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            no_improvements += 1
            if no_improvements == patience:  # Early Stopping
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return rec_df


def rewire_graph(graph, p=0.4):
    graph_clone = graph.clone()
    edge_index = graph_clone.edge_index
    nodes_a, nodes_b = edge_index

    # Create mask for edges to rewire
    mask = torch.rand(nodes_a.size(0), device=edge_index.device) < p
    nodes_a_to_keep = nodes_a[~mask]
    nodes_b_to_keep = nodes_b[~mask]
    nodes_a_to_rewire = nodes_a[mask]
    nodes_b_to_rewire = nodes_b[mask]

    # Use graph labels directly on the GPU
    labels = graph_clone.y

    # For each edge to be rewired, pick a new target with the same label
    labels_rewired = labels[nodes_b_to_rewire]
    rewired_nodes_b = torch.empty_like(nodes_b_to_rewire)

    # Efficient vectorized sampling of new nodes
    for lbl in labels_rewired.unique():
        # Get valid nodes with the same label
        mask_lbl = (labels == lbl)
        candidates = torch.arange(labels.size(0), device=edge_index.device)[mask_lbl]

        # Get the indices where we need to rewire
        to_rewire = (labels_rewired == lbl).nonzero(as_tuple=True)[0]

        # Randomly select new target nodes
        rewired_nodes_b[to_rewire] = candidates[torch.randint(0, candidates.size(0), (to_rewire.size(0),), device=edge_index.device)]

    # Combine kept and rewired edges
    new_nodes_a = torch.cat([nodes_a_to_keep, nodes_a_to_rewire])
    new_nodes_b = torch.cat([nodes_b_to_keep, rewired_nodes_b])

    # Sort edges by source node for consistency
    order = new_nodes_a.argsort()
    new_edge_index = torch.stack([new_nodes_a[order], new_nodes_b[order]], dim=0)

    graph_clone.edge_index = new_edge_index
    return graph_clone


def train_step_rew(model, data, train_mask, optimizer, lambda_entropy=0, rew=0.1):
    ''' Training step: yields average loss over batch. '''
    model.train()
    optimizer.zero_grad()
    data = rewire_graph(data, rew)
    _, out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    if lambda_entropy:
        loss += lambda_entropy * losses.entropy_regularization(out[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def training_rew(model, data, train_mask, val_mask, optimizer, epochs=100, patience=50, lambda_entropy=0, rew=0.1):
    ''' Training function '''
    rec_df = pd.DataFrame(columns=['train_loss', 'val_acc', 'val_f1'])
    best_score = 0
    no_improvements = 0
    best_model_state = None

    for epoch in range(1, epochs+1):
        train_loss = train_step_rew(model, data, train_mask, optimizer, lambda_entropy, rew)
        val_acc, val_f1, _, _ = test(model, data, val_mask)
        rec_df.loc[len(rec_df)] = [train_loss, val_acc, val_f1]
        if val_acc + val_f1 > best_score:
            best_score = val_acc + val_f1
            no_improvements = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            no_improvements += 1
            if no_improvements == patience:  # Early Stopping
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return rec_df


def get_anch_pos_neg_idx(graph, train_mask):
    # Use torch.nonzero to get indices of True values in train_mask
    anchors_idx = torch.nonzero(train_mask, as_tuple=True)[0].tolist()
    # Get labels of anchors (assuming graph.y is a tensor)
    anchors_lab = graph.y[train_mask].tolist()

    # Create mapping: label -> list of indices
    labels = set(anchors_lab)
    lab_to_idx = {label: [] for label in labels}
    for idx, lab in zip(anchors_idx, anchors_lab):
        lab_to_idx[lab].append(idx)

    positives_idx = []
    negatives_idx = []
    for idx, label in zip(anchors_idx, anchors_lab):
        # Choose a positive index (ensure it is not the same as idx if possible)
        pos_idx = random.choice(lab_to_idx[label])
        while pos_idx == idx and len(lab_to_idx[label]) > 1:
            pos_idx = random.choice(lab_to_idx[label])
        positives_idx.append(pos_idx)
        # Choose a negative index from a different label
        neg_label = random.choice(list(labels - {label}))
        neg_idx = random.choice(lab_to_idx[neg_label])
        negatives_idx.append(neg_idx)

        # (Optional) Verify assumptions
        assert label == graph.y[pos_idx].item()
        assert idx != pos_idx
        assert label != graph.y[neg_idx].item()
        assert idx != neg_idx

    return anchors_idx, positives_idx, negatives_idx


def train_step_protonet(model, data, train_mask, optimizer):
    model.train()
    optimizer.zero_grad()
    emb, _ = model(data.x, data.edge_index)

    anchors_idx, positives_idx, negatives_idx = get_anch_pos_neg_idx(data, train_mask)
    anchors_emb = emb[anchors_idx]
    positives_emb = emb[positives_idx]
    negatives_emb = emb[negatives_idx]

    loss = F.triplet_margin_loss(anchors_emb, positives_emb, negatives_emb, margin=1)
    loss.backward()
    optimizer.step()

    return loss.item()


def get_prototypes(model, graph, test_mask):
    model.eval()
    prototypes = {}
    with torch.no_grad():
        emb, _ = model(graph.x, graph.edge_index)
        # Use boolean indexing on tensor directly
        test_idx = torch.nonzero(test_mask, as_tuple=True)[0]
        test_emb = emb[test_idx]
        test_lab = graph.y[test_mask].tolist()
        for label in set(test_lab):
            # Boolean mask: graph.y[test_mask] equals label
            label_mask = (graph.y[test_mask] == label)
            prototypes[label] = torch.mean(test_emb[label_mask], dim=0)
    return prototypes


def test_protonet(model, graph, test_mask, prototypes):
    model.eval()
    with torch.no_grad():
        emb, _ = model(graph.x, graph.edge_index)
        test_idx = torch.nonzero(test_mask, as_tuple=True)[0]
        test_emb = emb[test_idx]
        test_lab = graph.y[test_mask].tolist()

        # Vectorize prototype matching:
        # Stack prototypes into a tensor and store their labels in a list.
        proto_labels = sorted(prototypes.keys())
        proto_tensor = torch.stack([prototypes[label] for label in proto_labels])
        # Compute distances: shape (num_test, num_prototypes)
        distances = torch.cdist(test_emb, proto_tensor)
        # Get index of nearest prototype for each test sample
        min_indices = torch.argmin(distances, dim=1)
        y_pred = [proto_labels[idx] for idx in min_indices.tolist()]

    accuracy = sum(1 for t, p in zip(test_lab, y_pred) if t == p) / len(test_lab)
    f1 = f1_score(test_lab, y_pred, average="weighted")
    return accuracy, f1, test_lab, y_pred


def training_protonet(model, data, train_mask, val_mask, optimizer, epochs=100, patience=50):
    rec_df = pd.DataFrame(columns=['train_loss', 'val_acc', 'val_f1'])
    best_score = 0
    no_improvements = 0
    best_model_state = None  # Keep best model state in memory

    for epoch in range(1, epochs+1):
        train_loss = train_step_protonet(model, data, train_mask, optimizer)
        prototypes = get_prototypes(model, data, val_mask)
        val_acc, val_f1, _, _ = test_protonet(model, data, val_mask, prototypes)
        rec_df.loc[len(rec_df)] = [train_loss, val_acc, val_f1]

        if val_acc + val_f1 > best_score:
            best_score = val_acc + val_f1
            no_improvements = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            no_improvements += 1
            if no_improvements == patience:  # Early Stopping
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return rec_df


def train_step_siamesenet_PW(model, data, train_mask, optimizer, lambda_entropy=0, augm=False, device='cpu'):
    ''' Training step: yields average loss over batch. '''
    model.train()
    optimizer.zero_grad()
    if augm: data = rewire_graph(data, 0.1)
    emb, out = model(data.x, data.edge_index)
    anchors_idx, positives_idx, negatives_idx = get_anch_pos_neg_idx(data, train_mask)
    anchors_emb = emb[anchors_idx]
    positives_emb = emb[positives_idx]
    negatives_emb = emb[negatives_idx]
    CE_loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    if lambda_entropy != 0.:
        CE_loss += lambda_entropy * losses.entropy_regularization(out[train_mask])
    if np.random.rand() < 0.5:
        PW_loss = losses.PWLoss()(anchors_emb, positives_emb, torch.ones(anchors_emb.size(0)).to(device))
    else:
        PW_loss = losses.PWLoss()(anchors_emb, negatives_emb, torch.zeros(anchors_emb.size(0)).to(device))
    loss = CE_loss + PW_loss
    loss.backward()
    optimizer.step()
    return loss.item()


def train_step_siamesenet_TPL(model, data, train_mask, optimizer, lambda_entropy=0, augm=False):
    ''' Training step: yields average loss over batch. '''
    model.train()
    optimizer.zero_grad()
    if augm: data = rewire_graph(data, 0.1)
    emb, out = model(data.x, data.edge_index)
    anchors_idx, positives_idx, negatives_idx = get_anch_pos_neg_idx(data, train_mask)
    anchors_emb = emb[anchors_idx]
    positives_emb = emb[positives_idx]
    negatives_emb = emb[negatives_idx]
    CE_loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    if lambda_entropy != 0.:
        CE_loss += lambda_entropy * losses.entropy_regularization(out[train_mask])
    TPL_loss = F.triplet_margin_loss(anchors_emb, positives_emb, negatives_emb, margin=1)
    loss = CE_loss + TPL_loss
    loss.backward()
    optimizer.step()
    return loss.item()


def training_siamesenet(model, data, train_mask, val_mask, optimizer, epochs=100, patience=50, CL_Loss='TPL', lambda_entropy=0, device='cpu'):
    ''' Training function '''
    rec_df = pd.DataFrame(columns=['train_loss', 'val_acc', 'val_f1'])
    best_score = 0
    no_improvements = 0
    best_model_state = None

    for epoch in range(1, epochs+1):
        train_loss = train_step(model, data, train_mask, optimizer, lambda_entropy)
        if CL_Loss == 'PW':
            train_loss = train_step_siamesenet_PW(model, data, train_mask, optimizer, lambda_entropy, device=device)
        elif CL_Loss == 'PW_A':
            train_loss = train_step_siamesenet_PW(model, data, train_mask, optimizer, lambda_entropy, augm=True, device=device)
        elif CL_Loss == 'TPL':
            train_loss = train_step_siamesenet_TPL(model, data, train_mask, optimizer, lambda_entropy)
        elif CL_Loss == 'TPL_A':
            train_loss = train_step_siamesenet_TPL(model, data, train_mask, optimizer, lambda_entropy, augm=True)
        else:
            raise ValueError(f'Invalid CL_Loss: {CL_Loss} not in [PW, PW_A, TPL, TPL_A]')

        val_acc, val_f1, _, _ = test(model, data, val_mask)
        rec_df.loc[len(rec_df)] = [train_loss, val_acc, val_f1]
        if val_acc + val_f1 > best_score:
            best_score = val_acc + val_f1
            no_improvements = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            no_improvements += 1
            if no_improvements == patience:  # Early Stopping
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return rec_df


def pretrain_graphcl(model, data, train_mask, optimizer, epochs=100):
    """ Pre-training function for Graph Contrastive Learning (GraphCL). """
    model.train()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        with torch.no_grad():
            data_1 = rewire_graph(data, p=0.2)
        emb1, _ = model(data.x, data.edge_index)
        emb2, _ = model(data_1.x, data_1.edge_index)
        loss = losses.nt_xent_loss(emb1[train_mask], emb2[train_mask], tau=0.5)
        loss.backward()
        optimizer.step()
