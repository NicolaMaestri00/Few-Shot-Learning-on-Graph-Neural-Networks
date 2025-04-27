"""    Utility functions    """

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










from sklearn.metrics import f1_score
import pandas as pd
import copy


def train_step_graph(model, train_loader, optimizer, lambda_entropy=0):
    ''' Training step: returns average loss over batch '''
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        _, out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        if lambda_entropy != 0:
            loss += lambda_entropy * losses.entropy_regularization(out)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def test_graph(model, test_loader):
    ''' Evaluation: returns accuracy and F1 score '''
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in test_loader:
            _, out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)        # Predicted class
            y_true.extend(data.y.tolist())  # Append true labels
            y_pred.extend(pred.tolist())    # Append predicted labels
    micro_acc = sum([1 for t, p in zip(y_true, y_pred) if t == p]) / len(y_true)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    return micro_acc, weighted_f1, y_true, y_pred


def training_graph(model, train_loader, val_loader, optimizer, epochs=250, patience=50, lambda_entropy=0):
    ''' Training function '''
    rec_df = pd.DataFrame(columns=['train_loss', 'val_acc', 'val_f1'])
    best_score = 0
    no_improvements = 0
    best_model_state = None

    for epoch in range(epochs):
        train_loss = train_step_graph(model, train_loader, optimizer, lambda_entropy)
        val_acc, val_f1, _, _ = test_graph(model, val_loader)
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




def gc_augment_graph(graph, n=1, device='cpu'):
    graph_clone = graph.clone()
    node_as = random.choices(range(graph_clone.num_nodes), k=n)
    node_bs = random.choices(range(graph_clone.num_nodes), k=n)
    graph_clone.edge_index = torch.cat((graph_clone.edge_index, torch.tensor([node_as, node_bs]).to(device)), dim=1)
    graph_clone.edge_index = graph_clone.edge_index[:, graph_clone.edge_index[0, :].argsort()]
    return graph_clone

from torch_geometric.data import Batch


def gc_train_step_augm(model, train_loader, optimizer, lambda_entropy=0, device='cpu'):
    ''' Training step: returns average loss over batch '''
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        graphs = data.to_data_list()
        augmented_graphs = [gc_augment_graph(graph, int(graph.num_edges * 0.1), device) for graph in graphs]
        new_data = Batch.from_data_list(augmented_graphs)
        optimizer.zero_grad()
        _, out = model(new_data.x, new_data.edge_index, new_data.batch)
        loss = F.cross_entropy(out, new_data.y)
        if lambda_entropy != 0:
            loss += lambda_entropy * losses.entropy_regularization(out)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def gc_training_augm(model, train_loader, val_loader, optimizer, epochs=250, patience=50, lambda_entropy=0, device='cpu'):
    ''' Training function '''
    rec_df = pd.DataFrame(columns=['train_loss', 'val_acc', 'val_f1'])
    best_score = 0
    no_improvements = 0
    best_model_state = None

    for epoch in range(epochs):
        train_loss = gc_train_step_augm(model, train_loader, optimizer, lambda_entropy, device)
        val_acc, val_f1, _, _ = test_graph(model, val_loader)
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



import torch
import numpy as np
import random
import copy
import torch.nn.functional as F  # for F.cross_entropy
import pandas as pd
from torch_geometric.data import Batch

def gc_rewire_graph(graph, p=0.4):
    # Work on a copy so that the original graph remains unchanged.
    new_graph = copy.deepcopy(graph)
    device = new_graph.edge_index.device

    # Get source and target nodes from edge_index.
    src, dst = new_graph.edge_index[0], new_graph.edge_index[1]
    # Create canonical (ordered) edges for undirected uniqueness.
    sorted_src = torch.min(src, dst)
    sorted_dst = torch.max(src, dst)
    canonical_edges = torch.stack((sorted_src, sorted_dst), dim=0)  # shape (2, num_edges)

    # Try using torch.unique with return_index; fall back if not supported.
    try:
        unique_edges, unique_indices, inverse, counts = torch.unique(
            canonical_edges,
            dim=1,
            return_index=True,
            return_inverse=True,
            return_counts=True
        )
    except TypeError:
        # Fallback for older PyTorch versions:
        unique_edges, inverse, counts = torch.unique(
            canonical_edges,
            dim=1,
            return_inverse=True,
            return_counts=True
        )
        num_unique = unique_edges.size(1)
        unique_indices = []
        for i in range(num_unique):
            idxs = (inverse == i).nonzero(as_tuple=True)[0]
            unique_indices.append(idxs[0].item())

    num_unique = unique_edges.size(1)

    # Build a list of unique undirected edges using vectorized conversion.
    edges_list = list(zip(unique_edges[0].tolist(), unique_edges[1].tolist()))

    # We'll use a set for fast duplicate checking (store edges as canonical tuples).
    new_edges_set = set()
    new_edges = []  # This will store both directions of each edge.

    if new_graph.edge_attr is not None:
        # For each unique edge, pick an attribute (using argmax on the edge attribute).
        unique_attrs = [int(torch.argmax(new_graph.edge_attr[idx]).item()) for idx in unique_indices]
        # Group unique edges by their attribute.
        label_groups = {}
        for i, label in enumerate(unique_attrs):
            label_groups.setdefault(label, []).append(i)

        # For each label group, decide per edge whether to rewire (with probability p).
        old_group = []       # Indices that remain unchanged.
        rewired_group = {}   # For each label, indices chosen for rewiring.
        for label, indices in label_groups.items():
            indices = np.array(indices)
            mask = np.random.rand(len(indices)) < p  # True means rewire.
            old_idxs = indices[~mask]
            rewired_idxs = indices[mask]
            old_group.extend(old_idxs.tolist())
            rewired_group[label] = rewired_idxs.tolist()

        # Add the non-rewired edges (both directions).
        for i in old_group:
            a, b = edges_list[i]
            key = (min(a, b), max(a, b))
            if key not in new_edges_set:
                new_edges_set.add(key)
                new_edges.append((a, b))
                new_edges.append((b, a))

        # For rewired edges, shuffle endpoints within each label.
        for label, idx_list in rewired_group.items():
            if not idx_list:
                continue
            nodes_a = [edges_list[i][0] for i in idx_list]
            nodes_b = [edges_list[i][1] for i in idx_list]
            random.shuffle(nodes_a)
            random.shuffle(nodes_b)
            for a, b in zip(nodes_a, nodes_b):
                key = (min(a, b), max(a, b))
                if key not in new_edges_set:
                    new_edges_set.add(key)
                    new_edges.append((a, b))
                    new_edges.append((b, a))
    else:
        # If no edge attributes, treat all unique edges uniformly.
        mask = np.random.rand(len(edges_list)) < p
        rewired_a, rewired_b = [], []
        old_indices = []
        for i, m in enumerate(mask):
            if m:
                rewired_a.append(edges_list[i][0])
                rewired_b.append(edges_list[i][1])
            else:
                old_indices.append(i)
        for i in old_indices:
            a, b = edges_list[i]
            key = (min(a, b), max(a, b))
            if key not in new_edges_set:
                new_edges_set.add(key)
                new_edges.append((a, b))
                new_edges.append((b, a))
        random.shuffle(rewired_a)
        random.shuffle(rewired_b)
        for a, b in zip(rewired_a, rewired_b):
            key = (min(a, b), max(a, b))
            if key not in new_edges_set:
                new_edges_set.add(key)
                new_edges.append((a, b))
                new_edges.append((b, a))

    # Convert new_edges to a tensor and sort by the source node.
    edge_index = torch.tensor(new_edges, device=device).t().contiguous()
    sorted_idx = edge_index[0].argsort()
    new_graph.edge_index = edge_index[:, sorted_idx]
    return new_graph


from torch_geometric.data import Batch


def gc_train_step_rew(model, train_loader, optimizer, lambda_entropy=0):
    ''' Training step: returns average loss over batch '''
    model.train()
    total_loss = 0
    for data in train_loader:
        graphs = data.to_data_list()
        rewired_graphs = [gc_rewire_graph(graph, p=0.1) for graph in graphs]
        new_data = Batch.from_data_list(rewired_graphs)
        optimizer.zero_grad()
        _, out = model(new_data.x, new_data.edge_index, new_data.batch)
        loss = F.cross_entropy(out, new_data.y)
        if lambda_entropy != 0:
            loss += lambda_entropy * losses.entropy_regularization(out)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def gc_training_rew(model, train_loader, val_loader, optimizer, epochs=250, patience=50, lambda_entropy=0):
    ''' Training function '''
    rec_df = pd.DataFrame(columns=['train_loss', 'val_acc', 'val_f1'])
    best_score = 0
    no_improvements = 0
    best_model_state = None

    for epoch in range(epochs):
        train_loss = gc_train_step_rew(model, train_loader, optimizer, lambda_entropy)
        val_acc, val_f1, _, _ = test_graph(model, val_loader)
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


from sklearn.metrics import f1_score
from torch_geometric.data import Batch
import pandas as pd

def gc_train_step_siamesenet_PW(model, triplet_train_loader, optimizer, lambda_entropy=0, device='cpu', augm=False):
    ''' Train step for Siamese Net with pairwise loss '''
    model.train()
    total_loss = 0
    for anchor, positive, negative in triplet_train_loader:
        if augm:
            graphs_pos = positive.to_data_list()
            graphs_neg = negative.to_data_list()
            rewired_graphs_pos = [gc_rewire_graph(graph, p=0.1) for graph in graphs_pos]
            rewired_graphs_neg = [gc_rewire_graph(graph, p=0.1) for graph in graphs_neg]
            positive = Batch.from_data_list(rewired_graphs_pos)
            negative = Batch.from_data_list(rewired_graphs_neg)
        optimizer.zero_grad()
        anchor_emb, out = model(anchor.x, anchor.edge_index, anchor.batch)
        positive_emb, _ = model(positive.x, positive.edge_index, positive.batch)
        negative_emb, _ = model(negative.x, negative.edge_index, negative.batch)
        CE_loss = F.cross_entropy(out, anchor.y)
        if lambda_entropy != 0.:
            CE_loss += lambda_entropy * losses.entropy_regularization(out)
        if np.random.rand() < 0.5:
            PW_loss = losses.PWLoss()(anchor_emb, positive_emb, torch.ones(anchor_emb.size(0)).to(device))
        else:
            PW_loss = losses.PWLoss()(anchor_emb, negative_emb, torch.zeros(anchor_emb.size(0)).to(device))
        loss = CE_loss + PW_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(triplet_train_loader)
    return avg_loss


def gc_train_step_siamesenet_TPL(model, triplet_train_loader, optimizer, lambda_entropy=0., augm=False):
    ''' Train step for Siamese Net with triplet loss '''
    model.train()
    total_loss = 0
    for anchor, positive, negative in triplet_train_loader:
        if augm:
            graphs_pos = positive.to_data_list()
            graphs_neg = negative.to_data_list()
            rewired_graphs_pos = [gc_rewire_graph(graph, p=0.1) for graph in graphs_pos]
            rewired_graphs_neg = [gc_rewire_graph(graph, p=0.1) for graph in graphs_neg]
            positive = Batch.from_data_list(rewired_graphs_pos)
            negative = Batch.from_data_list(rewired_graphs_neg)
        optimizer.zero_grad()
        anchor_emb, out = model(anchor.x, anchor.edge_index, anchor.batch)
        positive_emb, _ = model(positive.x, positive.edge_index, positive.batch)
        negative_emb, _ = model(negative.x, negative.edge_index, negative.batch)
        CE_loss = F.cross_entropy(out, anchor.y)
        if lambda_entropy != 0.:
            CE_loss += lambda_entropy * losses.entropy_regularization(out)
        TPL_loss = F.triplet_margin_loss(anchor_emb, positive_emb, negative_emb, margin=0.2)
        loss = CE_loss + TPL_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(triplet_train_loader)
    return avg_loss


def gc_training_siamesenet(model, train_loader, val_loader, optimizer, epochs=250, patience=50, CL_Loss='TPL', lambda_entropy=0., device='cpu'):
    ''' Training Siamese Network '''
    rec_df = pd.DataFrame(columns=['train_loss', 'val_acc', 'val_f1'])
    best_score = 0
    no_improvements = 0
    best_model_state = None
    for epoch in range(epochs):
        if CL_Loss == 'PW':
            train_loss = gc_train_step_siamesenet_PW(model, train_loader, optimizer, lambda_entropy, device)
        elif CL_Loss == 'TPL':
            train_loss = gc_train_step_siamesenet_TPL(model, train_loader, optimizer, lambda_entropy)
        elif CL_Loss == 'PW_A':
            train_loss = gc_train_step_siamesenet_PW(model, train_loader, optimizer, lambda_entropy, device, augm=True)
        elif CL_Loss == 'TPL_A':
            train_loss = gc_train_step_siamesenet_TPL(model, train_loader, optimizer, lambda_entropy, augm=True)
        else:
            raise ValueError(f'Invalid CL_Loss: {CL_Loss} not in [PW, TPL, PW_A, TPL_A]')

        val_acc, val_f1, _, _ = test_graph(model, val_loader)
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



def gc_pretrain_graphcl(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for data in train_loader:
        with torch.no_grad():
            graphs = data.to_data_list()
            rewired_graphs_1 = [gc_rewire_graph(graph, p=0.2) for graph in graphs]
            data_1 = Batch.from_data_list(rewired_graphs_1)

        _, emb1 = model(data.x, data.edge_index, data.batch)
        _, emb2 = model(data_1.x, data_1.edge_index, data_1.batch)
        loss = losses.nt_xent_loss(emb1, emb2, tau=0.5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def gc_pretraining(model, train_loader, optimizer, epochs=100):
    for epoch in range(epochs):
        loss = gc_pretrain_graphcl(model, train_loader, optimizer)











from sklearn.metrics import f1_score
import pandas as pd

def gc_get_prototypes(model, train_dataset):
    model.eval()
    prototypes = {}
    with torch.no_grad():
        for data in train_dataset:
            emb, _ = model(data.x, data.edge_index, data.batch)
            label = data.y.item()
            if label not in prototypes:
                prototypes[label] = [emb]
            else:
                prototypes[label].append(emb)
    # Now compute the mean for each label
    for label in prototypes:
        prototypes[label] = torch.mean(torch.stack(prototypes[label], dim=0), dim=0)
    return prototypes


def gc_test_protonet(model, prototypes, test_dataset):
    model.eval()
    y_true, y_pred = [], []
    # Convert prototypes to 1D tensors and stack them.
    proto_labels = list(prototypes.keys())
    proto_list = [prototypes[label].squeeze() for label in proto_labels]
    proto_tensor = torch.stack(proto_list)

    with torch.no_grad():
        for data in test_dataset:
            test_emb, _ = model(data.x, data.edge_index, data.batch)
            test_emb = test_emb.squeeze()  # Ensure test embedding is 1D
            # Compute distances using broadcasting
            distances = torch.norm(proto_tensor - test_emb.unsqueeze(0), dim=1)
            min_idx = torch.argmin(distances).item()
            y_true.append(data.y.item())
            y_pred.append(proto_labels[min_idx])

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return accuracy, f1, y_true, y_pred



def gc_train_step_protonet(model, triplet_train_loader, optimizer):
    ''' Training step for ProtoNet  '''
    model.train()
    total_loss = 0
    for anchor, positive, negative in triplet_train_loader:
        optimizer.zero_grad()
        anchor_emb, _ = model(anchor.x, anchor.edge_index, anchor.batch)
        positive_emb, _ = model(positive.x, positive.edge_index, positive.batch)
        negative_emb, _ = model(negative.x, negative.edge_index, negative.batch)
        loss = F.triplet_margin_loss(anchor_emb, positive_emb, negative_emb, margin=1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(triplet_train_loader)
    return avg_loss


def gc_training_protonet(model, train_dataset, train_loader, val_dataset, optimizer, epochs=250, patience=30):
    ''' Training ProtoNet '''
    rec_df = pd.DataFrame(columns=['train_loss', 'val_acc', 'val_f1'])
    best_score = 0
    no_improvements = 0
    best_model_state = None
    for epoch in range(epochs):
        train_loss = gc_train_step_protonet(model, train_loader, optimizer)
        prototypes = gc_get_prototypes(model, train_dataset)
        val_acc, val_f1, _, _ = gc_test_protonet(model, prototypes, val_dataset)
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



