import torch
import numpy as np
import random
from sklearn.metrics import f1_score
import pandas as pd
import copy

import torch.nn.functional as F
import losses


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def save_df(dfs=[], file_names=[]):
    for df, file_name in zip(dfs, file_names):
        df.to_csv(f'{file_name}.csv', index=False)


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


def training(model, data, train_mask, val_mask, optimizer, epochs=100, patience=50, device='cpu', lambda_entropy=0):
    ''' Training function '''
    rec_df = pd.DataFrame(columns=['train_loss', 'val_acc', 'val_f1'])
    best_score = 0
    no_improvements = 0
    best_model_state = None

    model.to(device)
    data = data.to(device)

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
