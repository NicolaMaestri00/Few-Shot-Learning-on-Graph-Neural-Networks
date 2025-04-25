"""
Data Module - Handles loading and preprocessing of graph datasets
for node classification tasks.
"""

import torch
import torch_geometric
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.transforms import NormalizeFeatures, Compose, RandomNodeSplit


def get_data(dataset_name: str, samples_per_class: list, device: torch.device) -> tuple:
        """Get graph"""
        if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
            dataset = Planetoid(root='data/Planetoid', name=dataset_name, transform=NormalizeFeatures())
            data = dataset[0]
        elif dataset_name in ['CS', 'Physics']:
            transform = Compose([
                NormalizeFeatures(),
                RandomNodeSplit(split='random', num_train_per_class=20, num_val=500, num_test=1000)
            ])
            dataset = Coauthor(root='data/Coauthor', name=dataset_name, transform=transform)
            data = dataset[0]
        elif dataset_name == 'Amazon Computers':
            transform = Compose([
                NormalizeFeatures(),
                RandomNodeSplit(split='random', num_train_per_class=20, num_val=500, num_test=1000)
            ])
            dataset = Amazon(root='data/Amazon', name='Computers', transform=transform)
            data = dataset[0]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets are: Cora, CiteSeer, PubMed, CS, Physics, Amazon Computers.")
        data = data.to(device)
        shots = []
        for n_per_class in samples_per_class:
            train_mask = extract_training_mask(data, n_per_class, device)
            shots.append((n_per_class, train_mask))
        
        return data, shots


def extract_training_mask(data: torch_geometric.data.Data, n_per_class: int, device: torch.device) -> torch.Tensor:
    """
    Training mask extractor:
    Creates a new training mask for node classification with a fixed number of samples
    """
    new_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)  # Create directly on device
    num_classes = int(data.y.max().item()) + 1
    
    for c in range(num_classes):
        class_indices = (data.y == c).nonzero(as_tuple=False).view(-1)
        # 1. Use up to the nodes already in the train_mask
        default_train_indices = class_indices[data.train_mask[class_indices]]
        n_default = min(20, n_per_class)
        selected = default_train_indices[:n_default]
        
        # 2. If additional nodes are needed, add nodes that are not in val or test
        if selected.numel() < n_per_class:
            # Candidate nodes: all nodes of class c not in val and not in test
            candidate_mask = ~(data.val_mask | data.test_mask)
            candidate_indices = class_indices[candidate_mask[class_indices]]
            # Exclude any that have already been selected
            already_selected = set(selected.tolist())
            additional_candidates = [idx for idx in candidate_indices.tolist() if idx not in already_selected]
            # Create tensor on the same device as selected
            additional_candidates = torch.tensor(additional_candidates, dtype=torch.long, device=device)
            
            # Number of additional nodes needed
            num_to_add = n_per_class - selected.numel()
            if additional_candidates.numel() > 0:
                selected = torch.cat([selected, additional_candidates[:num_to_add]])
                
        # Update the new training mask for these indices
        new_train_mask[selected] = True
        
    return new_train_mask  # Already on the right device