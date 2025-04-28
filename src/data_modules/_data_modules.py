"""    Data Module    """

import random

import torch
import torch_geometric
from torch.utils.data import Dataset
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures, Compose, RandomNodeSplit


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


def nc_get_data(dataset_name: str, samples_per_class: list, device: torch.device) -> tuple:
        """ Node Classification Dataset Loader """

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


class TripletDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = [data.y.item() for data in dataset]
        self.classes = set(self.labels)
        self.label_to_indices = {label: [] for label in self.classes}
        for i, label in enumerate(self.labels):
            self.label_to_indices[label].append(i)

    def __getitem__(self, index):
        anchor_graph = self.dataset[index]
        anchor_label = anchor_graph.y.item()
        # Select a positive example (same label as anchor)
        positive_index = random.choice(self.label_to_indices[anchor_label])
        positive_graph = self.dataset[positive_index]
        # Select a negative example (different label from anchor)
        negative_label = random.choice(list(self.classes - {anchor_label}))
        negative_index = random.choice(self.label_to_indices[negative_label])
        negative_graph = self.dataset[negative_index]

        return anchor_graph, positive_graph, negative_graph

    def __len__(self):
        return len(self.dataset)


def few_shot_sampler(dataset, needed_per_class):
    ''' Few-shot sampler: yields n_shots per class '''
    labels = [int(graph.y.item()) for graph in dataset]
    classes = set(labels)
    sampled_indices = []
    for cls in classes:
        cls_indices = [i for i, label in enumerate(labels) if label == cls]
        required = needed_per_class.get(cls, 0)
        if required > 0:
            sampled = random.sample(cls_indices, min(len(cls_indices), required))
            sampled_indices.extend(sampled)
    sampled_subset = [dataset[i] for i in sampled_indices]
    remaining_dataset = [dataset[i] for i in range(len(dataset)) if i not in sampled_indices]
    return sampled_subset, remaining_dataset


def gc_get_data(dataset_name: str, samples_per_class: list, device: torch.device) -> tuple:
    """ Graph Classification Dataset Loader """

    dataset = TUDataset(root='data/TUDataset', name=dataset_name)
    num_classes = dataset.num_classes

    shots = []
    train_dataset = []
    test_dataset = [graph for graph in dataset]

    for n_shots in samples_per_class:
        current_counts = {label: 0 for label in range(num_classes)}
        for graph in train_dataset:
            current_counts[int(graph.y)] += 1
        needed_per_class = {label: n_shots - current_counts[label] for label in current_counts}
        add_train_dataset, test_dataset = few_shot_sampler(test_dataset, needed_per_class)
        train_dataset.extend(add_train_dataset)
        train_dataset_dev = [graph.to(device) for graph in train_dataset]
        triplet_train_dataset = TripletDataset(train_dataset_dev)
        train_batch_size = 4 if n_shots <= 20 else 16
        train_loader = DataLoader(train_dataset_dev, batch_size=train_batch_size, shuffle=True)
        triplet_train_loader = DataLoader(triplet_train_dataset, batch_size=train_batch_size, shuffle=True)
        shots.append((n_shots,
                    train_loader,
                    triplet_train_loader,
                    train_dataset_dev))

    # Create a dictionary with the numer of classes per test
    test_dict = {label: 500 for label in range(num_classes)}
    test_dataset, val_dataset = few_shot_sampler(test_dataset, test_dict)
    test_dataset = [graph.to(device) for graph in test_dataset]
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return dataset, shots, test_dataset, test_loader
