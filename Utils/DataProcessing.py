import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from itertools import combinations
import sys
sys.path.append('/')

def load_dataset(base_path, file_name, low_memory=True):
    return np.array(pd.read_csv(f"{base_path}{file_name}", low_memory=low_memory))

def data_processing(ds_name):
    datasets_info = {
        'CICIDS2017': {
            'path': 'Data/CICIDS2017/',
            'low_memory': True
        },
        'CICIDS2018': {
            'path': 'Data/CICIDS2018/',
            'low_memory': False
        }
    }

    if ds_name not in datasets_info:
        raise ValueError("Invalid scenario name.")

    info = datasets_info[ds_name]
    base_path = info['path']
    low_memory = info['low_memory']

    # Load datasets
    x_train = load_dataset(base_path, 'x_train.csv', low_memory)
    x_valid = load_dataset(base_path, 'x_valid.csv', low_memory)
    x_test = load_dataset(base_path, 'x_test.csv', low_memory)
    x_drift = load_dataset(base_path, 'x_drift.csv', low_memory)
    x_non_drift = load_dataset(base_path, 'x_non_drift.csv', low_memory)
    y_train = load_dataset(base_path, 'y_train.csv', low_memory).ravel()
    y_valid = load_dataset(base_path, 'y_valid.csv', low_memory).ravel()
    y_test = load_dataset(base_path, 'y_test.csv', low_memory).ravel()
    y_drift = load_dataset(base_path, 'y_drift.csv', low_memory).ravel()
    y_non_drift = load_dataset(base_path, 'y_non_drift.csv', low_memory).ravel()
    print(f'{ds_name} loaded')
    return x_train, x_valid, x_test, x_drift, x_non_drift, y_train, y_valid, y_test, y_drift, y_non_drift


def Rasd_loader(x_data, y_data, batch_size=1024):
    x_data = torch.tensor(x_data, dtype=torch.float)
    y_data = torch.tensor(y_data, dtype=torch.double)
    loader = train_dataset = DataLoader(
    dataset=TensorDataset(x_data, y_data),
    batch_size=1024,
    shuffle=True
    , num_workers=4)
    return loader

def balance_dataset(x_train, y_train):
    if not isinstance(x_train, torch.Tensor):
        x_train = torch.tensor(x_train, dtype=torch.float32)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.long)
    unique_labels, counts = torch.unique(y_train, return_counts=True)
    min_count = torch.min(counts).item()
    min_label = unique_labels[torch.argmin(counts)].item()
    balanced_x = []
    balanced_y = []
    
    for label in unique_labels:
        print(label)
        indices = torch.where(y_train == label)[0]
        if label != min_label:
            indices = indices[torch.randperm(len(indices))[:min_count]]
        else:  
            indices = indices[:min_count]
        balanced_x.append(x_train[indices])
        balanced_y.append(y_train[indices])
    balanced_x = torch.cat(balanced_x, dim=0)
    balanced_y = torch.cat(balanced_y, dim=0)
    sorted_indices = torch.argsort(balanced_y)
    balanced_x = balanced_x[sorted_indices]
    balanced_y = balanced_y[sorted_indices]

    return balanced_x, balanced_y


def create_pairs(x_train, y_train):
    device = x_train.device
    pairs = []
    labels = []
    class_indices = [torch.where(y_train == i)[0] for i in torch.unique(y_train)]
    for indices in class_indices:
        for i, j in combinations(indices, 2):
            pairs.append(torch.stack([x_train[i], x_train[j]], dim=0))
            labels.append(torch.tensor(1, device=device))  # Similar
    for i, class_i_indices in enumerate(class_indices):
        for j, class_j_indices in enumerate(class_indices):
            if i != j:
                min_length = min(len(class_i_indices), len(class_j_indices))
                sample_indices_i = torch.randperm(len(class_i_indices))[:min_length]
                sample_indices_j = torch.randperm(len(class_j_indices))[:min_length]
                for k in range(min_length):
                    pairs.append(torch.stack([x_train[class_i_indices[sample_indices_i[k]]], x_train[class_j_indices[sample_indices_j[k]]]], dim=0))
                    labels.append(torch.tensor(0, device=device))  # Dissimilar

    pairs = torch.stack(pairs, dim=0)
    labels = torch.stack(labels, dim=0)
    return pairs, labels


class ContrastiveDataset(Dataset):
    def __init__(self, X, y, similar_samples_ratio):
        super().__init__()
        self.X = X
        self.y = y
        self.similar_samples_ratio = similar_samples_ratio  
        self.unique_labels = np.unique(y)
        n_similar = int(similar_samples_ratio * len(y))
        n_dissimilar = len(y) - n_similar
        self.label_to_similar_indices = {label: np.where(y == label)[0] for label in self.unique_labels}
        self.label_to_dissimilar_indices = {label: np.where(y != label)[0] for label in self.unique_labels}

        self.label_to_similar_indices = {label: np.random.choice(indices, n_similar, replace=True) 
                                         for label, indices in self.label_to_similar_indices.items()}
        self.label_to_dissimilar_indices = {label: np.random.choice(indices, n_dissimilar, replace=True) 
                                            for label, indices in self.label_to_dissimilar_indices.items()}

    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        x1 = self.X[index]
        y1 = self.y[index].item()  
        should_get_similar = np.random.rand() < self.similar_samples_ratio
        if should_get_similar:
            index2 = np.random.choice(self.label_to_similar_indices[y1])
        else:
            index2 = np.random.choice(self.label_to_dissimilar_indices[y1])
        x2 = self.X[index2]
        y2 = self.y[index2].item()  
        
        return x1, y1, x2, y2

def CADE_dataloader(X, y, batch_size, similar_samples_ratio, shuffle=True):
    dataset = ContrastiveDataset(X, y, similar_samples_ratio)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=4)
    return dataloader
