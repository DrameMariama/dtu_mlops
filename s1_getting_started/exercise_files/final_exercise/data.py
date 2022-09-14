import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def mnist():
    # exchange with the corrupted mnist dataset
    train_images = []
    train_labels = []
    for i in range(5):
        train = np.load(f"../../../data/corruptmnist/train_{i}.npz")
        images = train['images']
        labels = train["labels"]
        train_images.append(images)
        train_labels.append(labels)
    test =  np.load(f"../../../data/corruptmnist/test.npz")
    test_images = test["images"]
    test_labels = test["labels"]
    
    train_images = torch.from_numpy(np.vstack(train_images))
    train_labels = torch.from_numpy(np.hstack(train_labels))
    test_images = torch.from_numpy(np.array(test_images))
    test_labels = torch.from_numpy(np.array(test_labels))
    train = TensorDataset(train_images, train_labels)
    test = TensorDataset(test_images, test_labels)
    
    return train, test
mnist()