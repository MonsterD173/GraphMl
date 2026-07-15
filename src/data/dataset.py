import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def get_data(name='Cora', root='data/Cora', device=None):
    """
    Load dataset (currently supports Cora).
    Returns (data, num_features, num_classes)
    Compatible with demo.ipynb
    """
    if name.lower() != 'cora':
        raise ValueError("Only Cora supported in this baseline version")
    
    dataset = Planetoid(
        root=root,
        name='Cora',
        transform=T.NormalizeFeatures()
    )
    data = dataset[0]
    
    if device:
        data = data.to(device)
    
    return data, dataset.num_features, dataset.num_classes


def load_cora_data(root='data/Cora', device=None):
    """Legacy wrapper for backward compatibility."""
    data, num_feat, num_cls = get_data('Cora', root, device)
    return data, None


def print_statistics(data, num_features, num_classes, name='Cora'):
    """Print dataset statistics. Compatible with demo.ipynb"""
    print("=" * 60)
    print(f"{name.upper()} CITATION NETWORK - DATASET STATISTICS")
    print("=" * 60)
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {num_features}")
    print(f"Number of classes: {num_classes}")
    print(f"Is undirected: {data.is_undirected()}")
    print(f"Average degree: {data.num_edges / data.num_nodes:.2f}")
    
    print("\nData Split:")
    print(f"  Train: {data.train_mask.sum().item()} ({data.train_mask.sum().item()/data.num_nodes*100:.1f}%)")
    print(f"  Val:   {data.val_mask.sum().item()}")
    print(f"  Test:  {data.test_mask.sum().item()}")
    
    print("\nLabel distribution:")
    unique, counts = torch.unique(data.y, return_counts=True)
    for cls, cnt in zip(unique.tolist(), counts.tolist()):
        print(f"  Class {cls}: {cnt} papers")
    
    print("=" * 60)


def print_dataset_stats(data, dataset):
    """Legacy wrapper."""
    print_statistics(data, data.num_features, dataset.num_classes if hasattr(dataset, 'num_classes') else 'N/A')


if __name__ == "__main__":
    data, dataset = load_cora_data()
    print_dataset_stats(data, dataset)
