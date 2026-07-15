"""Two-layer Graph Convolutional Network (Kipf & Welling, 2017).

    H1 = ReLU( A_hat X W0 )
    Z  = A_hat H1 W1

where A_hat is the symmetrically normalised adjacency with self-loops.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Optional


class GCN(nn.Module):
    """
    2-layer Graph Convolutional Network (Kipf & Welling, 2017).
    
    Args:
        in_channels: Number of input features
        hidden_channels: Hidden dimension
        out_channels: Number of classes
        dropout: Dropout probability after first layer
    """
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        out_channels: int, 
        dropout: float = 0.5
    ):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with message passing.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
        Returns:
            log_softmax probabilities [num_nodes, out_channels]
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"in={self.conv1.in_channels}, "
                f"hidden={self.conv1.out_channels}, "
                f"out={self.conv2.out_channels}, "
                f"dropout={self.dropout})")
