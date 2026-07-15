"""MLP baseline — uses node features only, ignores citation edges.

This is the "no-graph" reference: if a GNN beats the MLP, the improvement is
evidence that citation structure carries useful signal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MLP(nn.Module):
    """
    Simple 2-layer MLP for node classification (feature-only baseline).
    
    Args:
        in_channels: Number of input features per node
        hidden_channels: Hidden dimension
        out_channels: Number of classes
        dropout: Dropout probability (default 0.5)
    """
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        out_channels: int, 
        dropout: float = 0.5
    ):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
        Returns:
            log_softmax probabilities [num_nodes, out_channels]
        """
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"in={self.lin1.in_features}, "
                f"hidden={self.lin1.out_features}, "
                f"out={self.lin2.out_features}, "
                f"dropout={self.dropout})")
