import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from typing import Type, Union, List
import numpy as np

    
def deduplicate_interaction_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    matrix_deduped_rows = matrix.groupby(matrix.index).mean()
    matrix_deduped = matrix_deduped_rows.T.groupby(level=0).mean().T
    return matrix_deduped
    
class LaplacianRegularizationLoss(nn.Module):
    def __init__(self, normalize_features: bool = True, lambda_reg: float = 1.0, cache_laplacian: bool = True):
        super().__init__()
        self.normalize_features = normalize_features
        self.lambda_reg = lambda_reg
        self.cache_laplacian = cache_laplacian
        self.laplacian_cache = {}
        
    def forward(self, features: torch.Tensor, interaction_matrix: pd.DataFrame, 
                sirna_labels: Union[List[str], torch.Tensor]) -> torch.Tensor:
        device = features.device

        unique_sirnas, inverse_indices = np.unique(sirna_labels, return_inverse=True)
        inverse_indices = torch.tensor(inverse_indices, device=device)
        n_unique = len(unique_sirnas)
        
        interaction_matrix = deduplicate_interaction_matrix(interaction_matrix)
        unique_interaction = interaction_matrix.loc[unique_sirnas, unique_sirnas]

        interaction_values = unique_interaction.values
        interaction_matrix = torch.tensor(interaction_values, dtype=torch.float32, device=device)
        ones = torch.ones(features.size(0), 1, device=device)
        count_per_label = torch.zeros(n_unique, 1, device=device)
        count_per_label.scatter_add_(0, inverse_indices.unsqueeze(1), ones)
        
        avg_features = torch.zeros(n_unique, features.size(1), device=device)
        avg_features.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, features.size(1)), features)
        avg_features = avg_features / (count_per_label + 1e-8)
        
        if self.normalize_features:
            avg_features = F.normalize(avg_features, p=2, dim=1)

        W = interaction_matrix.clone().fill_diagonal_(1.0)

        D_diag = torch.sum(W, dim=1)
        # print('D_diag', min(D_diag))
        D_diag = torch.clamp(D_diag, min=0)
        
        D_inv_sqrt_diag = 1.0 / torch.sqrt(D_diag + 1e-8)

        XTX_trace = torch.sum(avg_features * avg_features)

        X_scaled = avg_features * D_inv_sqrt_diag.unsqueeze(1)

        WX_scaled = torch.matmul(W, X_scaled)
        
        similarity_trace = torch.sum(X_scaled * WX_scaled)
        loss = XTX_trace - similarity_trace
        loss = loss/ n_unique
        return self.lambda_reg * loss




class GraphContrastiveLoss(nn.Module):
    def __init__(self, normalize_features: bool = True, lambda_reg: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.normalize_features = normalize_features
        self.lambda_reg = lambda_reg
        self.temperature = temperature
        
    def forward(self, features: torch.Tensor, 
                interaction_matrix: Union[pd.DataFrame, torch.Tensor],
                sirna_labels: Union[List[str], torch.Tensor]) -> torch.Tensor:
        device = features.device

        unique_sirnas, inverse_indices = np.unique(sirna_labels, return_inverse=True)
        inverse_indices = torch.tensor(inverse_indices, device=device)
        n_unique = len(unique_sirnas)
        
        if isinstance(interaction_matrix, pd.DataFrame):
            interaction_matrix = deduplicate_interaction_matrix(interaction_matrix)
            unique_interaction = interaction_matrix.loc[unique_sirnas, unique_sirnas]
            interaction_values = unique_interaction.values
            interaction_matrix = torch.tensor(interaction_values, dtype=torch.float32, device=device)
        else:
            interaction_matrix = interaction_matrix.to(device)
        
        ones = torch.ones(features.size(0), 1, device=device)
        count_per_label = torch.zeros(n_unique, 1, device=device)
        count_per_label.scatter_add_(0, inverse_indices.unsqueeze(1), ones)
        
        avg_features = torch.zeros(n_unique, features.size(1), device=device)
        avg_features.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, features.size(1)), features)
        avg_features = avg_features / (count_per_label + 1e-8)
        
        if self.normalize_features:
            avg_features = F.normalize(avg_features, p=2, dim=1)
        
        sim = torch.matmul(avg_features, avg_features.t()) / self.temperature
        
        if interaction_matrix.max() > 1:
            W = interaction_matrix / 1000.0
        else:
            W = interaction_matrix.clone()
        
        mask = torch.eye(n_unique, device=device)
        sim = sim * (1 - mask)
        W = W * (1 - mask)
        
        log_softmax = F.log_softmax(sim, dim=1)
        loss = -(W * log_softmax).sum(dim=1).mean()
        
        return self.lambda_reg * loss