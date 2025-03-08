import pandas as pd 
import torch
from typing import Type, Union, List, Tuple
import numpy as np
from .utils import get_submatrix

class PGMemoryBank:
    def __init__(self, feature_dim: int, full_matrix =None, momentum: float = 0.2):
        self.feature_dim = feature_dim
        self.full_matrix = full_matrix
        self.momentum = momentum
        
        self.features = {}  
        self.sirna_set = set()  
        
    def set_full_matrix(self,fm):
        self.full_matrix = fm
    
    def reset(self):
        self.features = {}
        self.sirna_set = set()
    
    def update(self, features: torch.Tensor, sirna_labels: List[str]):
        features_np = features.detach().cpu().numpy()
        
        for i, sirna in enumerate(sirna_labels):
            if sirna in self.features:
                old_feature = self.features[sirna]
                new_feature = old_feature * self.momentum + features_np[i] * (1 - self.momentum)
                self.features[sirna] = new_feature
            else:
                self.features[sirna] = features_np[i]
            
            self.sirna_set.add(sirna)
    
    def get_features_tensor(self, device=None) -> Tuple[torch.Tensor, List[str]]:
        if not self.features:
            return None, []
        
        sirna_list = list(self.sirna_set)
        features_list = [self.features[sirna] for sirna in sirna_list]
        
        features_tensor = torch.tensor(np.array(features_list), dtype=torch.float32)
        
        if device is not None:
            features_tensor = features_tensor.to(device)
        
        return features_tensor, sirna_list
    
    def compute_regularization_loss(self, lploss, device=None):

        features_tensor, sirna_list = self.get_features_tensor(device)
        
        if features_tensor is None or len(sirna_list) < 2:
            return torch.tensor(0.0, device=device if device else torch.device('cpu'))
        
        batch_matrix = get_submatrix(self.full_matrix, sirna_list, merge_duplicates=False)
        
        with torch.no_grad():
            loss = lploss(features_tensor, batch_matrix, sirna_list)
        
        return loss
    
    def compute_mixed_loss(self, lploss, cur_features: torch.Tensor, cur_sirna_labels: List[str], device=None):

        if not self.features:
            batch_matrix = get_submatrix(self.full_matrix, cur_sirna_labels, merge_duplicates=True)
            return lploss(cur_features, batch_matrix, cur_sirna_labels)
        
        memory_features, memory_sirna_list = self.get_features_tensor(device)
        combined_sirna_list = memory_sirna_list.copy()
        combined_features_list = [memory_features]
        
        cur_features_list = []
        new_sirna_list = []
        
        for i, sirna in enumerate(cur_sirna_labels):
            if sirna not in self.sirna_set:
                new_sirna_list.append(sirna)
                cur_features_list.append(cur_features[i])
        
        if cur_features_list:
            combined_sirna_list.extend(new_sirna_list)
            cur_features_tensor = torch.stack(cur_features_list)
            combined_features_list.append(cur_features_tensor)
        
        combined_features = torch.cat(combined_features_list, dim=0)
        
        batch_matrix = get_submatrix(self.full_matrix, combined_sirna_list)#, merge_duplicates=True)

        sirna_to_idx = {sirna: i for i, sirna in enumerate(combined_sirna_list)}
        

        for i, sirna in enumerate(cur_sirna_labels):
            if sirna in sirna_to_idx:
                idx = sirna_to_idx[sirna]
                if idx < memory_features.size(0):
                    alpha = 0.7  
                    combined_features[idx] =  cur_features[i] #+ (1-alpha) * combined_features[idx] 
        loss = lploss(combined_features, batch_matrix)
        
        return loss