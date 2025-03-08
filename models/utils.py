import torch
import pandas as pd
import numpy as np
from typing import List, Union

class Normalizer(torch.nn.Module):
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels
        return pixels / 255.0
    
    
def get_submatrix(full_matrix: pd.DataFrame, 
                  sirna_subset: Union[List[str], np.ndarray], 
                  ensure_order: bool = True) -> pd.DataFrame:

    if isinstance(sirna_subset, np.ndarray):
        sirna_subset = sirna_subset.tolist()
    
    submatrix = full_matrix.loc[sirna_subset, sirna_subset]
    if ensure_order:
        submatrix = submatrix.reindex(index=sirna_subset, columns=sirna_subset)
    
    return submatrix