import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Union


class SSLTransforms:

    
    def __init__(self, method, image_size: int = 256):
        self.method = method
        self.image_size = image_size
        
        base_transforms = [
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=[-90, 90])
        ]
        
        if method == 'byol' or method == 'simclr':
            self.transform1 = transforms.Compose(base_transforms + [
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], 
                                      p=1.0 if method == 'byol' else 0.5),
            ])
            
            self.transform2 = transforms.Compose(base_transforms + [
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], 
                                      p=0.1 if method == 'byol' else 0.5),

            ])
        
        elif method == 'dino':
            self.global_transform = transforms.Compose(base_transforms + [
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            ])
            
            local_base_transforms = [
                transforms.RandomResizedCrop(image_size, scale=(0.1, 0.4)),
                transforms.RandomHorizontalFlip(),
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            ]
            self.local_transform = transforms.Compose(local_base_transforms)
    
    def __call__(self, x: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, List[torch.Tensor]]]:

        if self.method == 'byol' or self.method == 'simclr':
            return self.transform1(x), self.transform2(x)
        
        elif self.method == 'dino':
            global_views = [self.global_transform(x) for _ in range(2)]
            local_views = [self.local_transform(x) for _ in range(8)]
            
            return {
                'global_views': global_views,
                'local_views': local_views
            }

