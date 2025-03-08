import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pickle
import numpy as np
from models.ssl_transforms import *

class RXDataset(Dataset):
    def __init__(self, csv_path, root_dir, latent_representation_path, domain_feature_dim = 128,
                 split='split', mode='train', transform=None):
        self.data = pd.read_csv(csv_path)
        # self.data = self.data[self.data[split] == mode].reset_index(drop=True)
        self.label_mapping = {label: idx for idx, label in enumerate(sorted(self.data["sirna"].unique()))}
        
        self.root_dir = root_dir
        self.transform = transform
        
        with open(latent_representation_path, 'rb') as f:
            self.latent_representation_dict = pickle.load(f)
        self.domain_feature_dim = domain_feature_dim
        
        if domain_feature_dim in self.latent_representation_dict:
            self.domain_feature_dict = self.latent_representation_dict[domain_feature_dim]['latent']
        elif str(domain_feature_dim) in self.latent_representation_dict:
            self.domain_feature_dict = self.latent_representation_dict[str(domain_feature_dim)]['latent']
        else:
            raise ValueError(f"Domain feature dimension {domain_feature_dim} not found in latent representation dict.")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        img_paths = [
            os.path.join(self.root_dir, row["experiment"], f"Plate{row['plate']}", f"{row['well']}_s{row['site']}_w{i}.png")
            for i in range(1, 7)
        ]
        
        img_channels = [transforms.ToTensor()(Image.open(p)) for p in img_paths]
        img_tensor = torch.cat(img_channels, dim=0)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        label = self.label_mapping[row["sirna"]]
        cell_type = row["cell_type"]
        sirna = row["sirna"]
        
        if cell_type in self.domain_feature_dict:
            domain_feature = self.domain_feature_dict[cell_type]
            if not isinstance(domain_feature, torch.Tensor):
                domain_feature = np.array(domain_feature, dtype=np.float32)
                domain_feature = torch.tensor(domain_feature, dtype=torch.float)
        else:
            raise ValueError(f"Cell type {cell_type} not found in latent representation dict.")
        
        return {
            "pixels": img_tensor,
            "label": label,
            "cell_type": cell_type,
            "sirna": sirna,
            "domain_feature": domain_feature
        }


class SSLDatasetAdapter(Dataset):
    
    def __init__(self, base_dataset: RXDataset, ssl_method, image_size: int = 256):
        self.base_dataset = base_dataset
        self.ssl_method = ssl_method
        self.transform = SSLTransforms(method=ssl_method, image_size=image_size)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        image = sample["pixels"]
        domain_feature = sample["domain_feature"]
        label = sample["label"]
        sirna = sample['sirna']           
 


        if self.ssl_method in ['byol','simclr']:
            view1, view2 = self.transform(image)

            return {
                "view1": view1,
                "view2": view2,
                "domain_feature": domain_feature,
                "labels": label,
                "sirna": sirna,
                "imgs": image
            }
        
        elif self.ssl_method == 'dino':
            views = self.transform(image)  
            
            views["domain_feature"] = domain_feature
            views["labels"] = label
            views["sirna"] = sirna
            views['imgs'] = image
            
            return views
        

def random_crop_256(img):
    return transforms.RandomCrop(256)(img)

def center_crop_256(img):
    return transforms.CenterCrop(256)(img)

def random_horizontal_flip(img):
    return torch.flip(img, dims=[2]) if torch.rand(1).item() > 0.5 else img

def random_vertical_flip(img):
    return torch.flip(img, dims=[1]) if torch.rand(1).item() > 0.5 else img

def random_rotation(img):
    return torch.rot90(img, k=1, dims=[1, 2]) if torch.rand(1).item() > 0.5 else img

def create_data_loaders_ddp(args):
    train_transforms = transforms.Compose([
        transforms.Lambda(random_crop_256),
    ])
    
    train_dataset = RXDataset(
        csv_path=args.csv_path,
        root_dir=args.root_dir,
        latent_representation_path = args.pkl_path,
        domain_feature_dim = args.domain_feature_dim,
        mode='train',
        transform=train_transforms
    )
    ssl_dataset = SSLDatasetAdapter(
        base_dataset=train_dataset,
        ssl_method=args.ssl_method,
        image_size=256
    )
    
    train_sampler = (
        DistributedSampler(train_dataset) 
        if dist.is_initialized() else None
    )
    
    train_loader = DataLoader(
        ssl_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True 
    )
    return train_loader
    



def create_data_loaders(args):
    train_transforms = transforms.Compose([
        transforms.Lambda(lambda img: transforms.RandomCrop(256)(img)),
        transforms.Lambda(lambda img: torch.flip(img, dims=[2]) if torch.rand(1).item() > 0.5 else img),
        transforms.Lambda(lambda img: torch.flip(img, dims=[1]) if torch.rand(1).item() > 0.5 else img),
        transforms.Lambda(lambda img: torch.rot90(img, k=1, dims=[1, 2]) if torch.rand(1).item() > 0.5 else img),
    ])
    
    valid_transforms = transforms.Compose([
        transforms.Lambda(lambda img: transforms.CenterCrop(256)(img))
    ])
    
    train_dataset = RXDataset(
        csv_path=args.csv_path,
        root_dir=args.root_dir,
        latent_representation_path = args.pkl_path,
        domain_feature_dim = args.domain_feature_dim,
        split=args.split,
        mode='train',
        transform=train_transforms
    )
    
    val_dataset = RXDataset(
        csv_path=args.csv_path,
        root_dir=args.root_dir,
        latent_representation_path = args.pkl_path,
        domain_feature_dim = args.domain_feature_dim,
        split=args.split,
        mode='valid',
        transform=valid_transforms
    )
    
    train_sampler = None
    val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False  
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader