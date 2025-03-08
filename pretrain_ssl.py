import os
import argparse
import yaml
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from models.ssl_models import *
from models.ssl_transforms import *
from models.resnet import *
from models.vit import *
from datasets import *
from trainer import SSLTrainer, SSLMethod
from models.gr_loss import *

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_process(rank, world_size, args):
    setup_ddp(rank, world_size)
    
    set_seed(args.seed + rank)
    
    config = vars(args)
    
    train_loader = create_data_loaders_ddp(args)
    
    if config['backbone'] == 'resnet':
        print('use resnet-50')

        encoder = ConditionalResNet(domain_feature_dim= config['domain_feature_dim'], 
                                    cond_strength = config['cond_strength'])
        config['weight_decay'] = 1e-5   
         
    elif config['backbone'] == 'vit-s':
        print('use vit-s')
        wsl_config = ViTConfig(
            model_type='small',  
            in_chans=6,
            img_size=256,
            patch_size=16,
            use_cls_token=config['use_cls'],
            pooling_type=config['pooling_type'],
            init_pos_embed_type='sincos',
            domain_feature_dim =  config['domain_feature_dim'] ,
            cond_strength = config['cond_strength'],
        )
        encoder = ViTModel(wsl_config)
        
        encoder.load_state_dict(encoder.from_pretrained('xxx'), strict=False)

    model = create_ssl_model(
        method=args.ssl_method,
        encoder=encoder,
        embed_dim=encoder.embed_dim,
        supervised_loss_fn = LaplacianRegularizationLoss(),
        supervised_weight = config['lp_reg']
    )

    trainer = SSLTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        local_rank=rank
    )
    
    trainer.train()
    
    cleanup()

def main():
    parser = argparse.ArgumentParser(description='Self-supervised learning pre-training script')

    parser.add_argument('--ssl_method', type=str, default='byol', choices=['byol', 'simclr', 'dino'])
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='adamw')

    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint save directory')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of data loading workers')
    parser.add_argument('--num_gpus', type=int, default=-1, help='Number of GPUs to use, -1 means use all available')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', default= 'True')
    parser.add_argument('--verbose', action='store_true', help='Whether to print verbose information')
    parser.add_argument('--wandb_project_name', type=str, default='xxx')
    parser.add_argument('--backbone', type=str, default='vit-s')
    parser.add_argument('--pooling_type', type=str, default='mean')
    parser.add_argument('--use_cls', action='store_true', default=True)
    parser.add_argument('--domain_feature_dim', type=int, default=128)
    parser.add_argument('--cond_strength', type=float, default=0.1)
    parser.add_argument('--lp_reg', type=float, default=0.0)

    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--pkl_path', type=str)
    args = parser.parse_args()
    
    if args.num_gpus == -1:
        args.num_gpus = torch.cuda.device_count()
    
    if args.num_gpus < 1:
        print("Error: No available GPUs! At least 1 GPU is required for DDP training.")
        return
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(args)
    
    print(f"Starting distributed training with {args.num_gpus} GPUs")
    print(f"SSL method: {args.ssl_method}")
    print(f"Batch size: {args.batch_size} (per GPU)")
    print(f"Total batch size: {args.batch_size * args.num_gpus}")
    
    mp.spawn(
        train_process,
        args=(args.num_gpus, args),
        nprocs=args.num_gpus,
        join=True
    )

if __name__ == "__main__":
    main()