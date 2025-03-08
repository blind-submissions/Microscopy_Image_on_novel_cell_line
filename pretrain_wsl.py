import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import wandb
from tqdm import tqdm
import argparse

import sys
sys.path.append('..') 
from pretrain.dataset import *
from models.vit import *
from models.utils import *
from models.resnet import *
import pandas as pd
full_matrix = pd.read_pickle("xxxx")

class DDPWSLTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        local_rank: int
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.local_rank = local_rank
        self.is_main_process = local_rank == 0

        self.device = torch.device(f"cuda:{local_rank}")
        self.model = self.model.to(self.device)
        if local_rank != -1:
            self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=True)

        self.total_steps = len(self.train_loader) * config['epochs']

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.95),
            weight_decay=config.get('weight_decay', 0.0)
        )

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            total_steps=self.total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            cycle_momentum=False,
            div_factor=25.0,
            final_div_factor=1000.0,
        )
        self.scaler = torch.cuda.amp.GradScaler()

        if config['use_wandb'] and self.is_main_process:
            wandb.init(
                project=config['wandb_project_name'],
                name=config['wandb_run_name'],
                config=config
            )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = torch.tensor(0.0).to(self.device)
        total_acc = torch.tensor(0.0).to(self.device)
        train_iter = self.train_loader

        if self.is_main_process:
            train_iter = tqdm(train_iter, desc=f'Epoch {epoch+1}')

        for batch_idx, batch in enumerate(train_iter):
            images = batch["pixels"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            feas = batch["domain_feature"].to(self.device, non_blocking=True)
            sirna_set = batch['sirna']
            batch_matrix = get_submatrix(full_matrix,sirna_set)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model.module.training_step({"pixels": images, "labels": labels, "domain_feature": feas, "batch_matrix": batch_matrix}, batch_idx)
                loss = outputs["loss"]
                acc = outputs["acc"]

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()     
            self.scheduler.step()

            total_loss += loss.item()
            total_acc += acc.item()
            current_lr = self.scheduler.get_last_lr()[0]

            if self.is_main_process:
                train_iter.set_postfix(
                    loss=loss.item(), 
                    acc=acc.item(),
                    lr=current_lr
                )
                
                if self.config['use_wandb'] and batch_idx % 10 == 0:
                    wandb.log({
                        "train_loss": loss.item(),
                        "train_acc": acc.item(),
                        "learning_rate": current_lr,
                        "epoch": epoch,
                        "step": batch_idx
                    })

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_acc, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        
        avg_loss = total_loss.item() / (len(self.train_loader) * world_size)
        avg_acc = total_acc.item() / (len(self.train_loader) * world_size)
        return avg_loss, avg_acc

    def validate(self):
        self.model.eval()
        total_loss = torch.tensor(0.0).to(self.device)
        total_acc = torch.tensor(0.0).to(self.device)

        with torch.no_grad():
            val_iter = self.val_loader
            if self.is_main_process:
                val_iter = tqdm(val_iter, desc='Validating')

            for batch in val_iter:
                images = batch["pixels"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)
                feas = batch["domain_feature"].to(self.device, non_blocking=True)
                sirna_set = batch['sirna']
                batch_matrix = get_submatrix(full_matrix,sirna_set)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = self.model.module.validation_step({"pixels": images, "labels": labels, "domain_feature": feas,  "batch_matrix": batch_matrix}, 0)
                    loss = outputs["loss"]
                    acc = outputs["acc"]
                
                total_loss += loss.item()
                total_acc += acc.item()

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_acc, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        
        avg_val_loss = total_loss.item() / (len(self.val_loader) * world_size)
        avg_val_acc = total_acc.item() / (len(self.val_loader) * world_size)
        
        if self.is_main_process and self.config['use_wandb']:
            wandb.log({
                "val_loss": avg_val_loss,
                "val_acc": avg_val_acc
            })
        return avg_val_loss, avg_val_acc

    def train(self):
        best_val_loss = float('inf')
        best_val_acc = 0.0
        early_stop_counter = 0
        patience = self.config.get('early_stopping_patience', 5)

        for epoch in range(self.config['epochs']):
            self.train_loader.sampler.set_epoch(epoch)
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            if self.is_main_process:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    backbone = self.config['backbone']
                    self.save_checkpoint(
                        os.path.join(self.config['checkpoint_dir'], f'best_wsl_{backbone}.pt'),
                        epoch,
                        val_loss,
                        val_acc
                    )
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                print(
                    f'Epoch {epoch+1}/{self.config["epochs"]}, '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}'
                )

                if early_stop_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        dist.barrier()

    def save_checkpoint(self, path, epoch, val_loss, val_acc):
        torch.save({
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),  
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': self.config
        }, path)

def main():
    parser = argparse.ArgumentParser(description='Train WSL model with DDP')
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    parser.add_argument('--batch_size', type=int, default=384)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--backbone', type=str, default='resnet')
    parser.add_argument('--num_classes', type=int, default=1138)
    parser.add_argument('--pooling_type', type=str, default='mean')
    parser.add_argument('--use_cls', action='store_true', default=True)
    parser.add_argument('--domain_feature_dim', type=int, default=128)
    parser.add_argument('--cond_strength', type=float, default=0.)
    parser.add_argument('--lp_reg', type=float, default=0.0)

    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--pkl_path', type=str)
 
    parser.add_argument('--split', type=str, default='split')

    parser.add_argument('--checkpoint_dir', type=str, default='xxx')
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project_name', type=str, default='xxx')
    parser.add_argument('--wandb_run_name', type=str, default='pretrain-run')

    args = parser.parse_args()
    config = vars(args)


    def setup_distributed():
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
            return local_rank
        return 0
    
    local_rank = setup_distributed()
    
    if local_rank == 0:
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        print(config)
        
    if config['backbone'] == 'densenet':
        print('use densenet-161')
        wsl_config = DenseNetWSLConfig(
            num_classes= config['num_classes'],
            domain_feature_dim= config['domain_feature_dim'], 
            cond_strength = config['cond_strength'],
            lp_reg = config['lp_reg'],
        )
        
        model = DenseNetWSLModel(wsl_config)
        config['weight_decay'] = 1e-5
    elif config['backbone'] == 'resnet':
        print('use resnet-50')
        wsl_config = ResNetWSLConfig(
            num_classes= config['num_classes'],
            domain_feature_dim= config['domain_feature_dim'], 
            cond_strength = config['cond_strength'],
            lp_reg = config['lp_reg'],
        )
        
        model = ResNetWSLModel(wsl_config)
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
        )
        model = ViTWSLModel(wsl_config, num_classes =  config['num_classes'])
        config['weight_decay'] = 0.05
        model.encoder.load_pretrained_weights()
        
    if dist.is_initialized():
        dist.barrier()
        
    train_loader, val_loader = create_data_loaders_ddp(args)  

    trainer = DDPWSLTrainer(model, train_loader, val_loader, config, local_rank)
    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()