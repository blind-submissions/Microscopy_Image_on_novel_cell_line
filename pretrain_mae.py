import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
import wandb
from tqdm import tqdm
import argparse

import sys
sys.path.append('..')
from pretrain.dataset import *
from models.vit import *
from models.mae import *
import pandas as pd 
full_matrix = pd.read_pickle("xxx")

class DDPMAETrainer:
    def __init__(
        self,
        model: MAEPreTrainingModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        local_rank: int
    ):
        self.model = model
        self.model.memory_bank.set_full_matrix(full_matrix)

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

        self.scaler = GradScaler()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.95),
            weight_decay=config.get('weight_decay', 0.05)
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

        if config['use_wandb'] and self.is_main_process:
            wandb.init(
                project=config['wandb_project_name'],
                name=config['wandb_run_name'],
                config=config
            )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = torch.tensor(0.0).to(self.device)
        train_iter = self.train_loader
        self.model.module.memory_bank.reset()
        if self.is_main_process:
            train_iter = tqdm(train_iter, desc=f'Epoch {epoch+1}')

        for batch_idx, batch in enumerate(train_iter):
            images = batch["pixels"].to(self.device)
            feas = batch["domain_feature"].to(self.device)
            sirna_set = batch['sirna']
            
            with autocast():
                outputs = self.model.module.training_step({"pixels": images, "domain_feature": feas, 'sirna': sirna_set}, batch_idx)
                loss = outputs["loss"]

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            current_lr = self.scheduler.get_last_lr()[0]

            if self.is_main_process:
                train_iter.set_postfix(
                    loss=loss.item(),
                    lr=current_lr
                )
                
                if self.config['use_wandb'] and batch_idx % 10 == 0:
                    wandb.log({
                        "train_loss": loss.item(),
                        "learning_rate": current_lr,
                        "epoch": epoch,
                        "step": batch_idx
                    })

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        
        avg_loss = total_loss.item() / (len(self.train_loader) * world_size)
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = torch.tensor(0.0).to(self.device)

        with torch.no_grad():
            val_iter = self.val_loader
            if self.is_main_process:
                val_iter = tqdm(val_iter, desc='Validating')

            for batch in val_iter:
                images = batch["pixels"].to(self.device)
                feas = batch["domain_feature"].to(self.device)

                with autocast():
                    outputs = self.model.module.validation_step({"pixels": images, "domain_feature": feas}, 0)
                    loss = outputs["loss"]
                
                total_loss += loss.item()

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        
        avg_val_loss = total_loss.item() / (len(self.val_loader) * world_size)
        
        if self.is_main_process and self.config['use_wandb']:
            wandb.log({"val_loss": avg_val_loss})
            
        return avg_val_loss

    def train(self):
        best_val_loss = float('inf')
        early_stop_counter = 0
        patience = self.config.get('early_stopping_patience', 5)

        for epoch in range(self.config['epochs']):
            self.train_loader.sampler.set_epoch(epoch)
            
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            if self.is_main_process:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(self.config['checkpoint_dir'], 'best_mae.pt'),
                        epoch,
                        val_loss
                    )
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                print(
                    f'Epoch {epoch+1}/{self.config["epochs"]}, '
                    f'Train Loss: {train_loss}, '
                    f'Val Loss: {val_loss}'
                )

                if early_stop_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
                
        if dist.is_initialized():
            dist.barrier()

    def save_checkpoint(self, path, epoch, val_loss):
        torch.save({
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }, path)

def main():
    parser = argparse.ArgumentParser(description='Train MAE model with DDP')
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    parser.add_argument('--batch_size', type=int, default=160)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--num_workers', type=int, default=10)
    
    parser.add_argument('--backbone', type=str, default='vit-s')
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--pooling_type', type=str, default='mean')
    parser.add_argument('--use_cls', action='store_true', default=True)
    parser.add_argument('--domain_feature_dim', type=int, default=512)
    parser.add_argument('--cond_strength', type=float, default=0.1)
    parser.add_argument('--lp_reg', type=float, default=0) 
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--pkl_path', type=str)
    parser.add_argument('--split', type=str, default='split')
    
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project_name', type=str, default='xx')
    parser.add_argument('--wandb_run_name', type=str, default='pretrain-run')
    parser.add_argument('--mae_pretrained_path', type=str, default="xx")

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

    vit_config = ViTConfig(
        model_type='small',  
        in_chans=6,
        img_size=256,
        patch_size=16,
        use_cls_token=config['use_cls'],
        pooling_type=config['pooling_type'],
        init_pos_embed_type='sincos',
        domain_feature_dim =  config['domain_feature_dim'] ,
        cond_strength= config['cond_strength']
    )
    vit_model = ViTModel(vit_config)
    
    mae_config = MAEConfig(
        mask_ratio=config['mask_ratio'],
        in_chans=6
    )
    model = MAEPreTrainingModel(vit_model, mae_config, lp_reg = config['lp_reg'])

    if dist.is_initialized():
        dist.barrier()
        
    train_loader, val_loader = create_data_loaders_ddp(args)

    trainer = DDPMAETrainer(model, train_loader, val_loader, config, local_rank)
    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()