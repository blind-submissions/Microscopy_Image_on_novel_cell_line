import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from enum import Enum
from typing import Dict, Optional
import pandas as pd 
from utils import *

class SSLMethod(Enum):
    BYOL = 'byol'
    SimCLR = 'simclr'
    DINO = 'dino'

full_matrix = pd.read_pickle("xxxx")
for i in range(len(full_matrix)):
    full_matrix.iloc[i, i] = 1.0 

class SSLTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: dict,
        local_rank: int = -1
    ):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.local_rank = local_rank
        self.is_main_process = local_rank in [-1, 0]
        self.use_ddp = local_rank != -1
        
        # Setup device
        if self.use_ddp:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device and setup DDP
        self.model = self.model.to(self.device)
        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=True)
        
        # Calculate total steps
        self.total_steps = len(self.train_loader) * config['epochs']
        
        opt_class = AdamW if config.get('optimizer', 'adamw').lower() == 'adamw' else torch.optim.Adam
        self.optimizer = opt_class(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.95),
            weight_decay=config.get('weight_decay', 0.0)
        )
        
        if config.get('scheduler', 'onecycle').lower() == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config['learning_rate'],
                total_steps=self.total_steps,
                pct_start=config.get('pct_start', 0.1),
                anneal_strategy='cos',
                cycle_momentum=False,
                div_factor=config.get('div_factor', 25.0),
                final_div_factor=config.get('final_div_factor', 1000.0),
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_steps,
                eta_min=config.get('min_lr', 1e-6)
            )
        
        self.use_amp = config.get('use_amp', True)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb and self.is_main_process:
            wandb.init(
                project=config.get('wandb_project_name', 'SSL-Training'),
                name=config.get('wandb_run_name', f"{config.get('ssl_method', 'ssl')}-training"),
                config=config
            )
        
        self.ssl_method = config.get('ssl_method', 'byol')
        
        self.patience = config.get('early_stopping_patience', 5)
    
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_ssl_loss = 0.0
        total_supervised_loss = 0.0
        
        # Set current epoch for DDP mode
        if self.use_ddp:
            self.train_loader.sampler.set_epoch(epoch)
        
        train_iter = self.train_loader
        if self.is_main_process:
            train_iter = tqdm(train_iter, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(train_iter):
            # Move data to device
            processed_batch = {}
            sirna_set = batch['sirna']
            batch_matrix = get_submatrix(full_matrix,sirna_set)
            processed_batch['batch_matrix'] = batch_matrix
 
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    processed_batch[k] = v.to(self.device,non_blocking=True)
                elif isinstance(v, list) and all(isinstance(item, torch.Tensor) for item in v):
                    processed_batch[k] = [item.to(self.device) for item in v]
                else:
                    processed_batch[k] = v
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    if self.use_ddp:
                        outputs = self.model.module.forward(processed_batch)
                    else:
                        outputs = self.model.forward(processed_batch)
                    
                    loss = outputs["loss"]
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.use_ddp:
                    outputs = self.model.module.forward(processed_batch)
                else:
                    outputs = self.model.forward(processed_batch)
                
                loss = outputs["loss"]
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Accumulate losses
            total_loss += loss.item()
            
            # Record SSL and supervised losses
            if "ssl_loss" in outputs:
                ssl_loss = outputs["ssl_loss"].item() if isinstance(outputs["ssl_loss"], torch.Tensor) else outputs["ssl_loss"]
                total_ssl_loss += ssl_loss
            elif self.ssl_method == "byol" and "byol_loss" in outputs:
                ssl_loss = outputs["byol_loss"].item() if isinstance(outputs["byol_loss"], torch.Tensor) else outputs["byol_loss"]
                total_ssl_loss += ssl_loss
            elif self.ssl_method == "simclr" and "simclr_loss" in outputs:
                ssl_loss = outputs["simclr_loss"].item() if isinstance(outputs["simclr_loss"], torch.Tensor) else outputs["simclr_loss"]
                total_ssl_loss += ssl_loss
            elif self.ssl_method == "dino" and "dino_loss" in outputs:
                ssl_loss = outputs["dino_loss"].item() if isinstance(outputs["dino_loss"], torch.Tensor) else outputs["dino_loss"]
                total_ssl_loss += ssl_loss
            
            if "supervised_loss" in outputs:
                sup_loss = outputs["supervised_loss"].item() if isinstance(outputs["supervised_loss"], torch.Tensor) else outputs["supervised_loss"]
                total_supervised_loss += sup_loss
            
            # Update progress bar
            if self.is_main_process:
                current_lr = self.scheduler.get_last_lr()[0]
                train_iter.set_postfix(
                    loss=loss.item(),
                    lr=current_lr
                )
                
                # Log to Wandb
                if self.use_wandb and batch_idx % 10 == 0:
                    log_dict = {
                        "train_loss": loss.item(),
                        "learning_rate": current_lr,
                        "epoch": epoch,
                        "step": batch_idx + epoch * len(self.train_loader)
                    }
                    
                    if "ssl_loss" in outputs or f"{self.ssl_method}_loss" in outputs:
                        log_dict["ssl_loss"] = ssl_loss
                    
                    if "supervised_loss" in outputs:
                        log_dict["supervised_loss"] = sup_loss
                    
                    wandb.log(log_dict)
        
        # Calculate average loss
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_ssl_loss = total_ssl_loss / num_batches
        avg_supervised_loss = total_supervised_loss / num_batches
        
        # Synchronize loss in DDP mode
        if self.use_ddp:
            world_size = dist.get_world_size()
            loss_tensor = torch.tensor(avg_loss).to(self.device)
            ssl_loss_tensor = torch.tensor(avg_ssl_loss).to(self.device)
            sup_loss_tensor = torch.tensor(avg_supervised_loss).to(self.device)
            
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(ssl_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(sup_loss_tensor, op=dist.ReduceOp.SUM)
            
            avg_loss = loss_tensor.item() / world_size
            avg_ssl_loss = ssl_loss_tensor.item() / world_size
            avg_supervised_loss = sup_loss_tensor.item() / world_size
        
        # Log to Wandb
        if self.is_main_process and self.use_wandb:
            wandb.log({
                "epoch_train_loss": avg_loss,
                "epoch_ssl_loss": avg_ssl_loss,
                "epoch_supervised_loss": avg_supervised_loss,
                "epoch": epoch
            })
        
        return avg_loss
    
    def train(self):
        best_train_loss = float('inf')
        early_stop_counter = 0
        
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(epoch)
            
            # Main process handles checkpoint saving and early stopping
            if self.is_main_process:
                # Save checkpoint
                checkpoint_dir = self.config.get('checkpoint_dir', './checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                method_name = self.config.get('ssl_method', 'ssl')
                backbone_name = self.config.get('backbone', 'encoder')
                
                # Save checkpoint for current epoch
                if (epoch + 1) % self.config.get('save_every', 10) == 0:
                    self.save_checkpoint(
                        os.path.join(checkpoint_dir, f'{method_name}_{backbone_name}_epoch{epoch+1}.pt'),
                        epoch,
                        train_loss
                    )
                
                # Save best checkpoint if loss improves
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    self.save_checkpoint(
                        os.path.join(checkpoint_dir, f'best_{method_name}_{backbone_name}.pt'),
                        epoch,
                        train_loss
                    )
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                print(
                    f'Epoch {epoch+1}/{self.config["epochs"]}, '
                    f'Train Loss: {train_loss:.4f}, '
                    f'Best Loss: {best_train_loss:.4f}, '
                    f'Early Stop Counter: {early_stop_counter}/{self.patience}'
                )
                
                # Early stopping
                if early_stop_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Save final encoder
        if self.is_main_process:
            self.save_encoder(os.path.join(checkpoint_dir, f'final_encoder_{backbone_name}.pt'))
        
        # Wait for all processes to complete in DDP mode
        if self.use_ddp:
            dist.barrier()
    
    def save_checkpoint(self, path: str, epoch: int, loss: float):
        state_dict = self.model.module.state_dict() if self.use_ddp else self.model.state_dict()
        
        torch.save({
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }, path)
        
        print(f"Checkpoint saved to {path}")
    
    def save_encoder(self, path: str):
        model = self.model.module if self.use_ddp else self.model
        
        if self.ssl_method.lower() == 'byol':
            encoder_state_dict = model.online_encoder.state_dict()
        elif self.ssl_method.lower() == 'simclr':
            encoder_state_dict = model.encoder.state_dict()
        elif self.ssl_method.lower() == 'dino':
            encoder_state_dict = model.student_encoder.state_dict()
        else:
            # Generic approach: try to get encoder attribute
            if hasattr(model, 'encoder'):
                encoder_state_dict = model.encoder.state_dict()
            elif hasattr(model, 'online_encoder'):
                encoder_state_dict = model.online_encoder.state_dict()
            elif hasattr(model, 'student_encoder'):
                encoder_state_dict = model.student_encoder.state_dict()
            else:
                raise ValueError(f"Unrecognized SSL method: {self.ssl_method}, cannot extract encoder")
        
        torch.save({
            'encoder_state_dict': encoder_state_dict,
            'config': self.config
        }, path)
        
        print(f"Encoder saved to {path}")