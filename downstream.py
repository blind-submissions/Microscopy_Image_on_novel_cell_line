import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import wandb
from tqdm import tqdm
import argparse
import sys
sys.path.append('..') 
from dataset import *
from models.vit import *

class WSLTrainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.total_steps = len(self.train_loader) * config['epochs']

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.95),
            weight_decay=config.get('weight_decay', 1e-5)
        )

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            total_steps=self.total_steps,
            pct_start=0.05,
            anneal_strategy='cos',
            cycle_momentum=False,
            div_factor=25.0,
            final_div_factor=1000.0,
        )

        if config['use_wandb']:
            wandb.init(
                project=config['wandb_project_name'],
                name=config['wandb_run_name'],
                config=config
            )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_acc = 0
        train_iter = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')

        for batch_idx, batch in enumerate(train_iter):
            images = batch["pixels"].to(self.device)
            labels = batch["label"].to(self.device)
            feas = batch["domain_feature"].to(self.device)
            outputs = self.model.training_step({"pixels": images, "labels": labels, "domain_feature": feas}, batch_idx)
            loss = outputs["loss"]
            acc = outputs["acc"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_acc += acc.item()
            current_lr = self.scheduler.get_last_lr()[0]

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

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_acc / len(self.train_loader)
        return avg_loss, avg_acc

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_acc = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                images = batch["pixels"].to(self.device)
                labels = batch["label"].to(self.device)
                feas = batch["domain_feature"].to(self.device)
 
                outputs = self.model.validation_step({"pixels": images, "labels": labels,  "domain_feature": feas}, 0)
                loss = outputs["loss"]
                acc = outputs["acc"]
                
                total_loss += loss.item()
                total_acc += acc.item()

        avg_val_loss = total_loss / len(self.val_loader)
        avg_val_acc = total_acc / len(self.val_loader)
        
        if self.config['use_wandb']:
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
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(
                    os.path.join(self.config['checkpoint_dir'], 'best_model.pt'),
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

    def save_checkpoint(self, path, epoch, val_loss, val_acc):
        torch.save({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': self.config
        }, path)

def main():
    parser = argparse.ArgumentParser(description='Train ViT classifier on de novo cell line')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=16)

    parser.add_argument('--num_classes', type=int, default=1138)
    parser.add_argument('--backbone', type=str, default='vit-s')
    parser.add_argument('--pooling_type', type=str, default='mean')
    parser.add_argument('--use_cls', action='store_true', default=True)
    
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--cell_line', type=str, default='U2OS')
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--split', type=str, default='split1')
    parser.add_argument('--domain_feature_dim', type=int, default=0)
    parser.add_argument('--cond_strength', type=int, default=0.1)
    parser.add_argument('--pkl_path', type=str)
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints/downstream')
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project_name', type=str, default='xxx')
    parser.add_argument('--wandb_run_name', type=str, default='downstream-run')
    parser.add_argument('--pretrained_path', type=str, default="xxx")

    args = parser.parse_args()
    config = vars(args)
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)


    
    if config['backbone'] == 'vit-s':
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
        config['weight_decay'] = 1e-5
        model.encoder.load_pretrained_weights()
        
    
    train_loader, val_loader = create_data_loaders(args)

    trainer = WSLTrainer(model, train_loader, val_loader, config)
    trainer.train()

if __name__ == '__main__':
    main()