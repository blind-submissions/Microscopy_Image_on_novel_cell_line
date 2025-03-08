import torch
import torch.nn as nn
import timm
from transformers import PreTrainedModel, PretrainedConfig
from .vit import *

from .gr_loss import *
from .pgmb import PGMemoryBank

class Normalizer(torch.nn.Module):
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return pixels / 255.0
    
class MAEConfig(PretrainedConfig):
    model_type = "mae"
    
    def __init__(
        self,
        mask_ratio: float = 0.75,
        decoder_embed_dim: int = 192, 
        decoder_depth: int = 6,        
        decoder_num_heads: int = 6,    
        in_chans: int = 6,            
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mask_ratio = mask_ratio
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.in_chans = in_chans

class MAEDecoder(nn.Module):
    def __init__(self, d_encoder, patch_size, in_chans, embed_dim, depth, num_heads):
        super().__init__()
        self.decoder_embed = nn.Linear(d_encoder, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        temp_model = timm.create_model(
            'vit_tiny_patch16_224',
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads
        )
        self.decoder_blocks = temp_model.blocks
        self.decoder_norm = temp_model.norm
        
        self.decoder_pred = nn.Linear(embed_dim, patch_size * patch_size * in_chans, bias=True)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)

    def forward(self, x, ids_restore):
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        for blk in self.decoder_blocks:
            x_ = blk(x_)
        x_ = self.decoder_norm(x_)
        
        x_ = self.decoder_pred(x_)
        
        return x_

class MAEPreTrainingModel(PreTrainedModel):
    config_class = MAEConfig
    
    def __init__(self, vit_model: nn.Module, mae_config: MAEConfig, lp_reg = 0.):
        super().__init__(mae_config)
        self.config = mae_config
        
        self.vit = vit_model
        
        d_model = self.vit.config.patch_size ** 2 * self.config.in_chans  
        self.decoder = MAEDecoder(
            d_encoder=self.vit.config.embed_dim,  
            patch_size=self.vit.config.patch_size,
            in_chans=self.config.in_chans,        # 
            embed_dim=mae_config.decoder_embed_dim,
            depth=mae_config.decoder_depth,
            num_heads=mae_config.decoder_num_heads
        )
        self.patch_proj = nn.Linear(
            d_model,
            self.vit.config.embed_dim
        )
        self.input_norm = torch.nn.Sequential(
            Normalizer(),
            nn.InstanceNorm2d(
                num_features=6,  
                affine=False, 
                track_running_stats=False
            )
        )
        self.grloss = LaplacianRegularizationLoss(
                        normalize_features=True,
                    )
        self.memory_bank = PGMemoryBank(feature_dim=self.vit.config.embed_dim)
        self.lp_reg = lp_reg
        
    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        ids_shuffle = torch.argsort(noise, dim=1) 
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
        
    def patchify(self, imgs):
        p = self.vit.config.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.config.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.config.in_chans))
        return x

    def unpatchify(self, x):
        p = self.vit.config.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.config.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.config.in_chans, h * p, h * p))
        return imgs

    def forward(self, imgs: torch.Tensor, feas = None) -> Dict[str, torch.Tensor]:

        embeddings = None
        imgs = self.input_norm(imgs)
        patches = self.patchify(imgs)
        
        patches_masked, mask, ids_restore = self.random_masking(patches, self.config.mask_ratio)
        patches_masked = self.patch_proj(patches_masked)
        latent = self.vit.forward_features_from_patches(patches_masked, feas, return_all_tokens=True)
        
        if self.vit.config.use_cls_token:
            latent = latent[:, 1:]
        
        pred = self.decoder(latent, ids_restore)
        if self.lp_reg>0:
            embeddings = self.vit(imgs, feas)
        return {
            'patches': patches,  
            'pred': pred,       
            'mask': mask,
            'ids_restore': ids_restore,
            'embeddings': embeddings 
        }
        
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()        
        return loss
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:

        imgs = batch["pixels"]
        feas = batch["domain_feature"]
        # Forward pass
        outputs = self(imgs, feas)
        
        # Compute loss
        main_loss = self.compute_loss(
            outputs['pred'],
            outputs['patches'],
            outputs['mask']
        )
        if self.lp_reg>0:
            embeddings = outputs['embeddings']
            sirna = batch['sirna']
            self.memory_bank.update(embeddings, sirna)
            if (batch_idx + 1 ) % 1 == 0:
                global_reg_loss = self.memory_bank.compute_mixed_loss(self.grloss, embeddings, sirna, imgs.device)
                loss = (1- self.lp_reg) * main_loss + self.lp_reg * global_reg_loss # self.grloss(embeddings, batch_matrix, sirna)
        else: 
            loss = main_loss
        # Convert predictions back to image space
        # pred_imgs = self.unpatchify(outputs['pred'])
        return {
            "loss": loss,
            "mask": outputs['mask']
        }
        
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step - same as training step"""
        imgs = batch["pixels"]
        feas = batch["domain_feature"]
        outputs = self(imgs, feas)
        
        loss = self.compute_loss(
            outputs['pred'],
            outputs['patches'],
            outputs['mask']
        )
        return {
            "loss": loss,
            "mask": outputs['mask']
        }
