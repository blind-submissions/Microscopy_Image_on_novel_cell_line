import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Dict, Tuple, List, Optional, Union
from enum import Enum


class MLP(nn.Module):    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 norm_layer: bool = True, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_dim))
        if norm_layer:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if norm_layer:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        if num_layers > 1:
            layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DINOHead(nn.Module):    
    def __init__(self, in_dim, hidden_dim, out_dim, norm_last_layer=True, bottleneck_dim=256):
        super().__init__()
        self.norm_last_layer = norm_last_layer
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        
        if norm_last_layer:
            self.last_layer.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        
        if self.norm_last_layer:
            w = self.last_layer.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            x = F.linear(x, w)
        else:
            x = self.last_layer(x)
        
        return x

class BYOLModel(nn.Module):    
    def __init__(
        self, 
        encoder: nn.Module,
        embed_dim: int,
        projector_dim: int = 2048, 
        projector_hidden_dim: int = 4096,
        predictor_hidden_dim: int = 1024,
        target_momentum: float = 0.996,
        supervised_loss_fn = None,
        supervised_weight: float = 0.0,
    ):
        super().__init__()
        self.online_encoder = encoder
        
        self.online_projector = MLP(
            input_dim=embed_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_dim
        )
        
        self.online_predictor = MLP(
            input_dim=projector_dim,
            hidden_dim=predictor_hidden_dim,
            output_dim=projector_dim
        )
        
        self.target_encoder = deepcopy(encoder)
        self.target_projector = deepcopy(self.online_projector)
        self._stop_gradient(self.target_encoder)
        self._stop_gradient(self.target_projector)
        
        self.target_momentum = target_momentum
        
        self.supervised_loss_fn = supervised_loss_fn
        self.supervised_weight = supervised_weight
    
    def _stop_gradient(self, network: nn.Module):
        for param in network.parameters():
            param.requires_grad = False
    
    def _update_target_network(self):
        for online_param, target_param in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_param.data = (
                self.target_momentum * target_param.data 
                + (1 - self.target_momentum) * online_param.data
            )
        
        for online_param, target_param in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            target_param.data = (
                self.target_momentum * target_param.data 
                + (1 - self.target_momentum) * online_param.data
            )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        view1 = batch["view1"]
        view2 = batch["view2"]
        domain_features = batch.get("domain_feature", None)
        batch_matrix = batch.get("batch_matrix", None)
        
        kwargs = {"domain_features": domain_features} if domain_features is not None else {}
        embed1 = self.online_encoder(view1, **kwargs)
        proj1 = self.online_projector(embed1)
        pred1 = self.online_predictor(proj1)
        
        with torch.no_grad():
            embed2 = self.target_encoder(view2, **kwargs)
            proj2 = self.target_projector(embed2)
        
        embed2_online = self.online_encoder(view2, **kwargs)
        proj2_online = self.online_projector(embed2_online)
        pred2 = self.online_predictor(proj2_online)
        
        with torch.no_grad():
            embed1_target = self.target_encoder(view1, **kwargs)
            proj1_target = self.target_projector(embed1_target)
        
        byol_loss1 = self._byol_loss(pred1, proj2)
        byol_loss2 = self._byol_loss(pred2, proj1_target)
        byol_loss = (byol_loss1 + byol_loss2) / 2
        
        loss = byol_loss
        losses_dict = {"byol_loss": byol_loss}
        
        imgs = batch['imgs']
        embeddings = self.online_encoder(imgs, domain_features)
        if self.supervised_weight >0  and batch_matrix is not None:
            sup_loss = self.supervised_loss_fn(embeddings, batch_matrix)
            loss = loss + self.supervised_weight * sup_loss
            losses_dict["supervised_loss"] = sup_loss
            losses_dict["total_loss"] = loss
        
        self._update_target_network()
        
        return {"loss": loss, **losses_dict}
    
    def _byol_loss(self, online_pred: torch.Tensor, target_proj: torch.Tensor) -> torch.Tensor:
        online_pred = F.normalize(online_pred, dim=-1, p=2)
        target_proj = F.normalize(target_proj, dim=-1, p=2)
        
        return 2 - 2 * (online_pred * target_proj).sum(dim=-1).mean()
    
    def get_embedding(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.online_encoder(x, **kwargs)


class SimCLRModel(nn.Module):
    
    def __init__(
        self, 
        encoder: nn.Module,
        embed_dim: int,
        projector_dim: int = 128, 
        projector_hidden_dim: int = 2048,
        temperature: float = 0.1,
        supervised_loss_fn = None,
        supervised_weight: float = 0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        
        self.projector = MLP(
            input_dim=embed_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_dim
        )
        
        self.supervised_loss_fn = supervised_loss_fn
        self.supervised_weight = supervised_weight
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        view1 = batch["view1"]
        view2 = batch["view2"]
        domain_features = batch.get("domain_feature", None)
        batch_matrix = batch.get("batch_matrix", None)
        
        kwargs = {"domain_features": domain_features} if domain_features is not None else {}
        embed1 = self.encoder(view1, **kwargs)
        embed2 = self.encoder(view2, **kwargs)
        
        z1 = self.projector(embed1)
        z2 = self.projector(embed2)
        
        simclr_loss = self._simclr_loss(z1, z2, self.temperature)
        
        loss = simclr_loss
        losses_dict = {"simclr_loss": simclr_loss}
        
        imgs = batch['imgs']
        embeddings = self.encoder(imgs, domain_features)
        if self.supervised_weight >0  and batch_matrix is not None:
            sup_loss = self.supervised_loss_fn(embeddings, batch_matrix)
            loss = loss + self.supervised_weight * sup_loss
            losses_dict["supervised_loss"] = sup_loss
            losses_dict["total_loss"] = loss
        
        return {"loss": loss, **losses_dict}
    
    def _simclr_loss(self, z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        batch_size = z_i.size(0)
        
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        representations = torch.cat([z_i, z_j], dim=0)
        
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device))
        
        for i in range(batch_size):
            mask[i, i + batch_size] = 0
            mask[i + batch_size, i] = 0
        
        negatives = similarity_matrix[mask].view(batch_size * 2, -1)
        
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1) / temperature
        
        labels = torch.zeros(batch_size * 2, dtype=torch.long, device=z_i.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def get_embedding(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.encoder(x, **kwargs)


class DINOModel(nn.Module):
    
    def __init__(
        self, 
        encoder: nn.Module,
        embed_dim: int,
        num_prototypes: int = 32768,
        projector_hidden_dim: int = 2048,
        temperature_teacher: float = 0.04,
        temperature_student: float = 0.1,
        target_momentum: float = 0.996,
        supervised_loss_fn = None,
        supervised_weight: float = 0.0,
    ):
        super().__init__()
        self.student_encoder = encoder
        
        self.student_head = DINOHead(
            in_dim=embed_dim,
            hidden_dim=projector_hidden_dim,
            out_dim=num_prototypes,
            bottleneck_dim=256
        )
        
        self.teacher_encoder = deepcopy(encoder)
        self.teacher_head = DINOHead(
            in_dim=embed_dim,
            hidden_dim=projector_hidden_dim,
            out_dim=num_prototypes,
            bottleneck_dim=256,
            norm_last_layer=True
        )
        
        self._stop_gradient(self.teacher_encoder)
        self._stop_gradient(self.teacher_head)
        
        self.register_buffer("center", torch.zeros(1, num_prototypes))
        
        self.teacher_temp = temperature_teacher
        self.student_temp = temperature_student
        
        self.target_momentum = target_momentum
        
        self.supervised_loss_fn = supervised_loss_fn
        self.supervised_weight = supervised_weight
    
    def _stop_gradient(self, network: nn.Module):
        for param in network.parameters():
            param.requires_grad = False
    
    def _update_teacher(self):
        for student_param, teacher_param in zip(
            self.student_encoder.parameters(), self.teacher_encoder.parameters()
        ):
            teacher_param.data = (
                self.target_momentum * teacher_param.data 
                + (1 - self.target_momentum) * student_param.data
            )
        
        for student_param, teacher_param in zip(
            self.student_head.parameters(), self.teacher_head.parameters()
        ):
            teacher_param.data = (
                self.target_momentum * teacher_param.data 
                + (1 - self.target_momentum) * student_param.data
            )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        global_views = batch["global_views"]
        local_views = batch["local_views"]
        domain_features = batch.get("domain_feature", None)
        batch_matrix = batch.get("batch_matrix", None)
        
        kwargs = {"domain_features": domain_features} if domain_features is not None else {}
        
        all_views = global_views + local_views
        
        student_outputs = []
        student_embeddings = []
        
        for view in all_views:
            embed = self.student_encoder(view, **kwargs)
            student_embeddings.append(embed)
            output = self.student_head(embed)
            student_outputs.append(output)
        
        teacher_outputs = []
        with torch.no_grad():
            for view in global_views:
                embed = self.teacher_encoder(view, **kwargs)
                output = self.teacher_head(embed)
                teacher_outputs.append(output)
        
        dino_loss = self._dino_loss(student_outputs, teacher_outputs, global_views)
        
        loss = dino_loss
        losses_dict = {"dino_loss": dino_loss}
        
        if self.supervised_weight >0 and batch_matrix is not None:
            sup_loss = self.supervised_loss_fn(student_embeddings[0], batch_matrix)
            loss = loss + self.supervised_weight * sup_loss
            losses_dict["supervised_loss"] = sup_loss
            losses_dict["total_loss"] = loss
        
        self._update_teacher()
        
        return {"loss": loss, **losses_dict}
    
    def _dino_loss(self, student_outputs, teacher_outputs, global_views):
        total_loss = 0
        n_loss_terms = 0
        
        for i, student_out in enumerate(student_outputs):
            for j, teacher_out in enumerate(teacher_outputs):
                if i == j and i < len(global_views):
                    continue
                
                center = self.center.clone().detach()
                
                teacher_out = teacher_out - center
                
                student_out = student_out / self.student_temp
                teacher_out = F.softmax(teacher_out / self.teacher_temp, dim=-1)
                
                loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1).mean()
                
                total_loss += loss
                n_loss_terms += 1
        
        self.center = self.center * 0.9 + torch.mean(teacher_outputs[0].detach(), dim=0, keepdim=True) * 0.1
        
        total_loss /= n_loss_terms
        
        return total_loss
    
    def get_embedding(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.student_encoder(x, **kwargs)


def create_ssl_model(
    method,
    encoder: nn.Module,
    embed_dim: int,
    **kwargs
) -> nn.Module:
    if method == 'byol':
        return BYOLModel(encoder, embed_dim, **kwargs)
    
    elif method == 'simclr':
        return SimCLRModel(encoder, embed_dim, **kwargs)
    
    elif method == 'dino':
        return DINOModel(encoder, embed_dim, **kwargs)
    
    else:
        raise ValueError(f"Unsupported method: {method}")