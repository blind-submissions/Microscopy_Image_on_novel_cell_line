import torch
import torch.nn as nn
import torchvision.models as models
from typing import Type, Union, List
import torch
import torch.nn as nn
from torchvision.models import densenet161
from typing import Dict
from transformers import PretrainedConfig, PreTrainedModel

class Normalizer(torch.nn.Module):
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels
        return pixels / 255.0
    
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, domain_feature_dim: int, cond_strength: float = 1.0):
        super().__init__()
        self.num_features = num_features
        self.cond_strength = cond_strength
        

        self.bn = nn.BatchNorm2d(num_features, affine=True)  

    def forward(self, x: torch.Tensor, domain_features: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        
        return out
        

class ConditionalBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Union[nn.Module, None] = None,
        domain_feature_dim: int = 64,
        cond_strength: float = 1.0,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = ConditionalBatchNorm2d(planes, domain_feature_dim, cond_strength)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = ConditionalBatchNorm2d(planes, domain_feature_dim, cond_strength)
        
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = ConditionalBatchNorm2d(planes * self.expansion, domain_feature_dim, cond_strength)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor, domain_features: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, domain_features)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, domain_features)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, domain_features)

        if self.downsample is not None:
            identity = self.downsample[0](x)
            identity = self.downsample[1](identity, domain_features)

        out += identity
        out = self.relu(out)

        return out

class ConditionalResNet(nn.Module):
    def __init__(
        self,
        domain_feature_dim: int = 64,
        cond_strength: float = 1.0,

    ):
        super().__init__()
        
        self.domain_feature_dim = domain_feature_dim
        self.cond_strength = cond_strength
        
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = ConditionalBatchNorm2d(64, domain_feature_dim, cond_strength)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.input_norm = torch.nn.Sequential(
            Normalizer(),
            nn.InstanceNorm2d(
                num_features=6,
                affine=False, 
                track_running_stats=False
            )
        )
        
    def _make_layer(
        self,
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes * ConditionalBottleneck.expansion:
            downsample = nn.ModuleList([
                nn.Conv2d(
                    inplanes,
                    planes * ConditionalBottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                ConditionalBatchNorm2d(
                    planes * ConditionalBottleneck.expansion,
                    self.domain_feature_dim,
                    self.cond_strength,
                ),
            ])

        layers = []
        layers.append(
            ConditionalBottleneck(
                inplanes,
                planes,
                stride,
                downsample,
                self.domain_feature_dim,
                self.cond_strength,
            )
        )
        
        inplanes = planes * ConditionalBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(
                ConditionalBottleneck(
                    inplanes,
                    planes,
                    domain_feature_dim=self.domain_feature_dim,
                    cond_strength=self.cond_strength,
                )
            )

        return nn.ModuleList(layers)
    
    @property
    def embed_dim(self) -> int:
        return 2048  
    
    def forward(self, x: torch.Tensor, domain_features: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.conv1(x)
        x = self.bn1(x, domain_features)
        x = self.relu(x)
        x = self.maxpool(x)

        for block in self.layer1:
            x = block(x, domain_features)
        for block in self.layer2:
            x = block(x, domain_features)
        for block in self.layer3:
            x = block(x, domain_features)
        for block in self.layer4:
            x = block(x, domain_features)

        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x.view(x.size(0), -1)
        # return x

def load_imagenet_weights(model: ConditionalResNet) -> None:
    pretrained_model = models.resnet50(pretrained=True)

    original_weight = pretrained_model.conv1.weight.data
    model.conv1.weight.data[:, :3, :, :] = original_weight
    model.conv1.weight.data[:, 3:, :, :] = original_weight  
    
    for i in range(1, 5):
        layer_name = f'layer{i}'
        pretrained_layer = getattr(pretrained_model, layer_name)
        conditional_layer = getattr(model, layer_name)
        
        for j, (pretrained_block, conditional_block) in enumerate(zip(pretrained_layer, conditional_layer)):
            conditional_block.conv1.weight.data = pretrained_block.conv1.weight.data
            conditional_block.conv2.weight.data = pretrained_block.conv2.weight.data
            conditional_block.conv3.weight.data = pretrained_block.conv3.weight.data
            
            if conditional_block.downsample is not None:
                conditional_block.downsample[0].weight.data = pretrained_block.downsample[0].weight.data
    



class ResNetWSLConfig(PretrainedConfig):
    model_type = "ResNetWSL"
    
    def __init__(
        self,
        num_classes: int = 1000,
        optimizer=None,
        input_norm=None,
        lr_scheduler=None,
        domain_feature_dim = 0,
        cond_strength = 0.,
        lp_reg = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.input_norm = input_norm
        self.lr_scheduler = lr_scheduler
        self.domain_feature_dim = domain_feature_dim
        self.cond_strength = cond_strength
        self.lp_reg = lp_reg
        
