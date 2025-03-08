# ICCV25 Submission

Source code for Integrating Biological Knowledge for Robust Microscopy Image Profiling on *De Novo* Cell Lines



##  Environments
This work requires anaconda with python 3.10 or later, cudatoolkit=12.4 and below packages
```  
timm==0.9.16        
pytorch>=2.4.0+cu124 
torchvision>=0.19.0+cu124
transformers==4.45.2
wandb
```

## Image Data

For RxRx1 and RxRx19a, please download from https://www.rxrx.ai/datasets (including metadata, features, and raw image)

## scFM

scGPT: https://github.com/bowang-lab/scGPT

scFoundation: https://github.com/biomap-research/scFoundation

cell line gene expression data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE288929

## Pretraining

Using the folowing commands for pretraining vit (DDP).
```
python pretrain_ssl.py --ssl_model simclr
python pretrain_ssl.py --ssl_model byol
python pretrain_ssl.py --ssl_model bino
torchrun --nproc_per_node=8 pretrain_wsl.py
torchrun --nproc_per_node=8 pretrain_mae.py
```

## Fine-tuning

```
python downstream.py
```