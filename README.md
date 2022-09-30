# Project Structure
#TODO

# Setup
# Training

```bash
conda create -n <name>
conda install python=3.9

pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
```
# Evalution

```bash
conda create -n <name>
conda install python=3.9

pip3 install torch
pip3 install torchvision
pip3 install comet-ml
```
## Backbone Resnet Pretraining Informations

Except otherwise noted, all models have been trained on 8x V100 GPUs with the following parameters.

| Parameter                | value  |
| ------------------------ | ------ |
| `--batch_size`           | `32`   |
| `--epochs`               | `90`   |
| `--lr`                   | `0.1`  |
| `--momentum`             | `0.9`  |
| `--wd`, `--weight-decay` | `1e-4` |
| `--lr-step-size`         | `30`   |
| `--lr-gamma`             | `0.1`  |