# DiGS-CP


[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-red.svg)](https://aaai.org/) 
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**DiGS-CP** is the official implementation of the paper *"From Discriminative to Generative: A Diffusion-Based Paradigm for Multi-Agent Collaborative Perception"*, accepted by **AAAI 2026**.

We propose DiGS-CP, a novel framework that shifts the collaborative perception paradigm from discriminative to generative. Unlike traditional methods that learn minimal task-specific features, DiGS-CP leverages a conditional diffusion model to provide fine-grained, task-agnostic supervision, encouraging the learning of comprehensive geometric representations. Combined with an efficient two-stage object-level transmission strategy, our method achieves state-of-the-art performance with low communication overhead.

<p align="center">
  <img src="fig/overview.svg" width="800"/>
  <br/>
  <em>Figure 1: Overview of DiGS-CP. During training, the diffusion head acts as a supervisor to refine fused features. During inference, the diffusion branch is removed, ensuring high efficiency. </em>
</p>

## 📰 News
*   **[2026-11-08]** DiGS-CP is accepted to **AAAI 2026**! 🎉
*   **[2026-01-06]** Code and pre-trained models are released.
## 🛠 Installation

you can refer to [OpenCOOD data introduction](https://opencood.readthedocs.io/en/latest/md_files/data_intro.html)
and [OpenCOOD installation](https://opencood.readthedocs.io/en/latest/md_files/installation.html) guide to prepare
data and install DiGS-CP. The installation is totally the same as OpenCOOD, except some dependent packages required by DiGS-CP.

## 📂 Data Preparation
mkdir a `dataset` folder under DiGS-CP. Put your OPV2V, V2XSet, DAIR-V2X data in this folder. You just need to put in the dataset you want to use.

```
DiGS-CP/dataset

. 
├── dair_v2x 
│   ├── v2x_c
│   ├── v2x_i
│   └── v2x_v
├── OPV2V
│   ├── additional
│   ├── test
│   ├── train
│   └── validate
└── V2XSET
    ├── test
    ├── train
    └── validate

```
## 🚀 Quick Start
1. Training

To train DiGS-CP on OPV2V dataset:
```bash
python opencood/train.py --hypes_yaml hypes_yaml/opv2v/lidar_only_with_noise/diffusion/pointpillar_diff.yaml --model_dir checkpoints/opv2v/OPV2V_best_epoch.pth
```
2. Evaluation

To evaluate the model performance:
```bash
python -u opencood/tools/inference_models.py
```

## 🦁 Model Zoo

We provide pre-trained checkpoints for DiGS-CP. 

**Usage:**
1. Create a `checkpoints` folder in the root directory.
2. Download the models and extract them into their respective subfolders as shown below:

```text
DiGS-CP/
└── checkpoints/
    ├── opv2v/          <-- Put OPV2V model files here
    ├── v2xset/         <-- Put V2XSet model files here
    └── dair_v2x/       <-- Put DAIR-V2X model files here
```

| Dataset | AP@0.5 | AP@0.7 | Download |
| :--- | :---: | :---: | :---: |
| **OPV2V** | 96.75% | 93.53% | [![Download](https://img.shields.io/badge/Model-Download-blue?logo=google-drive&logoColor=white)](https://drive.google.com/file/d/1KnUntsF32tfyI-fwK37A2IETbGxjL4D4/view?usp=drive_link) |
| **V2XSet** | 90.40% | 83.01% | [![Download](https://img.shields.io/badge/Model-Download-blue?logo=google-drive&logoColor=white)](https://drive.google.com/file/d/19LT6Ydzl1PCDDoa9f8CiD90hQVdqo0JA/view?usp=drive_link) |
| **DAIR-V2X** | 79.18% | 64.86% | [![Download](https://img.shields.io/badge/Model-Download-blue?logo=google-drive&logoColor=white)](https://drive.google.com/file/d/1h41b8uhJFvoOyGEReOYXsagQqqeBbLUW/view?usp=drive_link) |


## 📝 Citation
If you find this work useful for your research, please cite our paper.


