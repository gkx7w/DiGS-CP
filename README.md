# SDCoper

Truth is One:  Self-Distilled Collaborative Perception with Explicit Knowledge Extraction

## Installation

you can refer to [OpenCOOD data introduction](https://opencood.readthedocs.io/en/latest/md_files/data_intro.html)
and [OpenCOOD installation](https://opencood.readthedocs.io/en/latest/md_files/installation.html) guide to prepare
data and install SDCoper. The installation is totally the same as OpenCOOD, except some dependent packages required by SDCoper.

## Data Preparation
mkdir a `dataset` folder under SDCoper. Put your OPV2V, V2X-Sim, V2XSet, DAIR-V2X data in this folder. You just need to put in the dataset you want to use.

```
SDCoper/dataset

. 
├── my_dair_v2x 
│   ├── v2x_c
│   ├── v2x_i
│   └── v2x_v
├── OPV2V
│   ├── additional
│   ├── test
│   ├── train
│   └── validate
├── V2XSET
│   ├── test
│   ├── train
│   └── validate
├── v2xsim2-complete
│   ├── lidarseg
│   ├── maps
│   ├── sweeps
│   └── v1.0-mini
└── v2xsim2_info
    ├── v2xsim_infos_test.pkl
    ├── v2xsim_infos_train.pkl
    └── v2xsim_infos_val.pkl
```
