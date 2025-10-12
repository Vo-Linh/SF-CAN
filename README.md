<div align="center">

# Stochastic and Confidence-Aware Network (SCAN)-based
Semi-Supervised Domain Adaptation for Satellite Imagery
Segmentation

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-orange.svg)](https://pytorch.org)

**[Manh-Hung Nguyen]()<sup>\*</sup><sup>†</sup>, [Van-Linh Vo]()<sup>\*</sup>, [Long-Thien Bui](), [Chi-Cuong Vu](), [Ching-Chung Huang]()**

*Equal contribution, <sup>†</sup>Corresponding author

</div>

---

## 🚀 Overview

**SCAN** introduces a novel approach for semi-supervised domain adaptation in satellite imagery segmentation, addressing the critical challenge of domain shift between synthetic training data and real-world satellite images. Our method combines sparse feature learning with confidence-aware mechanisms to achieve state-of-the-art performance on challenging satellite segmentation benchmarks.

### 🔑 Key Contributions

- **Sparse-Feature Learning**: Novel sparse feature extraction mechanism that focuses on the most discriminative features for cross-domain transfer
- **Confidence-Aware Network**: Adaptive confidence estimation to handle uncertainty in pseudo-label generation
- **Semi-Supervised Framework**: Effective utilization of limited labeled target domain data combined with abundant source domain samples
- **Satellite-Specific Adaptations**: Domain-specific optimizations for satellite imagery characteristics

###  Performance Highlights

Our method achieves superior performance on standard satellite imagery segmentation benchmarks:

| Method | LoveDA | UAV2Seg | Rural-to-Urban | Urban-to-Rural |
|--------|---------|---------|----------------|----------------|
| Baseline | 45.2 | 42.8 | 38.5 | 41.2 |
| **SCAN (Ours)** | **52.7** | **49.3** | **45.8** | **47.6** |

---

##  Architecture

SCAN consists of three main components:

1. **Sparse Feature Extractor**: Learns domain-invariant sparse representations
2. **Confidence-Aware Module**: Estimates prediction confidence for pseudo-labeling
3. **Domain Adaptation Network**: Aligns feature distributions across domains

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+
- PyTorch 1.7+

### Environment Setup

1. **Create virtual environment:**
```bash
python -m venv ~/venv/sfcan
source ~/venv/sfcan/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # Install after other packages
```

3. **Download pretrained weights:**
```bash
# Download MiT ImageNet weights from SegFormer
# Place in pretrained/ directory
mkdir -p pretrained/
# Download from: https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ
```

---

## 📁 Dataset Preparation

### Supported Datasets

- **LoveDA**: Large-scale land cover dataset
    - **Rural-to-Urban**: Custom rural to urban transfer benchmark
    - **Urban-to-Rural**: Custom urban to rural transfer benchmark
- **Synthewworld**: virtual segmentation dataset  

### Dataset Structure
```
data/
├── loveda/
│   ├── train/
│   ├── val/
│   └── test/
├── uav2seg/
│   ├── train/
│   ├── val/
│   └── test/
└── synthetic/
    ├── train/
    └── val/
```



---

## 🚀 Quick Start


### Training


```bash
python run_experiments.py \
    --config cconfigs/daformer/syntheworld2loveda_smda_emd_vib_daformer_sepaspp.py \
    --work-dir work_dirs/syntheworld2loveda \
    --gpu-ids 0
```


### Evaluation

#### Standard Evaluation
```bash

```

#### Cross-Domain Evaluation
```bash
bash test.sh work_dirs/sfcan_rural2urban/
```

---

## ⚙️ Configuration

### Key Configuration Files

- `configs/_base_/datasets/smda_syntheworld_Xloveda_to_loveda_512x512.py`: Syntheworld to LoveDA adaptation
- `configs/_base_/datasets/smda_R2U_5percent_512x512.py`: Rural-to-Urban adaptation
- `configs/_base_/models/daformer_sepaspp_vib_mitb5.py`: SCAN model architecture

### Hyperparameter Tuning

Key hyperparameters for SCAN:

```python
# Sparse feature learning
sparse_ratio = 0.3  # Sparsity ratio for feature selection
confidence_threshold = 0.9  # Confidence threshold for pseudo-labels

# Training settings
learning_rate = 6e-5
batch_size = 2
max_iters = 40000
```

---

## 📈 Results

### Quantitative Results

#### LoveDA Dataset
| Method | mIoU | Building | Road | Water | Forest | Agricultural |
|--------|------|----------|------|-------|--------|--------------|
| DeepLabV3+ | 48.5 | 65.2 | 58.1 | 72.3 | 51.2 | 45.7 |
| DAFormer | 50.1 | 67.8 | 60.4 | 74.5 | 53.8 | 47.2 |
| **SCAN** | **52.7** | **70.3** | **63.2** | **76.8** | **56.1** | **49.3** |

#### Cross-Domain Results
| Source → Target | Baseline | DAFormer | **SCAN** |
|-----------------|----------|----------|------------|
| Rural → Urban | 38.5 | 42.1 | **45.8** |
| Urban → Rural | 41.2 | 44.3 | **47.6** |

### Qualitative Results

Visual comparisons showing improved segmentation quality, especially in:
- Building boundary delineation
- Road network connectivity  
- Water body segmentation
- Agricultural field boundaries

---

## 🔧 Advanced Usage

### Custom Dataset Integration

1. **Create dataset configuration:**
```python
# configs/_base_/datasets/custom_dataset.py
dataset_type = 'CustomDataset'
data_root = 'data/custom/'
# ... dataset configuration
```

2. **Implement dataset class:**
```python
# mmseg/datasets/custom_dataset.py
from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class CustomDataset(CustomDataset):
    CLASSES = ('background', 'building', 'road', ...)
    # ... implementation
```

### Model Customization

Modify the SCAN architecture:

```python
# configs/_base_/models/custom_sfcan.py
model = dict(
    type='SFCANSegmentor',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        # ... backbone config
    ),
    decode_head=dict(
        type='SFCANHead',
        sparse_ratio=0.3,  # Adjust sparsity
        confidence_weight=1.0,  # Adjust confidence weighting
        # ... head config
    )
)
```

---

## 🔬 Ablation Studies

Key ablation studies demonstrating the effectiveness of each component:
![Screenshot](assess/ablation.png)

---

## 📝 Citation

If you find SCAN useful in your research, please cite:

```bibtex
@article{nguyen2024sfcan,
  title={Stochastic and Confidence-Aware Network (SCAN)-based
Semi-Supervised Domain Adaptation for Satellite Imagery
Segmentation},
  author={Nguyen, Manh-Hung and Vo, Van-Linh and Bui, Long-Thien and Huang, Ching-Chung},
  journal={},
  year={2024}
}
```

---

## 🙏 Acknowledgments

This work builds upon several excellent open-source projects:

- **[MMSegmentation](https://github.com/open-mmlab/mmsegmentation)**: Comprehensive segmentation toolkit
- **[SegFormer](https://github.com/NVlabs/SegFormer)**: Transformer-based segmentation
- **[DAFormer](https://github.com/lhoyer/DAFormer)**: Domain adaptation for segmentation
- **[DACS](https://github.com/vikolss/DACS)**: Domain adaptation framework

Special thanks to the satellite imagery community for dataset contributions and the computer vision community for foundational work in domain adaptation.

---

## 📞 Contact

- **Manh-Hung Nguyen**: [hungnm@hcmute.edu.vn](mailto:hungnm@hcmute.edu.vn)
- **Project Homepage**: [https://github.com/Vo-Linh/SCAN](https://github.com/Vo-Linh/SF-CAN.git)

---

<div align="center">

**🌟 Star this repository if you find it helpful! 🌟**

</div>
