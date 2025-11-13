# Structured Slot Aggregation for Cross-Modal Video Object Segmentation

This repository contains a complete **reimplementation of the Guided Slot Attention Network (GSANet)** for **unsupervised video object segmentation (U-VOS)**. The project follows the architecture introduced in the CVPR 2024 paper [*“Guided Slot Attention for Unsupervised Video Object Segmentation” (Lee et al.)*](https://github.com/Hydragon516/GSANet) and rebuilds the full system end-to-end.

## Overview

<p align="center">
  <img src="assets/cows_triple.gif" width="600"/>
</p>

## Dataset

We use the [DUTS](https://saliencydetection.net/duts/) train dataset for model pretraining and [DAVIS 2016](https://davischallenge.org/davis2016/code.html) train dataset for finetuning and test dataset for evaluation. The optical flow maps for DAVIS 2016 were generated using [RAFT](https://github.com/princeton-vl/RAFT). The Final dataset structure is:

```bash
dataset/
├── DUTS_train/
│   ├── RGB/
│   │   ├── sun_ekmqudbbrseiyiht.jpg
│   │   ├── sun_ejwwsnjzahzakyjq.jpg
│   │   └── ...
│   └── GT/
│       ├── sun_ekmqudbbrseiyiht.png
│       ├── sun_ejwwsnjzahzakyjq.png
│       └── ...
│
├── DAVIS_train/
│   ├── RGB/
│   │   ├── bear_00000.jpg
│   │   ├── bear_00001.jpg
│   │   └── ...
│   ├── GT/
│   │   ├── bear_00000.png
│   │   ├── bear_00001.png
│   │   └── ...
│   └── FLOW/
│       ├── bear_00000.jpg
│       ├── bear_00001.jpg
│       └── ...
│
└── DAVIS_test/
    ├── blackswan/
    │   ├── RGB/
    │   │   ├── blackswan_00000.jpg
    │   │   │   ...
    │   ├── GT/
    │   │   ├── blackswan_00000.png
    │   │   │   ...
    │   └── FLOW/
    │       ├── blackswan_00000.jpg
    │       │   ...
    │
    ├── bmx-trees/
    └── ...
```

## Training Model

We use a two-stage learning strategy: **pretraining** and **finetuning**.

### **Pretraining**
Edit `config.py`.  
Modify the **data root path** and **GPU index** and run `pretrain.py`.

### **Finetuning**
Edit `config.py`.  
Set the **best model path** generated during pretraining and run `train.py`

## Result

Below is an example visualization from the **DAVIS-2016 \`cows\` sequence**, showing the model’s prediction, slot activations, and input modalities. Each grid contains **8 panels (A–H)** illustrating how GSANet interprets the scene.

<p align="center">
  <img src="assets/cows_visual_grid_00000.jpg" width="600"/>
</p>

### **Panel Descriptions**

- **A - RGB Frame**  
  The original input frame from the DAVIS sequence.

- **B - Optical Flow**  
  RAFT-generated flow showing motion cues in the scene.

- **C - Predicted Mask**  
  Final segmentation output after multi-scale decoding.

- **D - Ground Truth Mask**  
  DAVIS-provided per-frame annotation.

- **E - RGB Slot (Foreground)**  
  Slot Attention foreground activation derived from the RGB encoder.

- **F - RGB Slot (Background)**  
  Background slot activation from the RGB encoder.

- **G - Flow Slot (Foreground)**  
  Foreground slot activation derived from the motion encoder.

- **H - Flow Slot (Background)**  
  Background slot activation from optical flow.

### Quantitative Results (DAVIS-2016)

| Metric | Score |
|--------|--------|
| **Mean IoU (J-Mean)** | 0.8518 |
| **Mean Boundary F (F-Mean)** | 0.8400 |
| **J Recall** | 0.89 |
| **Temporal Decay (J & F)** | ~0.02 |


## Citation

If you use this reimplementation, please cite the original GSANet paper:

```bibtex
@InProceedings{Lee_2024_CVPR,
    author    = {Lee, Minhyeok and Cho, Suhwan and Lee, Dogyoon and Park, Chaewon and Lee, Jungho and Lee, Sangyoun},
    title     = {Guided Slot Attention for Unsupervised Video Object Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {3807-3816}
}
```

## Course   
- **Purdue University** – ECE 59500CV: Deep Learning for Computer Vision course 
