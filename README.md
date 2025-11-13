# Structured Slot Aggregation for Cross-Modal Video Object Segmentation

This repository contains a complete **reimplementation of the Guided Slot Attention Network (GSANet)** for **unsupervised video object segmentation (U-VOS)**.  
The project follows the architecture introduced in the CVPR 2024 paper *“Guided Slot Attention for Unsupervised Video Object Segmentation” (Lee et al.)* and rebuilds the full system end-to-end.

This implementation includes:

- Saliency-based pretraining using DUTS
- RGB + Optical Flow dual-stream video encoder using SegFormer-B2
- K-Means and slot-based prototype extraction
- Feature Aggregation Transformer (FAT) for cross-modal feature interaction
- Triple-iterative Slot Attention refinement for temporal consistency
- Multi-scale decoder for segmentation prediction
- Complete evaluation pipeline for DAVIS (J & F metrics)

This repo provides a fully functional, research-friendly reproduction of GSANet that can be trained, evaluated, and extended for further study in video segmentation and slot-based representation learning.

## Dataset

We use the [DUTS](https://saliencydetection.net/duts/) train dataset for model pretraining and [DAVIS 2016](https://davischallenge.org/davis2016/code.html) train dataset for finetuning. The optical flow maps for DAVIS 2016 were generated using [RAFT](https://github.com/princeton-vl/RAFT). The Final dataset structure is:

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



