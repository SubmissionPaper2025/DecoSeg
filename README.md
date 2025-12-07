<div align="center">
<h1> DecoSeg: Robust Medical Segmentation Framework via Contrast-enhanced Feature Decoupling </h1>
</div>

## 🎈 News

- [2025.2.19] Training and inference code released

## ⭐ Abstract

Medical image segmentation plays a significant role in treatment planning and disease tracking. However, it faces two key challenges: one is the fuzzy transition region (soft boundary) between foreground and background, which is exacerbated by low contrast; the other is the misleading co-occurrence of salient and non-salient objects, which affects the accuracy of the model in extracting key segmentation features. To overcome these challenges, we introduce DecoSeg, a new framework designed to enhance medical image segmentation. DecoSeg integrates Feature Decoupling Unit (FDU), which dynamically separates the encoded features into foreground, background, and uncertain regions, and uses an uncertainty loss to reduce the interference of soft boundaries. Plus, our Contrast-driven Feature Fusion Unit (CFFU) further leverages the foreground, background, and uncertain regions from the FDU to distinguish salient objects from non-salient ones, thereby optimizing the accuracy of detecting salient objects in scenes with misleading co-occurrences. DecoSeg is comprehensively evaluated on 5 different medical image datasets, verifying its superior performance, demonstrating its significant value in the field of medical image segmentation.

## 🚀 Introduction

<div align="center">
    <img width="400" alt="image" src="figures/challenge1.png?raw=true">
</div>

The challenges: (a) There is a blurred transition region (soft boundary) between the foreground and background. (b) salient objects often coexist with non-salient objects (misleading co-occurrence phenomenon).

## 📻 Overview

<div align="center">
<img width="800" alt="image" src="figures/network1.png?raw=true">
</div>

Illustration of the overall architecture of DecoSeg.


## 📆 TODO

- [x] Release code

## 🎮 Getting Started

### 1. Install Environment

```
conda create -n Net python=3.8
conda activate Net
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs PyWavelets
```

### 2. Prepare Datasets

- Download datasets: ISIC2018 from this [link](https://challenge.isic-archive.com/data/#2018).


- Folder organization: put datasets into ./data/datasets folder.

### 3. Train the DecoNet

```
python train.py --datasets ISIC2018
```

### 4. Test the DecoNet

```
python test.py --datasets ISIC2018
```

## 🖼️ Visualization

<div align="center">
<img width="800" alt="image" src="figures/com_pic.png?raw=true">
</div>

<div align="center">
We compare our method against 13 state-of-the-art methods. The red box indicates the area of incorrect predictions.
</div>

## ✨ Quantitative comparison

<div align="center">
<img width="800" alt="image" src="figures/com_tab.png?raw=true">
</div>

<div align="center">
Performance comparison with ten SOTA methods on ISIC2018, Kvasir, BUSI, COVID-19 and Monu-Seg datasets.
</div>
