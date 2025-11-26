# ARE-stereo: Real-time Stereo 3D Object Detection with Adaptive ROI Enhancement


ARE-stereo is a real-time 3D object detection model designed for urban rail transit scenarios. It adopts a monocular-stereo cascaded architecture and introduces two key strategies: **Pseudo Disparity Enhancement (PDE)** and **Adaptive ROI Augmentation (ARA)**. The model significantly improves detection performance for distant, small, and occluded objects while maintaining real-time efficiency, and has been validated through deployment in real-world tram scenarios.

## 🌟 Highlights

- Real-time performance: 70–90 ms inference latency on a single NVIDIA 2080 Ti GPU.  
- Higher accuracy: Outperforms existing real-time models on both the KITTI benchmark and our self-constructed DL-Stereo dataset.
-  PDE: Recovers suppressed regions of interest (distant or occluded small objects).
- ARA: Adaptively augments 3D ROIs to prevent loss of stereo features.
- Embedded deployable: Validated with TensorRT on Xavier / Orin, meeting real-time requirements for rail transit applications.
- Practical application: Tested on the real-world Dalian 202 tram line.

## 🚀 Model Architecture Overview

![图片说明](./images/fig2.png)  

The model consists of three core modules:

### 1. Feature Extractor
- The left and right views independently output their respective feature maps.
- prediction：2D center、offset、shape、depth、orientation

### 2. ROI Generator + PDE
- PDE leverages pseudo disparity inferred from the right-view disparity
- Warps right-view ROIs onto the left-view feature maps 
- Recovers distant/small-object ROIs whose confidences have been suppressed, improving recall

### 3. Stereo Detector + ARA
- Constructs local FCE (Feature Consistency Embedding) volumes
- ARA method augments ROIs according to depth uncertainty
- The stereo head predicts the final 3D refinement

---

# 📊 Experimental Results

## 1. KITTI 3D Detection Benchmark（Car, IoU=0.7）

### 1.1 Comparison with models that use additional supervision

| Model | Time ↓ | AP3D / APBEV (Val: Easy/Mod./Hard) | AP3D (Test: Easy/Mod./Hard) | Training Dependency |
|------|-------|--------------------------------------|-------------------------------|----------------------|
| OC-Stereo | 350ms | 64.07/77.66, 48.34/65.95, 40.39/51.20 | 55.15, 37.60, 30.25 | Depth, Mask |
| YOLOStereo3D | 160ms | 73.85/78.37, 48.62/57.58, 39.33/42.35 | 65.68, 41.25, 30.42 | Depth |
| TAG | 160ms | 74.36/79.08, 49.01/58.00, 40.99/43.65 | – | Depth |
| TS3D | 180ms | 70.90/–, 46.76/–, 35.94/– | 64.61, 41.20, 30.68 | Depth |
| OFSCF | – | 65.72/73.33, 45.60/47.26, 39.39/40.94 | – | Depth |
| **ARE-stereo (aux depth-head)** | **77ms** | **74.08/78.47, 49.45/58.37, 40.17/43.29** | **66.05, 41.71, 30.64** | Depth |

---

## 2. DL-Stereo Dataset


| Model | Time ↓ | Supervision | AP3D Car | AP3D Ped | AP3D Truck | AP3D Bus | APBEV Car | APBEV Ped | APBEV Truck | APBEV Bus |
|-------|--------|-------------|----------|----------|------------|-----------|------------|------------|--------------|------------|
| OC-Stereo | 362ms | Depth | 50.36 | 41.72 | 50.47 | 67.15 | 72.15 | 54.63 | 66.62 | 75.11 |
| YOLOStereo3D | 169ms | Depth | 57.49 | 45.62 | 55.33 | 71.88 | 69.17 | 52.03 | 65.12 | 73.69 |
| **ARE-stereo (aux depth-head)** | **92ms** | Depth | **57.62** | **46.38** | **55.58** | **72.49** | **69.45** | **52.82** | **65.33** | **74.88** |
| SIDE | 214ms | None | 48.89 | 40.66 | 47.29 | 63.39 | 61.44 | 48.48 | 59.93 | 68.66 |
| FGAS-RCNN | 114ms | None | 50.72 | 41.23 | 49.88 | 65.63 | 62.18 | 49.39 | 60.92 | 70.79 |
| RTS3D | 91ms | None | 50.98 | 43.15 | 49.51 | 66.34 | 63.79 | 50.29 | 61.70 | 70.71 |
| SAS3D | 78ms | None | 51.62 | 42.97 | 50.07 | 67.63 | 64.62 | 50.18 | 62.12 | 71.02 |
| **ARE-stereo** | **92ms** | None | **52.14** | **43.41** | **50.26** | **68.12** | **65.35** | **51.61** | **62.15** | **72.68** |

---

## 3. Ablation Study

### 3.1 PDE / ARA

| Baseline | PDE | ARA | Time ↓ | AP3D | APBEV |
|----------|-----|------|--------|-------|--------|
| ✓ | – | – | 72ms | 51.32 | 63.84 |
| ✓ | ✓ | – | 76ms | 51.57 | 64.27 |
| ✓ | ✓ | ✓ | 92ms | **52.14** | **65.35** |

### 3.2 Effects of different monocular stage

| Mono Model | Stereo Time | Mono AP3D | Stereo AP3D | Mono APBEV | Stereo APBEV |
|------------|-------------|-----------|--------------|--------------|----------------|
| KM3D | 91ms | 21.71 | 51.37 | 32.45 | 64.60 |
| MonoCon | 86ms | 22.26 | 51.65 | 33.38 | 64.53 |
| MonoLSS | 96ms | 22.79 | 51.89 | 33.66 | 65.03 |
| **ARE-stereo** | **92ms** | 22.17 | **52.14** | 32.97 | **65.35** |

---

# Inference Demo

```bash
python demo.py --model_path checkpoints/are_stereo.pth \
               --left_img demo/000001_left.png \
               --right_img demo/000001_right.png
```

# Training

<!-- ## Step 1 — Train Monocular Stage
```bash
python train_mono.py --cfg configs/mono.yaml
``` -->

## Train Stereo Stage
```bash
python main.py 
```
## DL-stereo dataset
> ⚠️
> Due to privacy policies, only a subset of the data is currently released; the full version will be made available after the paper is officially accepted. If you require early access, please contact the authors.

# 🙏 Acknowledgements
- [**CenterNet**](https://github.com/xingyizhou/CenterNet)
- [**RTM3D**](https://github.com/Banconxuan/RTM3D)
- [**RTS3D** ](https://github.com/Banconxuan/RTS3D)
