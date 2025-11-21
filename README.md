# ARE-Stereo: Stereo Real-time 3D Object Detection with Adaptive ROI Enhancement for Rail Transit

This repository contains the official implementation of:

> **Stereo Real-time 3D Object Detection with Adaptive ROI Enhancement for Rail Transit** :contentReference[oaicite:0]{index=0}

ARE-Stereo is a real-time stereo 3D object detector designed for **urban rail transit** environments.  
It uses a **monocular–stereo cascaded framework** and introduces two key strategies:

- **Pseudo Disparity Enhancement (PDE)** – restores suppressed ROIs using pseudo disparity.  
- **Adaptive ROI Augmentation (ARA)** – expands ROIs based on predicted depth and uncertainty to preserve useful stereo features. :contentReference[oaicite:1]{index=1}

ARE-Stereo is validated on both **KITTI** and the new **DL-Stereo** rail-transit dataset, achieving strong performance in **accuracy**, **robustness**, and **runtime**. :contentReference[oaicite:2]{index=2}

---

## 🔥 Highlights

- ⚡ **Real-time performance**  
  - **77 ms** per frame on KITTI  
  - **92 ms** per frame on DL-Stereo  
  (single RTX 2080Ti) :contentReference[oaicite:3]{index=3}

- 🎯 **High accuracy on KITTI**  
  Superior or comparable to YOLOStereo3D, RTS3D, and SAS3D. :contentReference[oaicite:4]{index=4}

- 🚆 **Tailored for rail transit**  
  Handles long braking distance, complex traffic, occlusion, and low compute resources. :contentReference[oaicite:5]{index=5}

- 🧠 **Stereo-aware enhancements**  
  - PDE recovers suppressed ROIs for distant/occluded objects.  
  - ARA adds depth-aware ROI augmentation to avoid feature loss. :contentReference[oaicite:6]{index=6}

- 💻 **RGB stereo only**  
  Training/inference require **no LiDAR supervision**.

---

## 🧩 Method Overview

ARE-Stereo adopts a **two-stage monocular–stereo cascade** (Fig. 2):

1. **Monocular Stage (ROI Generator)**  
   - Extracts left/right features via a shared backbone and neck.  
   - Predicts 2D center, offsets, 3D box size, depth, and observation angle.  
   - PDE fuses right-view heatmaps into left-view predictions.  
   :contentReference[oaicite:7]{index=7}

2. **Stereo Stage (FCE-based 3D Refinement)**  
   - Builds a **Feature Consistency Embedding (FCE)** volume from stereo features.  
   - ARA generates additional ROIs along the viewing ray.  
   - Extracted FCE blocks are refined into final 3D bounding boxes.  
   :contentReference[oaicite:8]{index=8}

---

## 📊 Benchmark Results

### KITTI – Car (IoU = 0.7)

From Table I (paper). :contentReference[oaicite:9]{index=9}

#### Stereo methods with extra depth supervision:

| Model                     | Time (ms) | AP3D-E | AP3D-M | AP3D-H |
|--------------------------|----------:|-------:|-------:|-------:|
| OC-Stereo                | 350       | 55.15  | 37.60  | 30.25  |
| YOLOStereo3D             | 160       | 65.68  | 41.25  | 30.42  |
| **ARE-Stereo (aux depth)** | **77**    | **66.05** | **41.71** | 30.64  |

#### Stereo methods *without* extra depth supervision:

| Model        | Time (ms) | AP3D-E | AP3D-M | AP3D-H |
|-------------|----------:|-------:|-------:|-------:|
| SIDE        | 208       | 61.22  | 44.46  | 37.15  |
| RTS3D       | 79        | 64.76  | 46.70  | 39.27  |
| SAS3D       | 68        | 65.26  | 47.07  | 39.62  |
| **ARE-Stereo** | **77** | **65.55** | **47.80** | **39.78** |

---

### DL-Stereo – Urban Rail Dataset

From Table II. :contentReference[oaicite:10]{index=10}

| Model                     | Time (ms) | Car | Ped | Truck | Bus |
|--------------------------|----------:|----:|----:|------:|----:|
| OC-Stereo                | 362       | 50.36 | 41.72 | 50.47 | 67.15 |
| YOLOStereo3D             | 169       | 57.49 | 45.62 | 55.33 | 71.88 |
| **ARE-Stereo (aux depth)** | **92** | **57.62** | **46.38** | **55.58** | **72.49** |
| SIDE                     | 214       | 48.89 | 40.66 | 47.29 | 63.39 |
| FGAS-RCNN                | 114       | 50.72 | 41.23 | 49.88 | 65.63 |
| RTS3D                    | 91        | 50.98 | 43.15 | 49.51 | 66.34 |
| SAS3D                    | 78        | 51.62 | 42.97 | 50.07 | 67.63 |
| **ARE-Stereo**             | **92** | **52.14** | **43.41** | **50.26** | **68.12** |

ARE-Stereo achieves **SOTA** without extra depth supervision.

---

## 🚆 DL-Stereo Dataset

DL-Stereo is a rail-transit 3D detection dataset captured on the **Dalian Tram Line 202**.  
Key features (Fig. 8 & 9): :contentReference[oaicite:11]{index=11}

- **4,492** stereo image pairs  
- Resolution: **1080 × 720 → 1080 × 640 crop**  
- Scenarios:
  - High-density vehicles  
  - Pedestrian–vehicle mixed scenes  
  - Pedestrians crossing tracks  
  - Night scenes (~20%)  
- Categories: **Car, Pedestrian, Truck, Bus**

Will be publicly released after clearance.

---

## 📦 Installation

Tested on:

- Ubuntu 20.04  
- Python ≥ 3.8  
- PyTorch ≥ 1.9  
- CUDA 11.x  
- GPU: 2080Ti / 3090

```bash
git clone https://github.com/your_name/ARE-stereo.git
cd ARE-stereo

pip install -r requirements.txt
