# ARE-Stereo: Stereo Real-time 3D Object Detection with Adaptive ROI Enhancement for Rail Transit

This repository contains the official implementation of:

> **Stereo Real-time 3D Object Detection with Adaptive ROI Enhancement for Rail Transit** :contentReference[oaicite:0]{index=0}  

ARE-Stereo is a real-time stereo 3D object detector designed for **urban rail transit** environments.  
It adopts a **monocular–stereo cascaded architecture** and introduces two key strategies:

- **Pseudo Disparity Enhancement (PDE)** – recovers suppressed regions of interest (ROIs) using pseudo disparity.
- **Adaptive ROI Augmentation (ARA)** – adaptively expands ROIs based on depth uncertainty to preserve critical stereo features while keeping computation affordable. :contentReference[oaicite:1]{index=1}  

ARE-Stereo is validated on both the **KITTI** benchmark and our self-built **DL-Stereo** urban rail dataset, demonstrating strong performance in terms of **accuracy, range, and inference speed**. :contentReference[oaicite:2]{index=2}  

---

## 🔥 Highlights

- ⚡ **Real-time**  
  - Achieves **77 ms** per frame on KITTI and **92 ms** per frame on DL-Stereo with a single NVIDIA 2080Ti. :contentReference[oaicite:3]{index=3}  

- 🎯 **High accuracy on KITTI (Car)**  
  - **ARE-Stereo (aux depth head)** reaches **SOTA-level performance** among stereo methods with extra depth supervision, surpassing YOLOStereo3D and TAG under AP3D @ IoU 0.7. :contentReference[oaicite:4]{index=4}  

- 🚆 **Rail-transit oriented design**  
  - Specifically optimized for long braking distances, limited onboard compute, and complex mixed-traffic rail scenarios (pedestrians, buses, trams, cars). :contentReference[oaicite:5]{index=5}  

- 🧠 **Stereo-aware ROI refinement**  
  - PDE boosts low-confidence ROIs from distant/occluded objects by warping right-view heatmaps into the left-view coordinate system using pseudo disparity.
  - ARA generates additional ROIs along the viewing ray using depth and its uncertainty, greatly improving robustness to monocular depth errors. :contentReference[oaicite:6]{index=6}  

- 💻 **Pure RGB stereo input**  
  - Training and inference rely only on stereo RGB images.
  - LiDAR is used *only* for dataset annotation in DL-Stereo.

---

## 🧩 Method Overview

ARE-Stereo follows a two-stage monocular–stereo cascade (see Fig. 2 in the paper): :contentReference[oaicite:7]{index=7}  

1. **Monocular Stage (ROI Generator)**
   - Shared backbone and neck extract features from the left and right images.
   - Left features predict:
     - 2D center heatmap
     - offset from 2D center to 3D projected center
     - 3D size
     - depth
     - observation angle :contentReference[oaicite:8]{index=8}  
   - Right features predict:
     - 2D center
     - depth
     - offset
   - PDE fuses right-view information into the left-view ROI heatmap.

2. **Stereo Stage (Stereo Detector)**
   - Builds a **Feature Consistency Embedding (FCE)** space by concatenating left/right features. :contentReference[oaicite:9]{index=9}  
   - A dedicated sampler extracts local FCE blocks for each ROI.
   - ARA generates additional ROIs along the depth ray using predicted depth and its Laplace uncertainty.
   - A stereo detection head refines 3D boxes; low-confidence results can be further iteratively refined. :contentReference[oaicite:10]{index=10}  

---

## 📊 Benchmark Results

### KITTI – Car Category (IoU = 0.7)

From Table I in the paper (Car, test and val splits). :contentReference[oaicite:11]{index=11}  

**Stereo methods with extra depth supervision**

| Model                     | Time (ms) | AP3D (Test) Easy | AP3D (Test) Mod. | AP3D (Test) Hard |
|--------------------------|----------:|-----------------:|-----------------:|-----------------:|
| OC-Stereo                | 350       | 55.15            | 37.60            | 30.25            |
| YOLOStereo3D             | 160       | 65.68            | 41.25            | 30.42            |
| TAG (wo/unannot)         | 160       | –                | –                | –                |
| **ARE-Stereo (aux depth)** | **77**    | **66.05**        | **41**
