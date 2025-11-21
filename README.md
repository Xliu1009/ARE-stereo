# ARE-stereo: Real-time Stereo 3D Object Detection with Adaptive ROI Enhancement

ARE-stereo 是一个面向城市轨道交通场景的实时 3D 目标检测模型，采用单目-双目级联结构，并提出 **Pseudo Disparity Enhancement（PDE）** 与 **Adaptive ROI Augmentation（ARA）** 两项核心策略。模型在远距离、小目标以及遮挡场景中显著有效提升检测精度，同时保持实时推理性能，已在真实有轨电车环境中完成部署与验证。

---

## 🌟 Highlights

- 实时：单 NVIDIA 2080Ti 上单帧推理约 70–90 ms  
- 高精度：在 KITTI 与 DL-Stereo 数据集上均优于现有实时双目 3D 检测方法  
- PDE：利用伪视差恢复远距/遮挡目标中被压低置信度的 ROI  
- ARA：基于深度不确定性自适应扩增 ROI，避免关键双目特征缺失  
- 可部署：已在 Jetson Xavier / Orin 等平台通过 TensorRT 加速部署  
- 面向轨道交通：实际在大连 202 路有轨电车部署  

---

## 🚀 模型结构概览（对应论文 Fig. 2）

![ARE-stereo Overall Pipeline (Fig. 2)]("assets/fig2_are_stereo_pipeline.jpg")


---

#  实验结果

## 1. KITTI 3D Detection（Car, IoU=0.7）

### 使用额外深度监督

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

### PDE / ARA

| Base | PDE | ARA | Time | AP3D | APBEV |
|------|-----|-----|------|------|-------|
| ✓ | – | – | 72 | 51.32 | 63.84 |
| ✓ | ✓ | – | 76 | 51.57 | 64.27 |
| ✓ | ✓ | ✓ | 92 | 52.14 | 65.35 |

---

# 📸 DL-Stereo 数据集示例（对应论文 Fig. 8）

### Fig. 8(a)：高密度车辆
![DL-Stereo Vehicles (Fig. 8a)]("assets/fig8a_high_density_vehicles.jpg")

### Fig. 8(b)：高密度行人
![DL-Stereo Pedestrians (Fig. 8b)]("assets/fig8b_high_density_pedestrians.jpg")

### Fig. 8(c)：横穿轨道
![DL-Stereo Crossing (Fig. 8c)]("assets/fig8c_crossing_rails.jpg")

### Fig. 8(d)：夜间场景
![DL-Stereo Night (Fig. 8d)]("assets/fig8d_night_scene.jpg")

---

### ⚠️ 数据集说明  
由于隐私政策，DL-stereo 数据集仅开放部分示例；完整版将在论文录用后开放。如需提前获取请联系作者。

---

# 🎨 检测结果示例（对应 Fig. 18）

### KITTI（Fig. 18a）
![KITTI Qualitative (Fig. 18a)]("assets/fig18a_kitti_qualitative.jpg")

### DL-Stereo（Fig. 18b）
![DL-Stereo Qualitative (Fig. 18b)]("assets/fig18b_dl_stereo_qualitative.jpg")

---

# 部署环境（对应 Fig. 7 + 附录）

### Fig. 7(a)
![Sensor Mount (Fig. 7a)]("assets/fig7a_stereo_lidar_system.jpg")

### Fig. 7(b)
![Platform Schematic (Fig. 7b)]("assets/fig7b_embedded_platform_schematic.jpg")

---

#  Inference

```bash
python demo.py --model_path checkpoints/are_stereo.pth --left_img demo/left.png --right_img demo/right.png
```

---

#  Training

```bash
python train_mono.py --cfg configs/mono.yaml
python train_stereo.py --cfg configs/stereo.yaml
```

---

#  Acknowledgements
CenterNet · RTM3D · RTS3D
