# ARE-stereo: Real-time Stereo 3D Object Detection with Adaptive ROI Enhancement

ARE-stereo 是一个为城市轨道交通场景设计的实时 3D 目标检测模型，采用单目-双目级联结构，并提出 **Pseudo Disparity Enhancement（PDE）** 与 **Adaptive ROI Augmentation（ARA）** 两个关键策略。模型在远距、小目标与遮挡目标上显著提升检测效果，同时保持实时性，已在真实有轨电车场景部署验证。


## 🌟 Highlights

- 实时性能：单 NVIDIA 2080Ti 上 70–90ms 推理速度  
- 更高精度：KITTI 与 自建DL-Stereo数据集上均优于现有实时模型  
- PDE：恢复被抑制 ROI（远距/遮挡小目标）  
- ARA：自适应扩增 3D ROI，避免 stereo 特征缺失  
- 嵌入式可部署：Xavier / Orin 上经 TensorRT 验证，满足轨交实时需求  
- 实际应用：已在大连 202 电车真实线路进行测试

## 🚀 Model Architecture Overview

模型包含 3 个核心模块：

### 1. Feature Extractor（共享 Backbone + 多任务预测头）
- 左右视图分别输出 feature map  
- 预测：2D center、offset、shape、depth、orientation

### 2. ROI Generator + PDE
- PDE 利用从右目视差推断的 pseudo disparity  
- Warp 右目 ROI 到左目特征  
- 恢复被压低置信度的远距/小目标 ROI，提高召回率

### 3. Stereo Detector + ARA
- 构建局部 FCE（Feature Consistency Embedding）体积  
- ARA 根据深度不确定性膨胀 ROI  
- Stereo head 进行最终 3D refinement

---

# 📊 Experimental Results

## 1. KITTI 3D Detection Benchmark（Car, IoU=0.7）

### 1.1 使用额外监督（如深度）的模型对比

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

## 3. Ablation Study（消融实验）

### 3.1 PDE / ARA 消融

| Baseline | PDE | ARA | Time ↓ | AP3D | APBEV |
|----------|-----|------|--------|-------|--------|
| ✓ | – | – | 72ms | 51.32 | 63.84 |
| ✓ | ✓ | – | 76ms | 51.57 | 64.27 |
| ✓ | ✓ | ✓ | 92ms | **52.14** | **65.35** |

### 3.2 不同 Monocular Stage 替换实验

| Mono Model | Stereo Time | Mono AP3D | Stereo AP3D | Mono APBEV | Stereo APBEV |
|------------|-------------|-----------|--------------|--------------|----------------|
| KM3D | 91ms | 21.71 | 51.37 | 32.45 | 64.60 |
| MonoCon | 86ms | 22.26 | 51.65 | 33.38 | 64.53 |
| MonoLSS | 96ms | 22.79 | 51.89 | 33.66 | 65.03 |
| **ARE-stereo** | **92ms** | 22.17 | **52.14** | 32.97 | **65.35** |

---

# 🧪 Inference Demo

```bash
python demo.py --model_path checkpoints/are_stereo.pth \
               --left_img demo/000001_left.png \
               --right_img demo/000001_right.png
```

# 🧵 Training

## Step 1 — Train Monocular Stage
```bash
python train_mono.py --cfg configs/mono.yaml
```

## Step 2 — Train Stereo Stage
```bash
python train_stereo.py --cfg configs/stereo.yaml
```

> ⚠️ **DL-stereo 数据集：**  
> 由于隐私政策，目前仅公布部分数据；完整版将在论文正式录用后发布。如需提前获取，请联系作者。

# 🙏 Acknowledgements
- CenterNet  
- RTM3D  
- RTS3D  
