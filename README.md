# Real-Time Vehicle & Pedestrian Detection for Automotive Vision Applications

> **YOLOv8s fine-tuned on self-driving traffic data — achieving mAP50 of 0.66 across 10 object classes in ~59 minutes of training on NVIDIA L4 GPU.**

This project is developed as part of the TEEP (Taiwan Experience Education Program) selection process at Tatung University, under the supervision of Prof. Chu Shou Yang. It demonstrates the application of transfer learning and real-time object detection for automotive vision systems — directly relevant to camera module and Around View Monitor (AVM) development at REC Technology Corporation.

---

## Demo Results

| Sample Annotations (Ground Truth) | Inference Results (Model Prediction) |
|---|---|
| <img width="2148" height="1595" alt="sample_annotations" src="https://github.com/user-attachments/assets/768a6e26-56af-4abe-8284-8b53e4d09470" /> | <img width="2148" height="1595" alt="inference_results" src="https://github.com/user-attachments/assets/020b829b-fc7a-4926-973a-33878603d6b1" /> |

---

## Project Overview

Modern automotive safety systems require precise, real-time detection of vehicles, pedestrians, and traffic signals. This project builds upon a previous CNN-based vehicle classifier (81% accuracy, 3-class) and elevates it to a full object detection pipeline using YOLOv8s with transfer learning — capable of localizing objects with bounding boxes and confidence scores across 10 classes simultaneously.

**Key upgrade from previous work:**

| Previous Project | This Project |
|---|---|
| Image Classification (CNN) | Object Detection (YOLOv8s) |
| 3 classes (bus, car, motor) | 10 classes including traffic lights |
| Accuracy metric | mAP50, Precision, Recall |
| No localization | Bounding box + confidence score |
| Static image only | Real-time capable (1.3ms/image) |

---

## Model Performance

### Test Set Results (990 images — never seen during training)

| Metric | Value |
|---|---|
| **mAP50** | **0.6604** |
| mAP50-95 | 0.3465 |
| Precision | 0.7709 |
| Recall | 0.6071 |
| Inference Speed | 2.5ms per image |

### Per-Class mAP50 Breakdown

| Class | Images | Instances | mAP50 |
|---|---|---|---|
| car | 976 | 4691 | 0.772 |
| trafficLight-Red | 204 | 497 | 0.789 |
| trafficLight | 136 | 216 | 0.760 |
| trafficLight-RedLeft | 103 | 140 | 0.725 |
| truck | 180 | 265 | 0.715 |
| trafficLight-GreenLeft | 20 | 30 | 0.720 |
| biker | 81 | 120 | 0.555 |
| trafficLight-Green | 130 | 427 | 0.585 |
| pedestrian | 238 | 764 | 0.517 |
| trafficLight-Yellow | 9 | 24 | 0.466 |

> **Note:** Lower mAP for `pedestrian` and `trafficLight-Yellow` is attributable to class imbalance — significantly fewer training instances compared to `car`. This is a known limitation and a direction for future improvement.

### Training Progress

<img width="2147" height="1184" alt="training_curves" src="https://github.com/user-attachments/assets/017f40d7-8739-4461-894f-c808b837197f" />

The curves demonstrate healthy training behavior:
- All three loss functions (Box, Classification, DFL) decrease consistently
- No signs of severe overfitting — validation loss stabilizes rather than diverging
- mAP50 grows from 0.30 at epoch 1 to 0.647 at epoch 80

---

## Technical Architecture

### Model: YOLOv8s (Small)

```
Architecture : CSPDarknet backbone + PANet neck + Detection head
Parameters   : 11,129,454
GFLOPs       : 28.5
Model size   : 22.5 MB (best.pt)
Input size   : 640 × 640 px
```

**Why YOLOv8s over YOLOv8n (nano)?**
The small variant offers significantly better accuracy (11.2M vs 3.2M parameters) while remaining well within the GPU memory budget of the NVIDIA L4 (23GB VRAM). For automotive safety applications where missed detections have real consequences, the accuracy trade-off justifies the additional compute.

**Why Transfer Learning?**
YOLOv8s was pre-trained on COCO (80 classes, 118k images). Rather than training from scratch — which would require far more data and time — we fine-tune the existing weights on our traffic-specific dataset. This approach leverages learned low-level features (edges, textures) while adapting high-level recognition to our target classes.

### Dataset

| Property | Value |
|---|---|
| Source | Roboflow Universe — Self-Driving Traffic Detection |
| Original size | 13,860 train / 1,980 val / 990 test |
| Training subset used | 3,465 (25% — proof-of-concept experiment) |
| Classes | 10 (biker, car, pedestrian, trafficLight variants, truck) |
| Format | YOLOv8 (normalized bounding boxes) |
| Domain | Dashcam footage, real-world road conditions |

**Why 25% subset?**
A full-dataset training run (~13k images × 80 epochs) would require 8–12 hours, exceeding the scope of this proof-of-concept. The 25% subset (3,465 images) was randomly sampled with a fixed seed (42) to ensure reproducibility. Validation and test sets were kept at full size for objective evaluation.

### Training Configuration

```python
model     = YOLOv8s (pretrained on COCO)
epochs    = 80
patience  = 20  # early stopping
imgsz     = 640
batch     = 16
optimizer = AdamW
lr0       = 0.001
GPU       = NVIDIA L4 (23 GB VRAM)
duration  = 0.989 hours (~59 minutes)
```

**Why AdamW with lr0=0.001?**
Transfer learning benefits from a lower learning rate — the pretrained weights are already good, so large updates would destroy learned features. AdamW adds weight decay regularization, helping prevent overfitting especially with a relatively small training subset.

---

## Repository Structure

```
vehicle-pedestrian-detection-yolov8/
│
├── vehicle_pedestrian_detection_yolov8.ipynb  # Main Colab notebook
│
├── assets/
│   ├── sample_annotations.png     # Dataset visualization
│   ├── training_curves.png        # Loss & mAP training progress
│   └── inference_results.png      # Model predictions on test images
│
└── README.md
```

---

## How to Reproduce

### Requirements

```bash
pip install ultralytics roboflow
```

### Steps

**1. Clone this repository**
```bash
git clone https://github.com/Irfansyah001/vehicle-pedestrian-detection-yolov8.git
cd vehicle-pedestrian-detection-yolov8
```

**2. Open the notebook**

Open `vehicle_pedestrian_detection_yolov8.ipynb` in Google Colab. Set runtime to GPU (L4 recommended).

**3. Download dataset**
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("selfdriving-traffic-detection").project("self-driving-traffic-detection")
dataset = project.version(2).download("yolov8")
```

**4. Run all cells sequentially**

The notebook will handle subset creation, training, evaluation, and visualization automatically.

### Quick Inference (after training)

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
results = model.predict("your_image.jpg", conf=0.25)
results[0].show()
```

---

## Key Findings & Discussion

**Strengths of this approach:**
- Transfer learning dramatically reduces training time (59 min vs estimated 8+ hours from scratch)
- Strong performance on dominant classes: `car` (mAP50 0.772), `truck` (0.715), `trafficLight-Red` (0.789)
- Inference at 2.5ms/image enables real-time deployment at >30 FPS

**Limitations and future work:**
- Class imbalance affects minority classes: `pedestrian` (517 instances) vs `car` (4,691 instances). Potential solution: weighted loss, oversampling, or targeted data augmentation
- Training on 25% of available data — scaling to full dataset expected to improve mAP50 to 0.70+
- Future direction: export to ONNX or TensorRT for edge deployment on automotive hardware

**Relevance to automotive vision systems:**
This pipeline mirrors the core detection stack required in Around View Monitor (AVM) systems — multi-class real-time object detection from camera feeds. The inclusion of traffic light state detection (Green/Red/Yellow/GreenLeft/RedLeft) extends the utility beyond simple obstacle detection toward decision-support for autonomous driving.

---

## License

This project is released under the MIT License. The dataset is provided by Roboflow Universe under its respective terms of use.
