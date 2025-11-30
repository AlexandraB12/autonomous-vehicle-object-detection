# 🏎️ Autonomous Vehicle Object Detection — Detecting Road Objects with YOLOv8

**Python Jupyter Notebook**  

---

## 🧠 Overview

Welcome to the world of **autonomous vehicles**! 🚗  
Modern self-driving cars rely heavily on **computer vision** to perceive their environment and make safe driving decisions. Object detection is at the heart of this technology, allowing vehicles to recognize:

- Other vehicles 🛻
- Pedestrians 🚶‍♂️
- Traffic lights 🚦
- Road signs 🛑
- Road obstacles 🪨

In this project, we explore **YOLOv8** (You Only Look Once), a **state-of-the-art real-time object detection framework**. YOLOv8 processes images quickly while maintaining high accuracy, making it ideal for autonomous driving applications.

This notebook demonstrates:

✅ Loading a pre-trained YOLOv8 model  
📸 Running inference on sample road images  
🖼️ Visualizing predictions with bounding boxes and labels  
⏱️ Measuring inference speed per image

---

## 📂 Dataset

The dataset contains images from urban streets, highways, and parking areas. Each image may contain multiple objects such as cars, trucks, pedestrians, and traffic lights.  

**Demo setup:** 4 sample images; workflow can scale to larger datasets or real-time video.

**Object Classes:**

| Object        | Description                       |
|---------------|-----------------------------------|
| Car 🚗        | All types of cars on the road     |
| Truck 🛻      | Trucks and delivery vehicles      |
| Pedestrian 🚶 | People walking along the road     |
| Traffic Light 🚦 | Red, yellow, green signals      |
| Others 🪨     | Obstacles or miscellaneous objects |

> Images should be stored in a folder named `images/`.

---

## 🎯 Project Objectives

1. Load a pre-trained **YOLOv8** model for object detection.  
2. Perform inference on images to detect road objects.  
3. Visualize results in a **2x2 grid** with bounding boxes and labels.  
4. Measure **processing time** for predictions.  
5. Provide a framework extendable to **video streams** for real-time detection.

---

## 💡 Tools & Libraries

**Python Installations**

```bash
pip install ultralytics tqdm opencv-python
```

**Python Imports**

```python
import os
import glob
import random
import time
import math
import warnings

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
from ultralytics import YOLO

warnings.filterwarnings("ignore")
```

**Tool explanations:**
- **NumPy:** numerical operations for arrays and data processing.
- **Pillow (PIL):** loading and manipulating images.
- **OpenCV (cv2):** advanced computer vision and image processing.
- **Matplotlib:** plotting and visualizing images.
- **Ultralytics YOLOv8:** pre-trained object detection model.
- **TQDM:** progress bars for loops.

---

## ⚙️ How to Run

1. Clone the repository or download the notebook and dataset folder.
2. Ensure required Python packages are installed.
3. Place images inside a folder named `images/`.
4. Run the notebook step by step to load images, predict objects, and visualize results.

---

## 📸 Exploratory Sample Visualization

Randomly select 4 images to see what the vehicle “sees”:

```python
# Load image paths
image_paths = glob.glob('./images/*')
selected_images = random.sample(image_paths, 4)
images = [Image.open(p) for p in selected_images]

# Display images in 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for ax, img in zip(axes.flatten(), images):
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

This helps inspect raw images before prediction.

---

## 🤖 YOLOv8 Model Setup

```python
# Load pre-trained YOLOv8 model (medium size)
yolo_model = YOLO("yolov8m.pt")
```

**YOLOv8 model variants:**
| Model | Description |
|-------|-------------|
| yolov8n.pt | Nano: lightweight, very fast |
| yolov8m.pt | Medium: balanced accuracy/speed |
| yolov8l.pt | Large: higher accuracy, slower |
| yolov8x.pt | Extra-large: best accuracy, slower |

---

## 🏃 Object Detection & Inference

```python
start_time = time.time()

for i, img in enumerate(images):
    results = yolo_model.predict(img)
    output = results[0]
    boxes = output.boxes

    print(f"Image {i+1} detections:")
    for j in range(len(boxes)):
        label = output.names[boxes.cls[j].item()]
        coords = boxes.xyxy[j].tolist()
        conf = np.round(boxes.conf[j].item(), 2)
        print(f" - {label} at {coords}, confidence: {conf}")
    print("-------")

    # Annotate image for visualization
    images[i] = output.plot()[..., ::-1]  # BGR to RGB

end_time = time.time()
print(f"Total inference time: {end_time - start_time:.2f} seconds")
```

We extract class names, bounding boxes, and confidence scores for each detected object.

---

## 🖼️ Display Predictions

```python
# Automatic grid adjustment
cols = 2
rows = math.ceil(len(images)/cols)
plt.figure(figsize=(12, rows*5))

for i, img in enumerate(images):
    plt.subplot(rows, cols, i+1)
    plt.imshow(img)
    plt.axis('off')

plt.tight_layout()
plt.show()
```

Annotated images show detected objects clearly, giving insights into vehicle perception.

---

## 📌 Key Takeaways

- **YOLOv8** detects cars, trucks, pedestrians, and traffic lights effectively.
- **Real-time performance:** ~1–2 seconds per image.
- **Visual insights:** Annotated images help engineers understand vehicle perception.
- **Extendable:** Framework can process video streams and larger datasets.

---

## 🔮 Next Steps

- Deploy the model for **real-time video detection**.
- Compare YOLOv8 with YOLOv5 or Faster R-CNN.
- Apply **data augmentation** for rare road scenarios.
- Integrate detections into **autonomous driving decision systems**.
- Evaluate using **precision, recall, mAP** metrics.

---

## 🧾 Author

Alexandra Boudia  
Data Scientist | Computer Vision Enthusiast | AI for Autonomous Vehicles

