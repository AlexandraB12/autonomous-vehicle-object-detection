# 🏎️ Autonomous Vehicle Object Detection — Detecting Road Objects with YOLOv8

**Python Jupyter Notebook**

---

## 🧠 Overview

Welcome to the world of **autonomous vehicles**! 🚗  
Modern self-driving cars rely heavily on computer vision to understand their environment and make safe decisions. Object detection is at the core of this technology, enabling vehicles to recognize:

- Other vehicles 🛻
- Pedestrians 🚶‍♂️
- Traffic lights 🚦
- Road signs 🛑
- Obstacles and miscellaneous objects 🪨

In this project, we explore **YOLOv8** (You Only Look Once), a cutting-edge real-time object detection framework. YOLOv8 can process images quickly and accurately, making it ideal for autonomous driving applications.  

This notebook demonstrates:

✅ Loading a pre-trained YOLOv8 model  
📸 Running inference on sample road images  
🖼️ Visualizing predictions with bounding boxes and labels  
⏱️ Measuring inference speed for each image  

---

## 📂 Dataset

The dataset consists of images from urban and highway scenarios with multiple objects. While this demo uses **4 sample images**, the workflow can be applied to large datasets or real-time streams.  

**Included objects**:

| Object        | Description                       |
|---------------|-----------------------------------|
| Car 🚗        | All types of cars on the road     |
| Truck 🛻      | Trucks and delivery vehicles      |
| Pedestrian 🚶 | People walking along the road     |
| Traffic Light 🚦 | Red, yellow, green signals      |
| Others 🪨     | Road obstacles or miscellaneous  |

> **Note:** Images are stored in the `images/` folder.  

---

## 🎯 Project Objectives

1. Load a pre-trained **YOLOv8** model for object detection.  
2. Perform inference on new images to detect road objects.  
3. Visualize results in a clean **2x2 grid** with bounding boxes and labels.  
4. Measure **processing time** and efficiency of the model.  
5. Provide a framework easily extendable to **video streams** for real-time detection.  

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

**Explanation of Tools**

- **NumPy**: Numerical operations for data processing  
- **Pillow (PIL)**: Image loading and manipulation  
- **OpenCV (cv2)**: Advanced computer vision functions  
- **Matplotlib**: Plotting and visualizing images  
- **Ultralytics YOLOv8**: Pre-trained model for object detection  
- **TQDM**: Progress bars for loops  

---

## ⚙️ How to Run

1. Clone the repository or download the notebook and dataset folder.  
2. Ensure required packages are installed.  
3. Place images inside a folder named `images/`.  
4. Run the notebook step by step to load images, predict objects, and visualize results.  

---

## 📸 Exploratory Sample Visualization

Randomly select 4 images to see what the car “sees”:

```python
# Load images
image_paths = glob.glob('./images/*')
selected_images = random.sample(image_paths, 4)
images = [Image.open(p) for p in selected_images]

# Display images in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for ax, img in zip(axes.flatten(), images):
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

> This step allows us to inspect the raw images before running predictions.

---

## 🤖 YOLOv8 Model Setup

```python
# Load pre-trained YOLOv8 model (medium size)
yolo_model = YOLO("yolov8m.pt")
```

YOLOv8 models come in various sizes:

| Model  | Description                      |
|--------|----------------------------------|
| yolov8n.pt | Nano: lightweight, fast        |
| yolov8m.pt | Medium: balanced accuracy/speed|
| yolov8l.pt | Large: higher accuracy, slower|
| yolov8x.pt | Extra-large: best accuracy    |

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
    images[i] = output.plot()[..., ::-1]  # Convert BGR to RGB
end_time = time.time()

print(f"Total inference time: {end_time - start_time:.2f} seconds")
```

> We extract class names, bounding boxes, and confidence scores for each detected object.  

---

## 🖼️ Display Predictions

```python
# Automatic grid adjustment based on number of images
cols = 2
rows = math.ceil(len(images) / cols)
plt.figure(figsize=(12, rows * 5))

for i, img in enumerate(images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.axis("off")

plt.tight_layout()
plt.show()
```

> The annotated images show objects detected by YOLOv8, providing clear insights into vehicle perception.

---

## 📌 Key Takeaways

- **YOLOv8** efficiently detects cars, trucks, pedestrians, and traffic lights.  
- **Real-time performance:** ~1–2 seconds per image.  
- **Visual insights:** Annotated images help engineers understand vehicle perception.  
- **Extendable:** This setup can be applied to video streams or larger datasets.  

---

## 🔮 Next Steps

- Deploy the model for **real-time video detection**.  
- Compare YOLOv8 with **YOLOv5** or **Faster R-CNN** models.  
- Apply **data augmentation** for rare road scenarios.  
- Integrate detections into **autonomous driving decision systems**.  
- Evaluate using additional metrics (precision, recall, mAP).  

---

## 🧾 Author

**Alexandra Boudia**  
Data Scientist | Computer Vision Enthusiast | AI for Autonomous Vehicles

---

