# Leaf Disease Detection using YOLOv5 & YOLOv8

A deep learning-based project for detecting diseases in plant leaves using YOLOv5 and YOLOv8 object detection models.  

---

## 📂 Dataset
- Custom dataset of 30 leaf classes including Apple, Bell Pepper, Tomato, Potato, Strawberry, Grape, and others.
- Each class contains labeled images in YOLO format (`.txt` files with bounding boxes and class IDs).
- Labels and images are stored separately
  
dataset/

├── train/images

├── train/labels

├── test/images

├── test/labels

└── data.yaml # YOLO configuration file


### Classes
| Class ID | Label |
|----------|-------|
| 1  | Apple Scab Leaf |
| 2  | Apple Leaf |
| 3  | Apple Rust Leaf |
| 4  | Bell Pepper Leaf Spot |
| 5  | Bell Pepper Leaf |
| 6  | Blueberry Leaf |
| 7  | Cherry Leaf |
| 8  | Corn Gray Leaf Spot |
| 9  | Corn Leaf Blight |
| 10 | Corn Rust Leaf |
| 11 | Peach Leaf |
| 12 | Potato Leaf Early Blight |
| 13 | Potato Leaf Late Blight |
| 14 | Potato Leaf |
| 15 | Raspberry Leaf |
| 16 | Soyabean Leaf |
| 17 | Soybean Leaf |
| 18 | Squash Powdery Mildew Leaf |
| 19 | Strawberry Leaf |
| 20 | Tomato Early Blight Leaf |
| 21 | Tomato Septoria Leaf Spot |
| 22 | Tomato Leaf Bacterial Spot |
| 23 | Tomato Leaf Late Blight |
| 24 | Tomato Leaf Mosaic Virus |
| 25 | Tomato Leaf Yellow Virus |
| 26 | Tomato Leaf |
| 27 | Tomato Mold Leaf |
| 28 | Tomato Two Spotted Spider Mites Leaf |
| 29 | Grape Leaf Black Rot |
| 30 | Grape Leaf |

---

## 🛠️ Model Training & Detection

### YOLOv5
1. Load dataset:
    ```python
    import yaml
    with open('data.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    class_names = data['names']
    print(class_names)

2. Train YOLOv5 model (example):
   ```bash
   !python train.py --img 128 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt
3. Detect leaves:
   ```bash
   !python detect.py --weights runs/train/yolov5s_results/weights/best.pt --img 128 --conf 0.25 --source test/images
   !python detect.py --weights runs/train/yolov5s_results/weights/best.pt --img 128 --conf 0.15 --source test/images
   !python detect.py --weights runs/train/yolov5s_results/weights/best.pt --source test/images --save-conf

### YOLOv8
1. Predict using trained YOLOv8 model
   ```bash
   !yolo task=detect mode=predict model="runs/detect/train3/weights/best.pt" data=data.yaml source=test/images

2. Outputs include:

   Predicted images with bounding boxes
   Confusion matrix and validation batch images
   Detection results saved to runs/detect/predict/

📊 Analysis

Visualize class distribution and number of images per class

Example:

   ```python
   import pandas as pd
   images_per_class = pd.DataFrame(map.values(), index=map.keys(), columns=["Number of Images"])
   images_per_class['Labels'] = d.values()
   images_per_class
     
📁 Project Structure

leaf-disease-detection/

├── train/                 # Training images & labels

├── test/                  # Testing images & labels

├── data.yaml              # YOLO dataset config

├── yolov5/                # YOLOv5 repo

├── runs/                  # Trained weights & prediction results

└── yolo.ipynb             # Colab notebook with dataset exploration & predictions

📬 Contact

Developer: Malla Prasanth

Email: mallaprasanth1234@gmail.com

GitHub: https://github.com/Prasanth-malla
