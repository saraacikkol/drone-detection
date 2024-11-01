# Drone Object Detection using YOLOv5
This README file outlines the steps and details for training a YOLOv5 model on Colab using a drone detection dataset sourced from Roboflow.

# Drone Detection Dataset

To support the development and evaluation of drone detection models, we present a comprehensive dataset specifically curated for this purpose. The dataset is sourced from the publicly available Drone Object Detection Dataset on Roboflow and contains annotated images captured in diverse environmental conditions and camera perspectives.

This dataset is designed to enable robust detection and classification of drones and other common flying objects.
https://universe.roboflow.com/project-ddrone/datasetdrone-trv98

Dataset Features:

a. Classes: Drones, Helicopters, Birds, and Airplanes

b. Annotations: Includes bounding boxes and labels in YOLO format

c. Diverse Conditions: Images taken under various environmental conditions and from different camera angles



# Install required packages
```
!pip install ultralytics
!pip install roboflow
```

# Access to GPU
```
!nvidia-smi
```

# Print current working directory
```
import os
HOME = os.getcwd()
print(HOME)
```
# Install YoLov5
```
%cd yolov5
%pip install -r requirements.txt comet_ml
!pip install -e .
```
# Add the current directory to the Python path.
```
import sys
sys.path.append('.')
```
# Import torch and the yolov5 utility functions
```
import torch
from yolov5 import utils
display = utils.notebook_init()  # checks
```
# Download the dataset
```
%cd {HOME}
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="your_api_key")
project = rf.workspace("workspace_name").project("project_name")
version = project.version(version_number)
dataset = version.download("yolov5")
```
# Train the model
```
!python train.py --img 384 --epochs 10 --data {dataset.location}/data.yaml --weights yolov5m.pt
```

# Evaluate the model

```
%cd {HOME}
!yolo task=detect mode=val model={model.location} data={dataset.location}/data.yaml
```

# Visualize the confusion matrix
```
%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/results.png', width=600)
```

# Display the results of your model's performance
```
%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/results.png', width=600)
```
