# Drone Detection Research
This README file outlines the steps and details for training a YOLOv8 model on Colab using a drone detection dataset sourced from Roboflow.

# Dataset
Drone Detection Dataset:
https://universe.roboflow.com/project-ddrone/datasetdrone-trv98

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
```import os
HOME = os.getcwd()
print(HOME)
```
# Install yolov8 using pip (recommended)
```
!pip install ultralytics==8.2.103 -q

from IPython import display
display.clear_output()
import ultralytics
ultralytics.checks()

from ultralytics import YOLO
from IPython.display import display, Image
```
# Download the dataset
```
%cd {HOME}
!pip install roboflow
from roboflow import Roboflow

rf = Roboflow(api_key="your_api_key")
project = rf.workspace("workspace_name").project("project_name")
version = project.version(version_number)
dataset = version.download("yolov8-obb")
```
# Train the model
```
%cd {HOME}
!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=25 imgsz=640
```

# Evaluate the model

