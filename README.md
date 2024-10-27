# Drone Object Detection using YOLOv8
This README file outlines the steps and details for training a YOLOv8 model on Colab using a drone detection dataset sourced from Roboflow.

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
# Install YoLov8 using pip (recommended)
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
dataset = version.download("yolov8")
```
# Train the model
```
!yolo task=detect mode=train model=yolov8n.pt data=/content/project_name/data.yaml epochs=5
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
