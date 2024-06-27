import gradio as gr
import cv2
import tempfile
from ultralytics import YOLOv10
from roboflow import Roboflow
#from IPython.display import Image
import supervision as sv


#rf = Roboflow(api_key="gPxEF3ZGSUeLAVy1OPUy")
#project = rf.workspace("universe-datasets").project("hard-hat-universe-0dy7t")
#version = project.version(26)
#dataset = version.download("yolov8")

model = YOLOv10('/lab/micah/yolov10/runs/detect/train3/weights/best.pt')

#!yolo task=detect mode=train epochs=10 batch=32 plots=True \
#model=/lab/micah/yolov10/yolov10n.pt
#data=/lab/micah/yolov10/hard hat uni/data.yaml

#imagePath='/lab/micah/Downloads/hardhattest.jpg'
#results = model(source=imagePath,conf=0.25)

model.val(data='/lab/micah/yolov10/hard hat uni/data.yaml')


