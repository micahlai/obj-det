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

#model = YOLOv10('/lab/micah/yolov10/runs/detect/train3/weights/best.pt')
model = YOLOv10('/lab/micah/yolov10/yolov10n.pt')

#!yolo task=detect mode=train epochs=10 batch=32 plots=True \
#model=/lab/micah/yolov10/yolov10n.pt
#data=/lab/micah/yolov10/hard hat uni/data.yaml

imagePath='/lab/micah/Downloads/hardhattest.jpg'
results = model(source=imagePath,conf=0.25)

print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
print(results[0].boxes.cls)


image = cv2.imread(imagePath)
detections = sv.Detections.from_ultralytics(results[0])

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)

annotated_image.save('/lab/micah/annotatedImage.jpg')
print('saved annotated image')

