import gradio as gr
import cv2
import tempfile
from yolov10.ultralytics import YOLOv10


model = YOLOv10.from_pretrained('jameslahm/yolov10l')

#!yolo task=detect mode=train epochs=10 batch=32 plots=True \
#model=/lab/micah/yolov10/yolov10n.pt
#data=/lab/micah/yolov10/hard hat uni/data.yaml


model.train(data='hard hat uni/data.yaml', epochs=50)
print("done training 0 layers frozen")

model.train(data='hard hat uni/data.yaml', epochs=50, freeze=20)
print("done training 20 layers frozen")
#validation = model.val(data='/lab/micah/yolov10/hard hat uni/data.yaml')


print("done training")

