import gradio as gr
import cv2
import tempfile
from yolov10.ultralytics import YOLOv10

model = YOLOv10.from_pretrained('jameslahm/yolov10l')

#!yolo task=detect mode=train epochs=10 batch=32 plots=True \
#model=/lab/micah/yolov10/yolov10n.pt
#data=/lab/micah/yolov10/hard hat uni/data.yaml


results = model.train(data='/lab/micah/obj-det/hard hat uni/data.yaml', epochs=30,freeze=20)

print("done training")

file=open('obj-det/results.txt', 'w')
file.write(str(results))
file.close()

print(results.speed)


