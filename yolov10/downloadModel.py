import gradio as gr
import cv2
import tempfile
from ultralytics import YOLOv10
from roboflow import Roboflow
#from IPython.display import Image


model = YOLOv10.from_pretrained('jameslahm/yolov10n)
model.export(...)
