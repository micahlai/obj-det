import gradio as gr
import cv2
import tempfile
from yolov10.ultralytics import YOLOv10
import time
import os
import torch

model = YOLOv10.from_pretrained('jameslahm/yolov10l')

torch.save(model, 'yolov10l.pt')