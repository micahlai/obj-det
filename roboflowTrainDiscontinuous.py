import gradio as gr
import cv2
import tempfile
from yolov10.ultralytics import YOLOv10
import os

save_dir = '/lab/micah/obj-det/testing runs/7-3 discontin'

os.makedirs(save_dir + '/results', exist_ok=True)
os.makedirs(save_dir + '/models', exist_ok=True)

model = YOLOv10.from_pretrained('jameslahm/yolov10l')

#!yolo task=detect mode=train epochs=10 batch=32 plots=True \
#model=/lab/micah/yolov10/yolov10n.pt
#data=/lab/micah/yolov10/hard hat uni/data.yaml

frozenLayers = [2,3,4,5,6,7,8,9,10]

print("training discontin")
results1 = model.train(data='/lab/micah/obj-det/datasets/Resistors/data.yaml', epochs=50,freeze=frozenLayers, project=save_dir + '/models', name='discontinuous')

file=open(save_dir + '/results/results discontin.txt', 'w')
file.write(str(results1))
file.close()

print("training contin")
results2 = model.train(data='/lab/micah/obj-det/datasets/Resistors/data.yaml', epochs=50,freeze=10, project=save_dir + '/models', name='continuous')

file=open(save_dir + '/results/results contin.txt', 'w')
file.write(str(results2))
file.close()

print("done training")



