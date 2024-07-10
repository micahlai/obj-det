import gradio as gr
import cv2
import tempfile
from yolov10.ultralytics import YOLOv10
import os
import time

save_dir = '/lab/micah/obj-det/testing runs/7-10 discontin'
data_dir = '/lab/micah/obj-det/datasets/Hard hat uni/data.yaml'

os.makedirs(save_dir + '/results', exist_ok=True)
os.makedirs(save_dir + '/models', exist_ok=True)

model = YOLOv10.from_pretrained('jameslahm/yolov10l')

#!yolo task=detect mode=train epochs=10 batch=32 plots=True \
#model=/lab/micah/yolov10/yolov10n.pt
#data=/lab/micah/yolov10/hard hat uni/data.yaml

frozenLayers = [2,3,4,5,6,7,8,9,10]

startTime = time.time()
timeToTrain = [0,0,0]

print("training discontin")
results1 = model.train(data=data_dir, epochs=100,freeze=frozenLayers, project=save_dir + '/models', name='discontinuous')

timeToTrain[0]=time.time()-startTime
startTime=time.time()

file=open(save_dir + '/results/results discontin.txt', 'w')
file.write(str(results1))
file.close()

print("training contin10")
results2 = model.train(data=data_dir, epochs=100,freeze=10, project=save_dir + '/models', name='continuous10')

timeToTrain[1]=time.time()-startTime
startTime=time.time()

file=open(save_dir + '/results/results contin10.txt', 'w')
file.write(str(results2))
file.close()

print("training contin0")
results3 = model.train(data=data_dir, epochs=100, project=save_dir + '/models', name='continuous0')

timeToTrain[2]=time.time()-startTime
startTime=time.time()

file=open(save_dir + '/results/results contin0.txt', 'w')
file.write(str(results3))
file.close()

file=open(save_dir + '/results/overallResults', 'w')
file.write(str(timeToTrain))
file.close()

print("done training")



