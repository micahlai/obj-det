import gradio as gr
import cv2
import tempfile
from yolov10.ultralytics import YOLOv10
import time
import os
import callbackFreezer

save_dir = '/lab/micah/obj-det/testing runs/7-15 branch test take 2'
freeze_set_path = '/lab/micah/obj-det/freeze set.txt'
dataset = '/lab/micah/obj-det/datasets/garage dataset/data.yaml'

os.makedirs(save_dir + '/results', exist_ok=True)
os.makedirs(save_dir + '/models', exist_ok=True)

startTS = time.time()
        
model = YOLOv10.from_pretrained('jameslahm/yolov10l')

freeze_set = open(freeze_set_path, 'r').read().split('\n')

callbackFreezer.layersToFreeze = freeze_set
model.add_callback("on_train_start", callbackFreezer.freeze_layer)

result = model.train(data='/lab/micah/obj-det/datasets/garage dataset/data.yaml',
                                  epochs=100,
                                  project=save_dir + '/models',
                                  name='branch test')

#result = model.train(data=dataset,epochs=100,freeze=freeze_set,project=save_dir + '/models',name='branch freeze')
        
# trainingTime = (time.time()-startTS)/3600

# file = open(save_dir + '/results/trainingResults.txt','w')
# file.write(str(result))
# file.close()

# file = open(save_dir + '/results/overallResults.txt','w')
# file.write(str(trainingTime))
# file.close()

