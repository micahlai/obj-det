import sys
import time
import saveData
import torch
from yolov10.ultralytics import YOLOv10

arg = sys.argv[1]

print("starting program")
saveData.initialize('/lab/micah/obj-det/test-parallel')
saveData.saveFile(arg,"If you see this, it worked")
print(arg)

model = YOLOv10.from_pretrained('jameslahm/yolov10l')
torch.save(model, f'{arg} yolov10l.pt')