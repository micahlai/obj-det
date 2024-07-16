from yolov10.ultralytics import YOLOv10

file = open('/lab/micah/obj-det/named_parameters.txt','w')

model = YOLOv10.from_pretrained('jameslahm/yolov10l')
for k, v in model.named_parameters():
    file.write(k.removeprefix('model.') + '\n')
file.close()

    