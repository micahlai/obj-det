from yolov10.ultralytics import YOLOv10
import colors

layersToFreeze = []
quiet = False

def freeze_layer(trainer):
    model = trainer.model
    count = 0
    if(not quiet):
        print(f"Freezing {layersToFreeze.count} layers")
    freeze = layersToFreeze  # layers to freeze 
    for k, v in model.named_parameters(): 
        v.requires_grad = True  # train all layers 
        if any(x == k for x in freeze): 
            print(f'freezing {k}') 
            v.requires_grad = False 
            count += 1
    if(not quiet):
        print(f"{colors.bcolors.OKGREEN}{count} layers are freezed{colors.bcolors.ENDC}")



