from yolov10.ultralytics import YOLOv10

layersToFreeze = []

def freeze_layer(trainer):
    model = trainer.model
    count = 0
    print(f"Freezing {layersToFreeze.count} layers")
    freeze = layersToFreeze  # layers to freeze 
    for k, v in model.named_parameters(): 
        v.requires_grad = True  # train all layers 
        if any(x in k for x in freeze): 
            print(f'freezing {k}') 
            v.requires_grad = False 
            count += 1
    print(f"{count} layers are freezed.")



