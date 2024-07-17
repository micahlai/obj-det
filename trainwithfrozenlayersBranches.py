from yolov10.ultralytics import YOLOv10
import time
import os

import saveData
import callbackFreezer
import colors

#save data init
save_dir = '/lab/micah/obj-det/testing runs/7-17 Full Test'
saveData.initialize(save_dir)

#read freeze set data
freeze_data = {}
freeze_set_path = '/lab/micah/obj-det/freeze sets'
for root, dirs, files in os.walk(freeze_set_path):
    for f in files:
        if(f.endswith('.txt')):
            freeze_data[f.removesuffix('.txt')]=open(os.path.join(root, f), 'r').read().split('\n')
#sort dictionary
freeze_data=dict(sorted(freeze_data.items(), key=lambda item:len(item[1])))

models = {}
model = YOLOv10.from_pretrained('jameslahm/yolov10l')

dataset = '/lab/micah/obj-det/datasets/garage dataset/data.yaml'

defaultEpochs = 2

mAPs = {}
trainingTimes = {}

def startTimer(trainer):
    global startTS
    startTS = time.time()

model.add_callback("on_train_start", callbackFreezer.freeze_layer)
model.add_callback("on_train_start", startTimer)

step = 1

for key,val in freeze_data.items():
    print(f"{colors.bcolors.OKCYAN}Now training {key} for {defaultEpochs} epochs{colors.bcolors.ENDC}")
    callbackFreezer.layersToFreeze = val
	#train with exception and save the mAP and training time data after each train
    try:
        models[key] = model
        result = model.train(data=dataset,
                            epochs=defaultEpochs,
                            project=save_dir + '/models',
                            name=key)
        
        trainingTime = (time.time()-startTS)/3600

        mAPs[key] = result.results_dict['metrics/mAP50(B)']
        trainingTimes[key] = trainingTime

        saveData.saveFile(f"individual results/{key}",str(result))
        saveData.saveFile(f"freeze data/{key}",val)
        saveData.saveJSON(f"mAP step {step}:{key}",mAPs)
        saveData.saveJSON(f"time step {step}:{key}",trainingTimes)


    except Exception as e:
        print(f"{colors.bcolors.WARNING}An error occured when training [{key}]{colors.bcolors.ENDC}")
        print(colors.bcolors.FAIL + str(e) + colors.bcolors.ENDC)

    step += 1

saveData.saveJSON("FINAL mAP RESULTS",mAPs)
saveData.saveJSON("FINAL TRAINING TIME RESULTS",trainingTimes)

saveData.plotDataBar(mAPs,trainingTimes)

print(f"{colors.bcolors.OKBLUE}Finished testing{colors.bcolors.ENDC}")
