from yolov10.ultralytics import YOLOv10
import time
import os

import saveData
import callbackFreezer
import colors
import readyaml

#save data init
save_dir = '/lab/micah/obj-det/testing runs/7-30 separate branch multidatset'
saveData.initialize(save_dir)

#read freeze set data
freeze_data = {}
freeze_set_path = '/lab/micah/obj-det/freeze set/freeze sets 7-30'
for root, dirs, files in os.walk(freeze_set_path):
    for f in files:
        if(f.endswith('.txt')):
            freeze_data[f.removesuffix('.txt')]=open(os.path.join(root, f), 'r').read().split('\n')
#sort dictionary
freeze_data=dict(sorted(freeze_data.items(), key=lambda item:len(item[1])))


model = YOLOv10.from_pretrained('jameslahm/yolov10l')

dataset_home_dir = '/lab/micah/obj-det/datasets/'
datasets = ['bodywash',
            'playing cards',
            'carbike',
            'grocery store']

defaultEpochs = 100

def startTimer(trainer):
    global startTS
    startTS = time.time()


model.add_callback("on_train_start", callbackFreezer.freeze_layer)
model.add_callback("on_train_start", startTimer)

for i in datasets:

    saveData.createSubfoldersInResults(i)

    mAPs = {}
    trainingTimes = {}
    models = {}

    yamlpath = dataset_home_dir + i + '/data.yaml'
    datasetClassCount = readyaml.returnClassCount(yamlpath)

    step = 1

    print(f"{colors.bcolors.HEADER}Now training [{i}] : Classes [{datasetClassCount}]{colors.bcolors.ENDC}")

    for key,val in freeze_data.items():
        print(f"{colors.bcolors.OKCYAN}Now training [{key}] for [{defaultEpochs}] epochs on [{i}]{colors.bcolors.ENDC}")
        callbackFreezer.layersToFreeze = val
        #train with exception and save the mAP and training time data after each train
        try:
            models[key] = model
            result = model.train(data=yamlpath,
                                epochs=defaultEpochs,
                                project=save_dir + '/models/' + i,
                                name=key)
            
            trainingTime = (time.time()-startTS)/3600

            mAPs[key] = result.results_dict['metrics/mAP50(B)']
            trainingTimes[key] = trainingTime

            saveData.saveFile(f"individual results/{i}/{key}",str(result))
            saveData.saveFile(f"freeze data/{i}/{key}",val)
            saveData.saveJSON(f"{i}/mAP step {step}:{key}",mAPs)
            saveData.saveJSON(f"{i}/time step {step}:{key}",trainingTimes)


        except Exception as e:
            print(f"{colors.bcolors.WARNING}An error occured when training [{key}] on [{i}]{colors.bcolors.ENDC}")
            print(colors.bcolors.FAIL + str(e) + colors.bcolors.ENDC)

        step += 1
    try:
        saveData.saveJSON(f"{i}/FINAL mAP RESULTS",mAPs)
        saveData.saveJSON(f"{i}/FINAL TRAINING TIME RESULTS",trainingTimes)

        saveData.plotDataBar(mAPs,trainingTimes,name=f"{i} : NC({datasetClassCount})")
    except Exception as e:
        print(f"{colors.bcolors.WARNING}An error occured when saving results for [{i}]{colors.bcolors.ENDC}")
        print(colors.bcolors.FAIL + str(e) + colors.bcolors.ENDC)

    print(f"{colors.bcolors.OKBLUE}Finished testing [{i}]{colors.bcolors.ENDC}")
print(f"{colors.bcolors.BOLD}Finished testing all datasets{colors.bcolors.ENDC}")
