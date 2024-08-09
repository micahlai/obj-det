from yolov10.ultralytics import YOLOv10
import time
import os

import saveData
import callbackFreezer
import colors
import readyaml

import sys
import printWithHost
        

completeStartTime = time.time()

#save data init
save_dir = '/lab/micah/obj-det/testing runs/8-8 alternate freeze set'
saveData.initialize(save_dir)
printWithHost.initialize(save_dir,log=True)
saveData.createSubfolderwithNameInResults("failures")

#read freeze set data
freeze_data = {}
freeze_set_path = '/lab/micah/obj-det/freeze set/freeze sets 7-19'
for root, dirs, files in os.walk(freeze_set_path):
    for f in files:
        if(f.endswith('.txt')):
            freeze_data[f.removesuffix('.txt')]=open(os.path.join(root, f), 'r').read().split('\n')
#sort dictionary
freeze_data=dict(sorted(freeze_data.items(), key=lambda item:len(item[1])))


os.environ['YOLO_VERBOSE'] = 'False'
model = YOLOv10.from_pretrained('jameslahm/yolov10l')

dataset_home_dir = '/lab/micah/obj-det/datasets/'

datasetArg = ""
for i in range(len(sys.argv)-1):
    datasetArg += sys.argv[i+1] + " "
datasetArg = datasetArg.rstrip()

pathExists = False
for root, dirs, files in os.walk(dataset_home_dir):
    for d in dirs:
        if(d == datasetArg):
            pathExists=True

if(not pathExists):
    printWithHost.hostPrint(f"Dataset: {datasetArg} does not exist",colors.bcolors.WARNING)
    exit()

defaultEpochs = 100

def startTimer(trainer):
    global startTS
    startTS = time.time()



model.add_callback("on_train_start", callbackFreezer.freeze_layer)
model.add_callback("on_train_start", startTimer)


saveData.createSubfoldersInResults(datasetArg)

mAPs = {}
trainingTimes = {}
models = {}

yamlpath = dataset_home_dir + datasetArg + '/data.yaml'
datasetClassCount = readyaml.returnClassCount(yamlpath)

step = 1

printWithHost.hostPrint(f"Now training [{datasetArg}] : Classes [{datasetClassCount}]",colors.bcolors.HEADER)

keys = list(freeze_data.keys())
for key,val in freeze_data.items():
    printWithHost.hostPrint(f"Now training [{key}] for [{defaultEpochs}] epochs on [{datasetArg}] ({keys.index(key)+1}/{len(keys)})",colors.bcolors.OKCYAN)
    callbackFreezer.layersToFreeze = val
    callbackFreezer.quiet=True
    #train with exception and save the mAP and training time data after each train
    try:
        models[key] = model
        result = model.train(data=yamlpath,
                            epochs=defaultEpochs,
                            project=save_dir + '/models/' + datasetArg,
                            name=key,
                            verbose=False,
                            device=[0,1],
                            data_parallel=True)
        
        trainingTime = (time.time()-startTS)/3600

        mAPs[key] = result.results_dict['metrics/mAP50(B)']
        trainingTimes[key] = trainingTime

        saveData.saveFile(f"individual results/{datasetArg}/{key}",str(result))
        saveData.saveFile(f"freeze data/{datasetArg}/{key}",val)
        saveData.saveJSON(f"{datasetArg}/mAP step {step}:{key}",mAPs)
        saveData.saveJSON(f"{datasetArg}/time step {step}:{key}",trainingTimes)
        saveData.saveJSON(f"{datasetArg}/FINAL mAP RESULTS",mAPs)
        saveData.saveJSON(f"{datasetArg}/FINAL TRAINING TIME RESULTS",trainingTimes)


    except Exception as e:
        printWithHost.hostPrint(f"An error occured when training [{key}] on [{datasetArg}]",colors.bcolors.WARNING)
        printWithHost.hostPrint(str(e),colors.bcolors.FAIL)
        
        saveData.saveFile(f"failures/Train {datasetArg}:{key}",str(e))

    step += 1
try:
    saveData.saveJSON(f"{datasetArg}/FINAL mAP RESULTS",mAPs)
    saveData.saveJSON(f"{datasetArg}/FINAL TRAINING TIME RESULTS",trainingTimes)

    #saveData.plotDataBar(mAPs,trainingTimes,name=f"{datasetArg} : NC({datasetClassCount})")
except Exception as e:
    printWithHost.hostPrint(f"An error occured when saving results for [{datasetArg}]",colors.bcolors.WARNING)
    printWithHost.hostPrint(str(e),colors.bcolors.FAIL)
    saveData.saveFile(f"failures/Save {datasetArg}:{key}",str(e))

totalTime = (time.time() - completeStartTime)/3600
printWithHost.hostPrint(f"Finished testing [{datasetArg}] in [{totalTime} hours]",colors.bcolors.OKBLUE)
saveData.saveFile(f"Total Time {datasetArg}", str(totalTime))

#readJSONData.combineResults(additionalTP=[save_dir], save_dir="testing runs/7-31 separate branch combined results")

