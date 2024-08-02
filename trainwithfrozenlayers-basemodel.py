from yolov10.ultralytics import YOLOv10
import time

import saveData
import colors
import readyaml
import readJSONData

completeStartTime = time.time()

#save data init
save_dir = '/lab/micah/obj-det/testing runs/8-1 different base model test'
saveData.initialize(save_dir)

baseModels = {}
baseModelTemplates = ['n','s','m','b','l','x']

for i in baseModelTemplates:
    baseModels[i] = YOLOv10.from_pretrained(f'jameslahm/yolov10{i}')

dataset_home_dir = '/lab/micah/obj-det/datasets/'
datasets = ['football',
            'hazard signs',
            'construction',
            'rock paper scissors',
            'carbike']

defaultEpochs = 100

def startTimer(trainer):
    global startTS
    startTS = time.time()

for i in baseModels.values():
    i.add_callback("on_train_start", startTimer)

for i in datasets:

    saveData.createSubfoldersInResults(i)

    mAPs = {}
    trainingTimes = {}
    models = {}

    yamlpath = dataset_home_dir + i + '/data.yaml'
    datasetClassCount = readyaml.returnClassCount(yamlpath)

    step = 1

    print(f"{colors.bcolors.HEADER}Now training [{i}] : Classes [{datasetClassCount}]{colors.bcolors.ENDC}")

    for key,val in baseModels.items():
        print(f"{colors.bcolors.OKCYAN}Now training [{key}] for [{defaultEpochs}] epochs on [{i}]{colors.bcolors.ENDC}")
        #train with exception and save the mAP and training time data after each train
        try:
            models[key] = val
            result = val.train(data=yamlpath,
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

totalTime = (time.time() - completeStartTime)/3600
print(f"{colors.bcolors.BOLD}Finished testing all datasets in [{totalTime} hours]{colors.bcolors.ENDC}")
saveData.saveFile("Total Time", totalTime)

