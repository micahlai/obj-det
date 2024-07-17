from yolov10.ultralytics import YOLOv10
import time
import matplotlib.pyplot as plt
import os

save_dir = '/lab/micah/obj-det/testing runs/7-2'

os.makedirs(save_dir + '/results', exist_ok=True)
os.makedirs(save_dir + '/models', exist_ok=True)
#use save data class


models = []
model = YOLOv10.from_pretrained('jameslahm/yolov10l')

#load freeze set data and save to dictionary
#key is file name, value is the data

dataset = '/lab/micah/obj-det/garage dataset/data.yaml'

defaultEpochs = 100
layersToFreeze = {}

#make these dictionaries
results = []
mAPs = []
trainingTimes = []

for key,val in layersToFreeze.items():
	#train with exception and save the mAP and training time data after each train
	#start time should be in callback function --> add two callbacks (freezing and start time)
    startTS = time.time()
    models.append(model)
    result = models[i].train(data=dataset, epochs=defaultEpochs, freeze=layersToFreeze[i])
    
    trainingTime = (time.time()-startTS)/3600


    results.append(result)
    mAPs.append(result.results_dict['metrics/mAP50(B)'])
    trainingTimes.append(trainingTime)

    file = open('/lab/micah/obj-det/results/trainingResults %s.txt' % str(i),'w')
    file.write(str(result))
    file.close()
    #save data to json

#save all of data to json
#graph that shit
    
    
    
    
    #IM GONNA KMS ASJFHAGSJN
