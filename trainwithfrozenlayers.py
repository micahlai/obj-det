from yolov10.ultralytics import YOLOv10
import time
import matplotlib.pyplot as plt

models = []
model = YOLOv10.from_pretrained('jameslahm/yolov10l')

dataset = '/lab/micah/obj-det/garage dataset/data.yaml'

defaultEpochs = 100
layersToFreeze = [0,5,10,17,20]

results = []
mAPs = []
#operationCounts = []
trainingTimes = []
saveDirs = []

for i in range(len(layersToFreeze)):
    startTS = time.time()
    models.append(model)
    result = models[i].train(data=dataset, epochs=defaultEpochs, freeze=layersToFreeze[i])
    
    trainingTime = (time.time()-startTS)/3600


    results.append(result)
    saveDirs.append(result.save_dir)
    mAPs.append(result.results_dict['metrics/mAP50(B)'])
    trainingTimes.append(trainingTime)

    file = open('/lab/micah/obj-det/results/trainingResults %s.txt' % str(i),'w')
    file.write(str(result))
    file.close()

figure, axis = plt.subplots(2,2)

axis[0,0].plot(layersToFreeze,mAPs)
axis[0,0].set_title("Layers Frozen vs mAP")

axis[0,1].plot(layersToFreeze,trainingTimes)
axis[0,1].set_title("Layers Frozen vs Training Time")

axis[1,0].scatter(mAPs,trainingTimes)
for i, txt in enumerate(layersToFreeze):
    axis[1,0].annotate(txt, (mAPs[i], trainingTimes[i]))
axis[1,0].set_title("mAP vs Training Time")

plt.savefig('/lab/micah/obj-det/results/results.eps', format='eps')

file = open('/lab/micah/obj-det/results/save directories.txt','w')
file.writelines(str(saveDirs))
file.close()



