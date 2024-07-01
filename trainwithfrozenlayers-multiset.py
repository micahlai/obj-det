from yolov10.ultralytics import YOLOv10
import time
import matplotlib.pyplot as plt
import os

models = []
model = YOLOv10.from_pretrained('jameslahm/yolov10l')
save_dir = '/lab/micah/obj-det/testing runs/7-1 multidataset trial2'

os.makedirs(save_dir + '/results', exist_ok=True)
os.makedirs(save_dir + '/models', exist_ok=True)

# datasets = ['garage dataset',
#             'Fall detection',
#             'hard hat uni',
#             'People in painting',
#             'Resistors']

datasets = ['garage dataset',
            'Fall detection']

defaultEpochs = 2
#layersToFreeze = [0,3,6,9,12,15,18,21]
layersToFreeze = [0,3]

results = []
mAPs = []
#operationCounts = []
trainingTimes = []
saveDirs = []

mean_mAPs = [0] * len(layersToFreeze)
mean_trainingTimes = [0] * len(layersToFreeze)

overallStartTS = time.time()

for i  in range(len(datasets)):
    results.append([])
    mAPs.append([])
    trainingTimes.append([])
    saveDirs.append([])
    models.append([])

    for j in range(len(layersToFreeze)):
        startTS = time.time()
        models[i].append(model)
        print("Training with " + datasets[i] + " and layers frozen: " + str(layersToFreeze[j]))
        result = models[i][j].train(data='/lab/micah/obj-det/datasets/' + datasets[i] + '/data.yaml',
                                  epochs=defaultEpochs,
                                  freeze=layersToFreeze[j],
                                  project=save_dir + '/models',
                                  name=datasets[i] + str(layersToFreeze[j]))
        
        trainingTime = (time.time()-startTS)/3600

        mAP = result.results_dict['metrics/mAP50(B)']
        
        results[i].append(result)
        saveDirs[i].append(result.save_dir)
        mAPs[i].append(mAP)
        trainingTimes[i].append(trainingTime)

        mean_mAPs[j] += mAP / len(datasets)
        mean_trainingTimes[j] += trainingTime / len(datasets)

        file = open(save_dir + '/results/trainingResults ' + datasets[i] +  ' ' + str(layersToFreeze[j]) + '.txt','w')
        file.write(str(result))
        file.close()
    
    

    figure, axis = plt.subplots(2,2)

    axis[0,0].plot(layersToFreeze,mAPs[i])
    axis[0,0].set_title("Layers Frozen vs mAP")

    axis[0,1].plot(layersToFreeze,trainingTimes[i])
    axis[0,1].set_title("Layers Frozen vs Training Time")

    axis[1,0].scatter(mAPs[i],trainingTimes[i])
    for j, txt in enumerate(layersToFreeze):
        axis[1,0].annotate(txt, (mAPs[i][j], trainingTimes[i][j]))
    axis[1,0].set_title("mAP vs Training Time")

    plt.savefig(save_dir + '/results/results %s.eps' % datasets[i], format='eps')

    file = open(save_dir + '/results/save directories %s.txt' % datasets[i],'w')
    file.writelines(str(saveDirs[i]))
    file.close()

figure, axis = plt.subplots(2,2)

axis[0,0].plot(layersToFreeze,mean_mAPs)
axis[0,0].set_title("Layers Frozen vs mAP")

axis[0,1].plot(layersToFreeze,mean_trainingTimes)
axis[0,1].set_title("Layers Frozen vs Training Time")

axis[1,0].scatter(mean_mAPs,mean_trainingTimes)
for i, txt in enumerate(layersToFreeze):
    axis[1,0].annotate(txt, (mean_mAPs[i], mean_trainingTimes[i]))
axis[1,0].set_title("mAP vs Training Time")

plt.savefig(save_dir + '/results/mean results.eps', format='eps')

totalRuntime = time.time() - overallStartTS
file = open(save_dir + '/results/overallResults.txt','w')

file.write('Total runtime: ' + str(totalRuntime) + '\n')
file.write('mean_mAPs')
file.writelines(mean_mAPs)
file.write('\n mean_trainingTimes')
file.writelines(mean_trainingTimes)
file.write('\n All mAPs')
for i in len(datasets):
    file.write('\n' + datasets[i])
    file.writelines(mAPs[i])
file.write('\n All Training Times')
for i in len(datasets):
    file.write('\n' + datasets[i])
    file.writelines(trainingTimes[i])



