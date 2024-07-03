from yolov10.ultralytics import YOLOv10
import time
import matplotlib.pyplot as plt
import os

save_dir = '/lab/micah/obj-det/testing runs/7-2'

os.makedirs(save_dir + '/results', exist_ok=True)
os.makedirs(save_dir + '/models', exist_ok=True)

# datasets = ['garage dataset',
#             'Fall detection',
#             'hard hat uni',
#             'People in painting',
#             'Resistors']

datasets = ['garage dataset',
            'hard hat uni',
            'People in painting',
            'Resistors']

defaultEpochs = 100
layersToFreeze = [0,4,8,12,16,20]

mAPs = []
#operationCounts = []
trainingTimes = []
saveDirs = []

mean_mAPs = [0] * len(layersToFreeze)
mean_trainingTimes = [0] * len(layersToFreeze)

overallStartTS = time.time()

for i  in range(len(datasets)):
    mAPs.append([])
    trainingTimes.append([])
    saveDirs.append([])
    for j in range(len(layersToFreeze)):
        startTS = time.time()
        
        model = YOLOv10.from_pretrained('jameslahm/yolov10l')
        print("Training with " + datasets[i] + " and layers frozen: " + str(layersToFreeze[j]))
        result = model.train(data='/lab/micah/obj-det/datasets/' + datasets[i] + '/data.yaml',
                                  epochs=defaultEpochs,
                                  freeze=layersToFreeze[j],
                                  project=save_dir + '/models',
                                  name=datasets[i] + str(layersToFreeze[j]))
        
        trainingTime = (time.time()-startTS)/3600

        mAP = result.results_dict['metrics/mAP50(B)']
        
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

axis[1,0].scatter(mean_trainingTimes,mean_mAPs)
for i, txt in enumerate(layersToFreeze):
    axis[1,0].annotate(txt, (mean_trainingTimes[i], mean_mAPs[i]))
axis[1,0].set_title("Training Time vs mAP")

plt.savefig(save_dir + '/results/mean results.eps', format='eps')

totalRuntime = time.time() - overallStartTS
file = open(save_dir + '/results/overallResults.txt','w')

file.write('Total runtime: ' + str(totalRuntime/3600) + '\n')

print("Testing Complete")



