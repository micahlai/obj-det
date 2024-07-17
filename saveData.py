import matplotlib.pyplot as plt
import os
import json


def initialize(sd):
    global save_dir
    save_dir = sd
    if(save_dir == ""):
        save_dir = '/lab/micah/obj-det/testing runs/unnamed'
    os.makedirs(save_dir + '/results/JSON DATA', exist_ok=True)
    os.makedirs(save_dir + '/results/freeze data', exist_ok=True)
    os.makedirs(save_dir + '/results/individual results', exist_ok=True)
    os.makedirs(save_dir + '/models', exist_ok=True)

#input array
def plotDataLine(layersToFreeze, mAPs, trainingTimes):
    figure, axis = plt.subplots(2,2)

    axis[0,0].plot(layersToFreeze,mAPs)
    axis[0,0].set_title("Layers Frozen vs mAP")

    axis[0,1].plot(layersToFreeze,trainingTimes)
    axis[0,1].set_title("Layers Frozen vs Training Time")

    axis[1,0].scatter(mAPs,trainingTimes)
    for i, txt in enumerate(layersToFreeze):
        axis[1,0].annotate(txt, (mAPs[i], trainingTimes[i]))
    axis[1,0].set_title("mAP vs Training Time")

    plt.savefig(save_dir + '/results/results.eps', format='jpg')

#input dictionary
def plotDataBar(mAPs, trainingTimes):
    figure, (ax1,ax2) = plt.subplots(2,1,figsize=(5,10))

    ax1.bar(range(len(mAPs)),list(mAPs.values()), tick_label=list(mAPs.keys()))
    ax1.set_title("mAPs")

    ax2.bar(range(len(trainingTimes)),list(trainingTimes.values()), tick_label=list(trainingTimes.keys()))
    ax2.set_title("Training Times")

    plt.draw()
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha='right')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, ha='right')

    plt.savefig(save_dir + '/results/results.jpg', format='jpg')

def saveFile(name, data):
    file = open(save_dir + f'/results/{name}.txt','w')
    if(type(data) is str):
        file.write(str(data))
    elif(type(data) is list):
        for i in data:
            file.write(str(i) + '\n')
    elif(type(data) is dict):
        for key, value in data.items():
            file.write('%s:%s\n' % (key,value))
    file.close()

def saveJSON(name, data):
    with open(f"{save_dir}/results/JSON DATA/{name}.json", "w") as outfile: 
        json.dump(data, outfile)