import matplotlib.pyplot as plt
import os
import json
import math


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
def plotDataBar(mAPs, trainingTimes, name=""):
    figure, (ax1,ax2) = plt.subplots(2,1,figsize=(10,10))

    mAPVals = list(mAPs.values())
    ax1.bar(range(len(mAPs)),mAPVals, tick_label=list(mAPs.keys()))
    ax1.set_title("mAPs")
    low1 = min(mAPVals)
    high1 = max(mAPVals)
    ax1.set_ylim([(low1-0.5*(high1-low1)), (high1+0.5*(high1-low1))])

    for i in range(len(mAPs)):
        ax1.text(i, round(mAPVals[i],3), round(mAPVals[i],3), ha = 'center')

    trainingTimeVals = list(trainingTimes.values())
    ax2.bar(range(len(trainingTimes)),trainingTimeVals, tick_label=list(trainingTimes.keys()))
    ax2.set_title("Training Times")
    low2 = min(trainingTimeVals)
    high2 = max(trainingTimeVals)
    ax2.set_ylim([(low2-0.5*(high2-low2)), (high2+0.5*(high2-low2))])


    for i in range(len(trainingTimes)):
        ax2.text(i, round(trainingTimeVals[i],5), round(trainingTimeVals[i],5), ha = 'center')

    plt.draw()
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=60, ha='right')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=60, ha='right')

    if(name != ""):
        plt.suptitle(name, fontsize=14)
        plt.savefig(save_dir + f'/results/results [{name}].jpg', format='jpg')
    else:
        plt.savefig(save_dir + f'/results/results [{name}].jpg', format='jpg')

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

def createSubfoldersInResults(name):

    os.makedirs(save_dir + '/results/JSON DATA/' + name, exist_ok=True)
    os.makedirs(save_dir + '/results/freeze data/' + name, exist_ok=True)
    os.makedirs(save_dir + '/results/individual results/' + name, exist_ok=True)
    os.makedirs(save_dir + '/models/' + name, exist_ok=True)
