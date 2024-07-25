import matplotlib.pyplot as plt
import os
import json
import math
import matplotlib.cm as cm
import freezeDataLookup
import readyaml

def initialize(sd, keepSubs = True, keepModels = True):
    global save_dir
    save_dir = sd
    if(save_dir == ""):
        save_dir = '/lab/micah/obj-det/testing runs/unnamed'
    if(keepSubs):
        os.makedirs(save_dir + '/results/JSON DATA', exist_ok=True)
        os.makedirs(save_dir + '/results/freeze data', exist_ok=True)
        os.makedirs(save_dir + '/results/individual results', exist_ok=True)
    if(keepModels):
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

    plt.close()

#input dictionary
def plotDataBar(mAPs, trainingTimes, name="",file=""):
    figure, (ax1,ax2) = plt.subplots(2,1,figsize=(10,20))
    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.4, bottom = 0.1, top = 0.95)

    labels = []
    for i in mAPs.keys():
        labels.append(f"{i}:[{len(freezeDataLookup.lookupData(i))}]")

    mAPVals = list(mAPs.values())
    ax1.bar(range(len(mAPs)),mAPVals, tick_label=labels)
    ax1.set_title("mAPs")
    low1 = min(mAPVals)
    high1 = max(mAPVals)
    ax1.set_ylim([(low1-0.5*(high1-low1)), (high1+0.5*(high1-low1))])

    for i in range(len(mAPs)):
        ax1.text(i, round(mAPVals[i],3), round(mAPVals[i],3), ha = 'left', rotation = 90)

    trainingTimeVals = list(trainingTimes.values())
    ax2.bar(range(len(trainingTimes)),trainingTimeVals, tick_label=labels)
    ax2.set_title("Training Times")
    low2 = min(trainingTimeVals)
    high2 = max(trainingTimeVals)
    ax2.set_ylim([(low2-0.5*(high2-low2)), (high2+0.5*(high2-low2))])


    for i in range(len(trainingTimes)):
        ax2.text(i, round(trainingTimeVals[i],5), round(trainingTimeVals[i],5), ha = 'left', rotation = 90)

    plt.draw()
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=-90, ha='left')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=-90, ha='left')

    if(name != ""):
        plt.suptitle(name, fontsize=14)
        plt.savefig(save_dir + f'/results/{file}results [{name}].jpg', format='jpg')
    else:
        plt.savefig(save_dir + f'/results/{file}results.jpg', format='jpg')

    plt.close()

def plotDataScatter(mAPs, trainingTimes, name="",file=""):

    mAPVals = list(mAPs.values())
    trainingTimeVals = list(trainingTimes.values())

    low1 = min(mAPVals)
    high1 = max(mAPVals)
    low2 = min(trainingTimeVals)
    high2 = max(trainingTimeVals)

    colorData = []
    for i in mAPs.keys():
        colorData.append(len(freezeDataLookup.lookupData(i)))

    plt.figure(figsize=(20,15))
    plt.scatter(trainingTimeVals,mAPVals,c=colorData,s=100)
    plt.xlim([(low2-0.1*(high2-low2)), (high2+0.1*(high2-low2))])
    plt.ylim([(low1-0.1*(high1-low1)), (high1+0.1*(high1-low1))])

    plt.xlabel("Training Time")
    plt.ylabel("mAPs")
    for i, txt in enumerate(mAPs.keys()):
        plt.annotate(txt, (trainingTimeVals[i], mAPVals[i]),ha="center")
    plt.colorbar()


    if(name != ""):
        plt.suptitle(name, fontsize=14)
        plt.savefig(save_dir + f'/results/{file}results [{name}].jpg', format='jpg')
    else:
        plt.savefig(save_dir + f'/results/{file}results.jpg', format='jpg')

    plt.close()

def plotDataCombined(mAPs, trainingTimes, name="",file=""):

    mAPVals = list(mAPs.values())
    trainingTimeVals = list(trainingTimes.values())

    low1 = min(mAPVals)
    high1 = max(mAPVals)
    low2 = min(trainingTimeVals)
    high2 = max(trainingTimeVals)

    labels = []
    for i in mAPs.keys():
        labels.append(f"{i}:[{len(freezeDataLookup.lookupData(i))}]")

    colorData = []
    for i in mAPs.keys():
        colorData.append(len(freezeDataLookup.lookupData(i)))

    plt.figure(figsize=(40,15))
    plt.subplot(2,2,1)
    plt.bar(range(len(mAPs)),mAPVals, tick_label=labels)
    plt.ylim([(low1-0.5*(high1-low1)), (high1+0.5*(high1-low1))])
    plt.xticks(rotation=90, ha='left')
    plt.title("mAP")

    for i in range(len(mAPs)):
        plt.text(i, round(mAPVals[i],3), round(mAPVals[i],3), ha = 'left', rotation = 90)


    plt.subplot(2,2,3)
    plt.bar(range(len(trainingTimes)),trainingTimeVals, tick_label=labels)
    plt.ylim([(low2-0.5*(high2-low2)), (high2+0.5*(high2-low2))])
    plt.xticks(rotation=90, ha='left')
    plt.title("Training Time")

    for i in range(len(trainingTimes)):
        plt.text(i, round(trainingTimeVals[i],5), round(trainingTimeVals[i],5), ha = 'left', rotation = 90)

    plt.subplot(1,2,2)
    plt.scatter(trainingTimeVals,mAPVals,c=colorData)
    plt.xlim([(low2-0.1*(high2-low2)), (high2+0.1*(high2-low2))])
    plt.ylim([(low1-0.1*(high1-low1)), (high1+0.1*(high1-low1))])

    plt.xlabel("Training Time")
    plt.ylabel("mAPs")
    for i, txt in enumerate(mAPs.keys()):
        plt.annotate(txt, (trainingTimeVals[i], mAPVals[i]),ha="center",fontsize=5)
    plt.colorbar()
    plt.tight_layout()


    if(name != ""):
        plt.suptitle(name, fontsize=20, va="bottom")
        plt.savefig(save_dir + f'/results/{file}results [{name}].jpg', format='jpg',dpi=300)
    else:
        plt.savefig(save_dir + f'/results/{file}results.jpg', format='jpg',dpi=300)

    plt.close()

def plotDataScatterByGradient(mAPs, trainingTimes, file="", unfrozenKey = "unfrozen", relativeToUnfrozen = True):
    defaultKeys = list(mAPs.keys())
    unfrozenExists = (unfrozenKey in defaultKeys) and relativeToUnfrozen
    keys = defaultKeys

    if(unfrozenExists):
        keys.remove(unfrozenKey)
    for grad in keys:
        # mAPVals = list(mAPs[grad].values())
        # trainingTimeVals = list(trainingTimes[grad].values())

        # mAPUnfrozenVal = list(mAPs[unfrozenKey].values())
        # trainingTimeUnfrozenVal = list(trainingTimes[unfrozenKey].values())
        # print(mAPVals)
        # if(unfrozenExists):
        #     for i in mAPVals:
        #         mAPVals[i] = mAPVals[i]/mAPUnfrozenVal[i]
        #     for i in trainingTimes:
        #         trainingTimeVals[i] = trainingTimeVals[i]/trainingTimeUnfrozenVal[i]
        mAPVals = []
        trainingTimeVals = []

        if(unfrozenExists):
            for dataset in mAPs[grad]:
                mAPVals.append(mAPs[grad][dataset]/mAPs[unfrozenKey][dataset])
                trainingTimeVals.append(trainingTimes[grad][dataset]/trainingTimes[unfrozenKey][dataset])
        else:
            mAPVals = list(mAPs[grad].values())
            trainingTimeVals = list(trainingTimes[grad].values())

        low1 = min(mAPVals)
        high1 = max(mAPVals)
        low2 = min(trainingTimeVals)
        high2 = max(trainingTimeVals)

        colorData = []
        for i in mAPs[grad].keys():
            colorData.append(readyaml.returnClassCountDefaultDir(i))

        plt.figure(figsize=(20,15))
        plt.scatter(trainingTimeVals,mAPVals,c=colorData,s=100,cmap="plasma")
        plt.xlim([(low2-0.1*(high2-low2)), (high2+0.1*(high2-low2))])
        plt.ylim([(low1-0.1*(high1-low1)), (high1+0.1*(high1-low1))])

        plt.xlabel("Training Time")
        plt.ylabel("mAPs")
        for i, txt in enumerate(mAPs[grad].keys()):
            plt.annotate(txt, (trainingTimeVals[i], mAPVals[i]),ha="center")
        plt.colorbar()

        if(unfrozenExists):
            plt.suptitle(f"{grad} (Relative to Unfrozen)", fontsize=14)
        else:
            plt.suptitle(grad, fontsize=14)
        plt.savefig(save_dir + f'/results/{file}results [{grad}].jpg', format='jpg')

        plt.close()

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

def createSubfolderwithNameInResults(path):
    os.makedirs(save_dir + '/results/' + path, exist_ok=True)
