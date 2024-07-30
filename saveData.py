import matplotlib.pyplot as plt
import os
import json
import math
import matplotlib.cm as cm
import freezeDataLookup
import readyaml
import statistics

def initialize(sd, keepSubs = True, keepModels = True):
    global save_dir
    save_dir = sd
    if(save_dir == ""):
        save_dir = '/lab/micah/obj-det/testing runs/unnamed'
    
    os.makedirs(save_dir + '/results/', exist_ok=True)
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

def plotDataScatter(mAPs, trainingTimes, name="",file="",origin=False,relative=False,unfrozenKey = "unfrozen"):


    mAPVals = list(mAPs.values())
    trainingTimeVals = list(trainingTimes.values())
    if(relative):
        mAPVals=[x/mAPs[unfrozenKey] for x in mAPVals]
        trainingTimeVals=[x/trainingTimes[unfrozenKey] for x in trainingTimeVals]

    low1 = min(mAPVals)
    high1 = max(mAPVals)
    low2 = min(trainingTimeVals)
    high2 = max(trainingTimeVals)

    colorData = []
    for i in mAPs.keys():
        colorData.append(len(freezeDataLookup.lookupData(i)))

    plt.figure(figsize=(20,15))
    plt.scatter(trainingTimeVals,mAPVals,c=colorData,s=100)
    if(origin):
        plt.xlim([0, (high2+0.1*(high2-low2))])
        plt.ylim([0, (high1+0.1*(high1-low1))])
    else:
        plt.xlim([(low2-0.1*(high2-low2)), (high2+0.1*(high2-low2))])
        plt.ylim([(low1-0.1*(high1-low1)), (high1+0.1*(high1-low1))])


    plt.xlabel("Training Time")
    plt.ylabel("mAPs")
    for i, txt in enumerate(mAPs.keys()):
        plt.annotate(txt, (trainingTimeVals[i], mAPVals[i]),ha="center")
    plt.colorbar()

    relativeTitle = ""
    if(relative):
        relativeTitle = "Relative "
    if(name != ""):
        plt.suptitle(relativeTitle + name, fontsize=14)
        plt.savefig(save_dir + f'/results/{file}results [{name}].jpg', format='jpg')
    else:
        plt.suptitle(relativeTitle, fontsize=14)
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
        
        sizeData = []
        for i in mAPs[grad].keys():
            sizeData.append(readyaml.returnSizeAverage(i) * 3000)

        plt.figure(figsize=(20,15))
        plt.scatter(trainingTimeVals,mAPVals,c=colorData,s=sizeData,cmap="plasma")
        plt.xlim([(low2-0.1*(high2-low2)), (high2+0.1*(high2-low2))])
        plt.ylim([(low1-0.1*(high1-low1)), (high1+0.1*(high1-low1))])

        plt.xlabel("Training Time")
        plt.ylabel("mAPs")
        for i, txt in enumerate(mAPs[grad].keys()):
            plt.annotate(f"{txt},s:{round(readyaml.returnHWRatioSTDev(txt),3)},hwr:{round(readyaml.returnHWRatioSTDev(txt),3)}", (trainingTimeVals[i], mAPVals[i]),ha="center")
        plt.colorbar()

        if(unfrozenExists):
            plt.suptitle(f"{grad} (Relative to Unfrozen)", fontsize=14)
        else:
            plt.suptitle(grad, fontsize=14)
        plt.savefig(save_dir + f'/results/{file}results [{grad}].jpg', format='jpg')

        plt.close()

def plotDataScatterByGradientTotal(mAPs, trainingTimes, file="", unfrozenKey = "unfrozen", relativeToUnfrozen = True):
    defaultKeys = list(mAPs.keys())
    unfrozenExists = (unfrozenKey in defaultKeys) and relativeToUnfrozen
    keys = defaultKeys

    low1=0
    low2=0
    high1=0
    high2=0

    if(unfrozenExists):
        keys.remove(unfrozenKey)

    plt.figure(figsize=(20,15))
    for grad in keys:

        if(unfrozenExists):
            for dataset in mAPs[grad]:
                mAPVals.append(mAPs[grad][dataset]/mAPs[unfrozenKey][dataset])
                trainingTimeVals.append(trainingTimes[grad][dataset]/trainingTimes[unfrozenKey][dataset])
        else:
            mAPVals = list(mAPs[grad].values())
            trainingTimeVals = list(trainingTimes[grad].values())

        
        low1 = min(mAPVals + [low1])
        high1 = max(mAPVals + [high1])
        low2 = min(trainingTimeVals + [low2])
        high2 = max(trainingTimeVals + [low2])
        
        sizeData = []
        for i in mAPs[grad].keys():
            sizeData.append(readyaml.returnSizeAverage(i) * 3000)

        plt.scatter(trainingTimeVals,mAPVals,s=sizeData)

        

    plt.xlim([(low2-0.1*(high2-low2)), (high2+0.1*(high2-low2))])
    plt.ylim([(low1-0.1*(high1-low1)), (high1+0.1*(high1-low1))])
    plt.xlabel("Training Time")
    plt.ylabel("mAPs")
    plt.legend(keys, loc="best")

    if(unfrozenExists):
        plt.suptitle(f"Total : Relative to Unfrozen", fontsize=14)
    else:
        plt.suptitle("Total", fontsize=14)
    plt.savefig(save_dir + f'/results/{file}results.jpg', format='jpg')
    plt.close()

def plotDataLineByGradientTotal(mAPs, trainingTimes, name="",file="", unfrozenKey = "unfrozen",datasetAttribute = "size",dontIgnore=[""],ignoreDataset=[""]):
    defaultKeys = list(mAPs.keys())
    unfrozenExists = (unfrozenKey in defaultKeys)
    keys = defaultKeys
    if(unfrozenExists):
        keys.remove(unfrozenKey)
    else:
        return False
    
    if(dontIgnore != [""]):
        keys = dontIgnore

    freezeData = freezeDataLookup.getAllData()
    keys = sorted(keys, key=freezeData.get)

    plt.figure(figsize=(20,15))

    annotations = {}
    averages = {}

    for grad in keys:
        datasets = [x for x in list(mAPs[grad].keys()) if x not in ignoreDataset]
        xVals = []
        yVals = []
        for dataset in datasets:
            xVal = 0
            if(datasetAttribute == "size"):
                xVal=(readyaml.returnSizeAverage(dataset))
            elif(datasetAttribute == "classes"):
                xVal=(readyaml.returnClassCountDefaultDir(dataset))
            elif(datasetAttribute == "count"):
                xVal=(readyaml.returnTrainingSetCount(dataset))
            else:
                xVal=(readyaml.returnHWRatioSTDev(dataset))
            xVals.append(xVal)

            mAP = mAPs[grad][dataset]/mAPs[unfrozenKey][dataset]
            TTime = trainingTimes[grad][dataset]/trainingTimes[unfrozenKey][dataset]
            yVals.append(mAP/TTime)

            annotations[dataset] = xVal
        
        SyVals = [x for _, x in sorted(zip(xVals,yVals))]
        SxVals = sorted(xVals)
        averages[grad]=statistics.mean(SyVals)
        
        plt.plot(SxVals,SyVals)


    for key,val in annotations.items():
        plt.annotate(key, (val, 0.5),ha="center",fontsize=10,rotation=90)

    if(datasetAttribute == "size"):
        plt.xlabel("Average Size")
        plt.suptitle("Favoribility vs Average Bounding Box Size", fontsize=14)
    elif(datasetAttribute == "ratio"):
        plt.xlabel("STDev of HW Ratio")
        plt.suptitle("Favoribility vs STDEV of HW Ratio", fontsize=14)
    elif(datasetAttribute == "classes"):
        plt.xlabel("# of classes")
        plt.suptitle("Favoribility vs # of Classes", fontsize=14)
    elif(datasetAttribute == "count"):
        plt.xlabel("# of Training Images")
        plt.suptitle("Favoribility vs # of Training Images", fontsize=14)
    plt.ylabel("Favoribility")

    legendLabels = keys
    legendLabels = sorted(keys, key=averages.get)
    plt.legend([f"{x} ({freezeData[x]},{round(averages[x],3)})" for x in legendLabels], loc="best")
    plt.savefig(save_dir + f'/results/{file}results.jpg', format='jpg')
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