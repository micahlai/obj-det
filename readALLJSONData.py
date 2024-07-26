import saveData
import json
import readyaml
import freezeDataLookup
import colors

trial1path = "testing runs/7-19 separate branch multidatset/results"
trial2path = "testing runs/7-18 separate branch multidatset/results"

dataset_home_dir = '/lab/micah/obj-det/datasets/'
datasets = ['part number',
            'garage dataset',
            'Resistors',
            'teeth',
            'toyota',
            'hard hat uni',
            'People in painting']

freeze_data = freezeDataLookup.getAllData()

printNoPathError = False

def getOrder(key):
    return freeze_data[key]

allmAP = {}
allTT = {}

keys = []

save_dir = "testing runs/7-25 separate branch combined results"
saveData.initialize(save_dir,keepSubs=False,keepModels=False)
saveData.createSubfolderwithNameInResults('dataset')
saveData.createSubfolderwithNameInResults('layer')

#print(freeze_data)

for i in datasets:

    mAP_path1 = f"{trial1path}/JSON DATA/{i}/FINAL mAP RESULTS.json"
    mAP_path2 = f"{trial2path}/JSON DATA/{i}/FINAL mAP RESULTS.json"
    trainTime_path1 = f"{trial1path}/JSON DATA/{i}/FINAL TRAINING TIME RESULTS.json"
    trainTime_path2 = f"{trial2path}/JSON DATA/{i}/FINAL TRAINING TIME RESULTS.json"

    mAPs1 = {}
    trainTime1 = {}
    mAPs2 = {}
    trainTime2 = {}

    mAPs = {}
    trainTime = {}


    try:
        with open(mAP_path1) as json_file:
            mAPs1 = json.load(json_file)
    except:
        if(printNoPathError):
            print(i + " doesn't exist for mAP in " + trial1path)

    try:
        with open(trainTime_path1) as json_file:
            trainTime1 = json.load(json_file)
    except:
        if(printNoPathError):
            print(i + " doesn't exist for training time in " + trial1path)

    try:
        with open(mAP_path2) as json_file:
            mAPs2 = json.load(json_file)
    except:
        if(printNoPathError):
            print(i + " doesn't exist for mAP in " + trial2path)

    try:
        with open(trainTime_path2) as json_file:
            trainTime2 = json.load(json_file)
    except:
        if(printNoPathError):
            print(i + " doesn't exist for training time in " + trial2path)

    for key, value in mAPs1.items():
        if key in mAPs2:
            mAPs[key] = (mAPs1[key] + mAPs2[key])/2
        else:
            mAPs[key] = mAPs1[key]

    for key, value in mAPs2.items():
        if key in mAPs1:
            mAPs[key] = (mAPs1[key] + mAPs2[key])/2
        else:
            mAPs[key] = mAPs2[key]

    for key, value in trainTime1.items():
        if key in trainTime2:
            trainTime[key] = (trainTime1[key] + trainTime2[key])/2
        else:
            trainTime[key] = trainTime1[key]

    for key, value in trainTime2.items():
        if key in trainTime1:
            trainTime[key] = (trainTime1[key] + trainTime2[key])/2
        else:
            trainTime[key] = trainTime2[key]

    myKeys = list(mAPs.keys())
    myKeys = sorted(myKeys, key=getOrder)

    SmAPs = {j:mAPs[j] for j in myKeys}
    StrainTime = {j:trainTime[j] for j in myKeys}

    allmAP[i] = SmAPs
    allTT[i] = StrainTime

    if(len(list(SmAPs.keys())) > len(keys)):
        keys = list(SmAPs.keys())

    # saveData.saveFile("smap", SmAPs)

    # saveData.plotDataBar(SmAPs,StrainTime, file="Bar ",name=f"dataset/{i} : NC({readyaml.returnClassCount(dataset_home_dir + i + '/data.yaml')})")
    # saveData.plotDataScatter(SmAPs,StrainTime, file="Scatter ",name=f"dataset/{i} : NC({readyaml.returnClassCount(dataset_home_dir + i + '/data.yaml')})")
    # saveData.plotDataCombined(SmAPs,StrainTime, file="Combined ",name=f"dataset/{i} : NC({readyaml.returnClassCount(dataset_home_dir + i + '/data.yaml')})")

NmAP = {key:{k:allmAP[k][key] for k in allmAP if key in allmAP[k]} for key in keys}
NTrainT = {key:{k:allTT[k][key] for k in allTT if key in allTT[k]} for key in keys}

saveData.saveFile("All mAP", NmAP)
saveData.saveFile("All training", NTrainT)
saveData.plotDataScatterByGradient(NmAP,NTrainT,file="layer/Relative ")
saveData.plotDataScatterByGradient(NmAP,NTrainT,relativeToUnfrozen=False,file="layer/")
