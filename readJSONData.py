import saveData
import json
import readyaml
import freezeDataLookup

trial1path = "testing runs/7-19 separate branch multidatset/results"
trial2path = "testing runs/7-18 separate branch multidatset/results"
trial3path = "testing runs/7-25 separate branch multidatset/results"

dataset_home_dir = '/lab/micah/obj-det/datasets/'
datasets = ['part number',
            'garage dataset',
            'Resistors',
            'teeth',
            'toyota',
            'hard hat uni',
            'People in painting']

freeze_data = freezeDataLookup.getAllData()
save_dir = "testing runs/7-29 separate branch combined results"
saveData.initialize(save_dir,keepSubs=False,keepModels=False)

def getOrder(key):
    return freeze_data[key]


for i in datasets:

    mAP_path1 = f"{trial1path}/JSON DATA/{i}/FINAL mAP RESULTS.json"
    mAP_path2 = f"{trial2path}/JSON DATA/{i}/FINAL mAP RESULTS.json"
    mAP_path3 = f"{trial3path}/JSON DATA/{i}/FINAL mAP RESULTS.json"
    trainTime_path1 = f"{trial1path}/JSON DATA/{i}/FINAL TRAINING TIME RESULTS.json"
    trainTime_path2 = f"{trial2path}/JSON DATA/{i}/FINAL TRAINING TIME RESULTS.json"
    trainTime_path3 = f"{trial3path}/JSON DATA/{i}/FINAL TRAINING TIME RESULTS.json"
    
    mAPs1 = {}
    trainTime1 = {}
    mAPs2 = {}
    trainTime2 = {}
    mAPs3 = {}
    trainTime3 = {}

    mAPs = {}
    trainTime = {}


    try:
        with open(mAP_path1) as json_file:
            mAPs1 = json.load(json_file)
    except:
        print(i + " doesn't exist for mAP in " + trial1path)

    try:
        with open(trainTime_path1) as json_file:
            trainTime1 = json.load(json_file)
    except:
        print(i + " doesn't exist for training time in " + trial1path)

    try:
        with open(mAP_path2) as json_file:
            mAPs2 = json.load(json_file)
    except:
        print(i + " doesn't exist for mAP in " + trial2path)

    try:
        with open(trainTime_path2) as json_file:
            trainTime2 = json.load(json_file)
    except:
        print(i + " doesn't exist for training time in " + trial3path)
        
    try:
        with open(mAP_path3) as json_file:
            mAPs3 = json.load(json_file)
    except:
        print(i + " doesn't exist for mAP in " + trial3path)

    try:
        with open(trainTime_path3) as json_file:
            trainTime3 = json.load(json_file)
    except:
        print(i + " doesn't exist for training time in " + trial3path)

    for key, value in mAPs1.items():
        if key in mAPs2:
            mAPs[key] = (mAPs1[key] + mAPs2[key])/2
        else:
            mAPs[key] = mAPs1[key]

    myKeys = list(mAPs1.keys())
    myKeys.extend(x for x in list(mAPs2.keys()) if x not in myKeys)
    myKeys.extend(x for x in list(mAPs3.keys()) if x not in myKeys)

    saveData.saveFile("map1", trainTime1)
    saveData.saveFile("map2", trainTime2)
    saveData.saveFile("map3", trainTime3)

    for key in myKeys:
        if key in mAPs1:
            mAPs[key] = mAPs1[key]
            trainTime[key] = trainTime1[key]
        if key in mAPs2:
            mAPs[key] = mAPs2[key]
            trainTime[key] = trainTime2[key]
        if key in mAPs3:
            mAPs[key] = mAPs3[key]
            trainTime[key] = trainTime3[key]


    myKeys = sorted(myKeys, key=getOrder)

    SmAPs = {i:mAPs[i]for i in myKeys}
    StrainTime = {i:trainTime[i]for i in myKeys}

    saveData.plotDataBar(SmAPs,StrainTime, file="Bar ",name=f"{i} : NC({readyaml.returnClassCount(dataset_home_dir + i + '/data.yaml')})")
    saveData.plotDataScatter(SmAPs,StrainTime, file="Scatter ",name=f"{i} : NC({readyaml.returnClassCount(dataset_home_dir + i + '/data.yaml')})")
    saveData.plotDataCombined(SmAPs,StrainTime, file="Combined ",name=f"{i} : NC({readyaml.returnClassCount(dataset_home_dir + i + '/data.yaml')})")