import saveData
import json
import readyaml
import freezeDataLookup
import colors

d_trialpaths = ["testing runs/7-19 separate branch multidatset",
              "testing runs/7-18 separate branch multidatset",
              "testing runs/7-25 separate branch multidatset",
              "testing runs/7-30 separate branch multidatset",
                "testing runs/8-1 full parallel test take 2"]

d_dataset_home_dir = '/lab/micah/obj-det/datasets/'
d_datasets = ['part number',
            'garage dataset',
            'Resistors',
            'teeth',
            'toyota',
            'hard hat uni',
            'People in painting',
            'grocery store',
            'football',
            'hazard signs',
            'construction',
            'rock paper scissors',
            'carbike',
            'car-dent',
            'cows',
            'rock-climbing',
            'clouds',
            'mri',
            'frc',
            'tesla',
            'thermal',
            'aerial-cars',
            'parking-lot',
            'playing cards',
            'bodywash']

d_ig = ['all cv2 after 10 + cv3',
'first 10',
'all cv1 after 10',
'first 10 + 23 cv3',
'all cv1 after 10 + cv3',
'all m.2 after 10 + cv3',
'all m.0 after 10 + cv3',
'all m.1 after 10 + cv3',
'all m + cv3',
'all m after 10',
'all m after 10 + cv3']

d_datasetIgnore = ['toyota','garage dataset','rock-climbing']
d_save_dir = "testing runs/8-5 separate branch combined results"

def combineResults(save_dir=d_save_dir,
                   TP=d_trialpaths,
                   additionalTP = [""],
                   dataset_home_dir=d_dataset_home_dir,
                   datasets=d_datasets,ig=d_ig,
                   datasetIgnore=d_datasetIgnore):
    
    freeze_data = freezeDataLookup.getAllData()
    saveData.initialize(save_dir,keepSubs=False,keepModels=False)
    saveData.createSubfolderwithNameInResults('dataset')
    saveData.createSubfolderwithNameInResults('layer')

    trialpaths = TP
    trialpaths.extend(x for x in additionalTP if x not in trialpaths)

    allmAP = {}
    allTT = {}

    keys = []

    printJSONParseErrors = True

    for i in datasets:

        mAP_paths = [f"{x}/results/JSON DATA/{i}/FINAL mAP RESULTS.json" for x in trialpaths]
        trainTime_paths = [f"{x}/results/JSON DATA/{i}/FINAL TRAINING TIME RESULTS.json" for x in trialpaths]

        mAP_temp = []
        trainTime_temp = []

        mAPs = {}
        trainTime = {}

        for j in mAP_paths:
            try:
                with open(j) as json_file:
                    mAP_temp.append(json.load(json_file))
            except:
                if(printJSONParseErrors):
                    print(colors.bcolors.WARNING + i + " doesn't exist for mAP in " + j + colors.bcolors.ENDC)

        for j in trainTime_paths:
            try:
                with open(j) as json_file:
                    trainTime_temp.append(json.load(json_file))
            except:
                if(printJSONParseErrors):
                    print(colors.bcolors.WARNING + i + " doesn't exist for training time in " + j + colors.bcolors.ENDC)

        myKeys = []
        for j in mAP_temp:
            myKeys.extend(x for x in list(j.keys()) if x not in myKeys)

        for key in myKeys:
            for j in range(len(mAP_temp)):
                if key in mAP_temp[j]:
                    mAPs[key] = mAP_temp[j][key]
                    trainTime[key] = trainTime_temp[j][key]


        myKeys = sorted(myKeys, key=freeze_data.get)

        SmAPs = {i:mAPs[i]for i in myKeys}
        StrainTime = {i:trainTime[i]for i in myKeys}

        allmAP[i] = SmAPs
        allTT[i] = StrainTime

        if(len(list(SmAPs.keys())) > len(keys)):
            keys = list(SmAPs.keys())

        # try:
        #     saveData.plotDataBar(SmAPs,StrainTime, file="dataset/Bar ",name=f"{i} : NC({readyaml.returnClassCount(dataset_home_dir + i + '/data.yaml')})")
        #     saveData.plotDataScatter(SmAPs,StrainTime, file="dataset/Scatter ",name=f"{i} : NC({readyaml.returnClassCount(dataset_home_dir + i + '/data.yaml')})")
        #     saveData.plotDataScatter(SmAPs,StrainTime, relative=True,origin=True,file="dataset/Relative Scatter ",name=f"{i} : NC({readyaml.returnClassCount(dataset_home_dir + i + '/data.yaml')})")
        #     saveData.plotDataScatter(SmAPs,StrainTime, relative=True,origin=False,file="dataset/Relative Scatter N-O ",name=f"{i} : NC({readyaml.returnClassCount(dataset_home_dir + i + '/data.yaml')})")
        #     saveData.plotDataScatter(SmAPs,StrainTime, relative=False,origin=True,file="dataset/Scatter O ",name=f"{i} : NC({readyaml.returnClassCount(dataset_home_dir + i + '/data.yaml')})")
        #     saveData.plotDataCombined(SmAPs,StrainTime, file="dataset/Combined ",name=f"{i} : NC({readyaml.returnClassCount(dataset_home_dir + i + '/data.yaml')})")
        # except Exception as e:
        #     print(f"{colors.bcolors.FAIL}{e}{colors.bcolors.ENDC}")

    NmAP = {key:{k:allmAP[k][key] for k in allmAP if key in allmAP[k]} for key in keys}
    NTrainT = {key:{k:allTT[k][key] for k in allTT if key in allTT[k]} for key in keys}

    saveData.saveFile("All mAP", NmAP)
    saveData.saveFile("All training", NTrainT)
    saveData.saveFile("Keys",keys)



    #saveData.plotDataScatterByGradient(NmAP,NTrainT,file="layer/Relative ")
    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Relative Class ",datasetAttribute="classes", dontIgnore = ig,ignoreDataset=datasetIgnore)
    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Relative Ratio ",datasetAttribute="ratio", dontIgnore = ig,ignoreDataset=datasetIgnore)
    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Relative Size ",datasetAttribute="size",dontIgnore=ig,ignoreDataset=datasetIgnore)
    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Relative Count ",datasetAttribute="count",dontIgnore=ig,ignoreDataset=datasetIgnore)

    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Relative LSRL Class ",yLimToLSRL=True,datasetAttribute="classes", dontIgnore = ig,ignoreDataset=datasetIgnore)
    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Relative LSRL Ratio ",yLimToLSRL=True,datasetAttribute="ratio", dontIgnore = ig,ignoreDataset=datasetIgnore)
    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Relative LSRL Size ",yLimToLSRL=True,datasetAttribute="size",dontIgnore=ig,ignoreDataset=datasetIgnore)
    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Relative LSRL Count ",yLimToLSRL=True,datasetAttribute="count",dontIgnore=ig,ignoreDataset=datasetIgnore)

    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Relative Cutoff Class ",correCutoff=0.5,yLimToLSRL=True,datasetAttribute="classes",ignoreDataset=datasetIgnore)
    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Relative Cutoff Ratio ",correCutoff=0.5,yLimToLSRL=True,datasetAttribute="ratio",ignoreDataset=datasetIgnore)
    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Relative Cutoff Size ",correCutoff=0.5,yLimToLSRL=True,datasetAttribute="size",ignoreDataset=datasetIgnore)
    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Relative Cutoff Count ",correCutoff=0.5,yLimToLSRL=True,datasetAttribute="count",ignoreDataset=datasetIgnore)

    saveData.plotBarByGradientTotal(NmAP,NTrainT,yVal="fav",file="Overall Fav ")
    saveData.plotBarByGradientTotal(NmAP,NTrainT,yVal="time",file="Overall Time ")
    saveData.plotBarByGradientTotal(NmAP,NTrainT,yVal="mAP",file="Overall mAP ")

if __name__ == "__main__":
    combineResults()