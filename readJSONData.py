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

d_datasetIgnore = ['toyota','garage dataset','rock-climbing','chess','go-positions']
d_save_dir = "testing runs/8-8 no branching results"

def combineResults(save_dir=d_save_dir,
                   TP=d_trialpaths,
                   additionalTP = [""],
                   dataset_home_dir=d_dataset_home_dir,
                   datasets=[],ig=d_ig,
                   datasetIgnore=d_datasetIgnore,
                   unfrozK="unfrozen"):
    
    if(datasets == []):
        datasets = open('datasets/dataset-info/dataset names.txt','r').read().split('\n')
        datasets.pop()

    print(datasets)

    

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

        try:
            saveData.plotDataBar(SmAPs,StrainTime, file="dataset/Bar ",name=f"{i} : NC({readyaml.returnClassCount(dataset_home_dir + i + '/data.yaml')})")
            saveData.plotDataScatter(SmAPs,StrainTime, file="dataset/Scatter ",name=f"{i} : NC({readyaml.returnClassCount(dataset_home_dir + i + '/data.yaml')})")
            saveData.plotDataScatter(SmAPs,StrainTime, relative=True,origin=True,file="dataset/Relative Scatter ",name=f"{i} : NC({readyaml.returnClassCount(dataset_home_dir + i + '/data.yaml')})")
            saveData.plotDataScatter(SmAPs,StrainTime, relative=True,origin=False,file="dataset/Relative Scatter N-O ",name=f"{i} : NC({readyaml.returnClassCount(dataset_home_dir + i + '/data.yaml')})")
            saveData.plotDataScatter(SmAPs,StrainTime, relative=False,origin=True,file="dataset/Scatter O ",name=f"{i} : NC({readyaml.returnClassCount(dataset_home_dir + i + '/data.yaml')})")
            saveData.plotDataCombined(SmAPs,StrainTime, file="dataset/Combined ",name=f"{i} : NC({readyaml.returnClassCount(dataset_home_dir + i + '/data.yaml')})")
        except Exception as e:
            print(f"{colors.bcolors.FAIL}{e}{colors.bcolors.ENDC}")

    NmAP = {key:{k:allmAP[k][key] for k in allmAP if key in allmAP[k]} for key in keys}
    NTrainT = {key:{k:allTT[k][key] for k in allTT if key in allTT[k]} for key in keys}

    saveData.saveFile("All mAP", NmAP)
    saveData.saveFile("All training", NTrainT)
    saveData.saveFile("Keys",keys)


    #saveData.plotDataScatterByGradient(NmAP,NTrainT,file="layer/Relative ")
    saveData.createSubfolderwithNameInResults("No Ylim : CV3")
    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="No Ylim : CV3/Class ",unfrozenKey=unfrozK,datasetAttribute="classes", dontIgnore = ig,ignoreDataset=datasetIgnore)
    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="No Ylim : CV3/Ratio ",unfrozenKey=unfrozK,datasetAttribute="ratio", dontIgnore = ig,ignoreDataset=datasetIgnore)
    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="No Ylim : CV3/Size ",unfrozenKey=unfrozK,datasetAttribute="size",dontIgnore=ig,ignoreDataset=datasetIgnore)
    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="No Ylim : CV3/Count ",unfrozenKey=unfrozK,datasetAttribute="count",dontIgnore=ig,ignoreDataset=datasetIgnore)

    saveData.createSubfolderwithNameInResults("Bar Results")
    saveData.plotBarByGradientTotal(NmAP,NTrainT,yVal="fav",unfrozenKey=unfrozK,file="Bar Results/Fav ",ignoreDataset=datasetIgnore)
    saveData.plotBarByGradientTotal(NmAP,NTrainT,yVal="time",unfrozenKey=unfrozK,file="Bar Results/Time ",ignoreDataset=datasetIgnore)
    saveData.plotBarByGradientTotal(NmAP,NTrainT,yVal="mAP",unfrozenKey=unfrozK,file="Bar Results/mAP ",ignoreDataset=datasetIgnore)

    saveData.createSubfolderwithNameInResults("No Branch Line Results")
    saveData.plotLineNoBranches(NmAP,NTrainT,yVal="fav",unfrozenKey=unfrozK,file="No Branch Line Results/Fav ",ignoreDataset=datasetIgnore)
    saveData.plotLineNoBranches(NmAP,NTrainT,yVal="time",unfrozenKey=unfrozK,file="No Branch Line Results/Time ",ignoreDataset=datasetIgnore)
    saveData.plotLineNoBranches(NmAP,NTrainT,yVal="mAP",unfrozenKey=unfrozK,file="No Branch Line Results/mAP ",ignoreDataset=datasetIgnore)

    saveData.createSubfolderwithNameInResults("YLim to CV3")
    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="YLim to CV3/Class ",unfrozenKey=unfrozK,yLimToLSRL=True,datasetAttribute="classes", dontIgnore = ig,ignoreDataset=datasetIgnore)
    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="YLim to CV3/Ratio ",unfrozenKey=unfrozK,yLimToLSRL=True,datasetAttribute="ratio", dontIgnore = ig,ignoreDataset=datasetIgnore)
    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="YLim to CV3/Size ",unfrozenKey=unfrozK,yLimToLSRL=True,datasetAttribute="size",dontIgnore=ig,ignoreDataset=datasetIgnore)
    # saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="YLim to CV3/Count ",unfrozenKey=unfrozK,yLimToLSRL=True,datasetAttribute="count",dontIgnore=ig,ignoreDataset=datasetIgnore)

    saveData.createSubfolderwithNameInResults("Cuttoff Line")
    saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Cuttoff Line/Class ",unfrozenKey=unfrozK,correCutoff=0.5,yLimToLSRL=True,datasetAttribute="classes",ignoreDataset=datasetIgnore)
    saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Cuttoff Line/Ratio ",unfrozenKey=unfrozK,correCutoff=0.5,yLimToLSRL=True,datasetAttribute="ratio",ignoreDataset=datasetIgnore)
    saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Cuttoff Line/Size ",unfrozenKey=unfrozK,correCutoff=0.5,yLimToLSRL=True,datasetAttribute="size",ignoreDataset=datasetIgnore)
    saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Cuttoff Line/Count ",unfrozenKey=unfrozK,correCutoff=0.5,yLimToLSRL=True,datasetAttribute="count",ignoreDataset=datasetIgnore)

    saveData.createSubfolderwithNameInResults("Cuttoff No Dotted Line")
    saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Cuttoff No Dotted Line/Class ",unfrozenKey=unfrozK,correCutoff=0.5,cuttoffDotted=False,yLimToLSRL=True,datasetAttribute="classes",ignoreDataset=datasetIgnore)
    saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Cuttoff No Dotted Line/Ratio ",unfrozenKey=unfrozK,correCutoff=0.5,cuttoffDotted=False,yLimToLSRL=True,datasetAttribute="ratio",ignoreDataset=datasetIgnore)
    saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Cuttoff No Dotted Line/Size ",unfrozenKey=unfrozK,correCutoff=0.5,cuttoffDotted=False,yLimToLSRL=True,datasetAttribute="size",ignoreDataset=datasetIgnore)
    saveData.plotDataLineByGradientTotal(NmAP,NTrainT,file="Cuttoff No Dotted Line/Count ",unfrozenKey=unfrozK,correCutoff=0.5,cuttoffDotted=False,yLimToLSRL=True,datasetAttribute="count",ignoreDataset=datasetIgnore)


if __name__ == "__main__":
    #combineResults(TP=['testing runs/8-6 no branching parallel'],unfrozK='0')
    combineResults()