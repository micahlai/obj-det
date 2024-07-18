import saveData
import json

mAP_path = "testing runs/7-17 Full Test/results/JSON DATA/FINAL mAP RESULTS.json"
trainTime_path = "testing runs/7-17 Full Test/results/JSON DATA/FINAL TRAINING TIME RESULTS.json"
mAPs = {}
trainTime = {}

save_dir = "testing runs/separate branch regraph"
saveData.initialize(save_dir)

with open(mAP_path) as json_file:
    mAPs = json.load(json_file)


with open(trainTime_path) as json_file:
    trainTime = json.load(json_file)


saveData.plotDataBar(mAPs,trainTime)