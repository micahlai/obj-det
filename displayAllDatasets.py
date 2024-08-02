import os
import readyaml

save_dir = '/lab/micah/obj-det/datasets'


    

def saveFile(name, data):
    file = open(save_dir + f'/{name}.txt','w')
    if(type(data) is list):
        for i in data:
            file.write(str(i) + '\n')
    elif(type(data) is dict):
        for key, value in data.items():
            file.write('%s:%s\n' % (key,value))
    else:
        file.write(str(data))
    file.close()

home_dir = '/lab/micah/obj-det/datasets'

d=home_dir
subdirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]

namelist = []
for i in subdirs:
    namelist.append(i.replace(d+'/',""))

saveFile("dataset names", namelist)

datasetData = {}
for i in namelist:
    datasetData[i] = [readyaml.returnClassCountDefaultDir(i),readyaml.returnTrainingSetCount(i)]

datasetData = {k: v for k, v in sorted(datasetData.items(), key=lambda item: item[1][1])}
saveFile("dataset info", datasetData)