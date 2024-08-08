import os
import readyaml
import pandas as pd

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
subdirs.remove(home_dir + '/dataset-info')

namelist = []
for i in subdirs:
    namelist.append(i.replace(d+'/',""))

saveFile("dataset-info/dataset names", namelist)


datasetData = {}
for i in namelist:
    datasetData[i] = [readyaml.returnClassCountDefaultDir(i),readyaml.returnTrainingSetCount(i)]

datasetData = {k: v for k, v in sorted(datasetData.items(), key=lambda item: item[1][1])}
saveFile("dataset-info/dataset info", datasetData)

datasetLinks = {}
for i in namelist:
    datasetLinks[i] = [readyaml.returnProjectName(i),readyaml.returnURL(i)]

saveFile("dataset-info/dataset urls", datasetLinks)


totalDatasetData = {}
for i in namelist:
    totalDatasetData[i] = [readyaml.returnProjectName(i),readyaml.returnClassCountDefaultDir(i),readyaml.returnTrainingSetCount(i),readyaml.returnURL(i)]

newData = {}
for key,val in totalDatasetData.items():
    newData[key]=tuple(val)

df = pd.DataFrame.from_dict(newData)
dft = df.transpose()
dfc = dft.set_axis(['project name','class count','training set count','roboflow url'],axis=1)
dfc.to_csv('datasets/dataset-info/all dataset info.csv')
