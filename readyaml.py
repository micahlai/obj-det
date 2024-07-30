import yaml
import os
import io
import statistics

default_path = '/lab/micah/obj-det/datasets/'

def returnClassCount(path):
    with open(path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded["nc"]

def returnClassCountDefaultDir(name, home_dir = default_path):
    yamlpath = home_dir + name + '/data.yaml'
    with open(yamlpath, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded["nc"]

def returnTrainingSetCount(name, home_dir = default_path):
    for root,dirs,files in os.walk(f"{home_dir + name}/train/images"):
        return len(files)
    

def returnHWData(name, home_dir = default_path):
    HWData = []
    home_path = home_dir + name
    for root, dirs, files in os.walk(home_path):
        for dir in dirs:
            if(dir == "labels"):
                for r,d,f in os.walk(os.path.join(root,dir)):
                    for file in f:
                        data=(open(os.path.join(r, file), 'r').read().split('\n'))
                        for line in data:
                            tensor = line.split(' ')
                            try:
                                HWData.append([float(tensor[3]),float(tensor[4])])
                            except:
                                pass
    return HWData
                            
def returnHWRatio(name,  _home_dir = default_path):
    data = returnHWData(name,home_dir=_home_dir)
    ratio = []
    for i in data:
        imgRatio = i[1]/i[0]
        ratio.append(imgRatio)
    return ratio

def returnSize(name,  _home_dir = default_path):
    data = returnHWData(name,home_dir=_home_dir)
    size = []
    for i in data:
        size.append(i[0] * i[1])
    return size

def returnHWRatioAverage(name,  home_dir = default_path):
    data = returnHWRatio(name, _home_dir=home_dir)
    return statistics.mean(data)

def returnHWRatioSTDev(name,  home_dir = default_path):
    data = returnHWRatio(name, _home_dir=home_dir)
    return statistics.stdev(data)

def returnSizeAverage(name,  home_dir = default_path):
    data = returnSize(name, _home_dir=home_dir)
    return statistics.mean(data)

def returnSizeSTDev(name,  home_dir = default_path):
    data = returnSize(name, _home_dir=home_dir)
    return statistics.stdev(data)



