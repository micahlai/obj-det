import readyaml

dataset_home_dir = '/lab/micah/obj-det/datasets/'
datasets = ['part number',
            'garage dataset',
            'Resistors',
            'teeth',
            'toyota',
            'car types']

for i in datasets:
    yamlpath = dataset_home_dir + i + '/data.yaml'
    datasetClassCount = readyaml.returnClassCount(yamlpath)
    print(f"{i} : {datasetClassCount}")