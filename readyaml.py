import yaml
import io

def returnClassCount(path):
    with open(path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded["nc"]

def returnClassCountDefaultDir(name, home_dir = '/lab/micah/obj-det/datasets/'):
    yamlpath = home_dir + name + '/data.yaml'
    with open(yamlpath, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded["nc"]
