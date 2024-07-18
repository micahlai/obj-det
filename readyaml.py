import yaml
import io

def returnClassCount(path):
    with open(path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded["nc"]
