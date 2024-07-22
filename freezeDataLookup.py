import os

freeze_data_path = "freeze set"

def lookupData(key):
    f = f"{key}.txt"
    for root, dirs, files in os.walk(freeze_data_path):
        if f in files:
            return open(os.path.join(root, f), 'r').read().split('\n')
    return []
