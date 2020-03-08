import os
import sys
from glob import glob
from pdb import set_trace

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)


def get_all_datasets():
    dir = os.path.abspath(os.path.join(root, "datasets"))
    datasets = dict()
    for datapath in os.listdir(dir):
        formatted_path = os.path.join(dir, datapath)
        if os.path.isdir(formatted_path):
            datasets.update({datapath: dict()})
            files = glob(os.path.join(formatted_path, "*.csv"))
            for f in files:
                fname = f.split('/')[-1].split("-")[0]
                datasets[datapath].update({fname: f})

    return datasets


if __name__=="__main__":
    get_all_datasets()
