import os
import sys
from glob import glob
from pdb import set_trace
import pandas as pd

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)


def make_new_groups():
    dir = os.path.abspath(os.path.join(root, "datasets"))
    inner_dirs = [1, 7, 14, 30, 90, 180, 365]
    for datapath in os.listdir(dir):
        formatted_path = os.path.join(dir, datapath)
        if os.path.isdir(formatted_path):
            try:
                file = glob(os.path.join(formatted_path, "*.csv"))[0]
                fname = formatted_path.split("/")[-1]
                dframe = pd.read_csv(file)
                klass = dframe['timeOpen']
                for val in inner_dirs:
                    new_klass = [a == val for a in klass]
                    dframe['timeOpen'] = new_klass
                    fpath = os.path.join(os.path.abspath(str(val)), fname + ".csv")
                    try:
                        dframe.to_csv(fpath, index=False)
                    except IOError:
                        os.makedirs("/".join(fpath.split("/")[:-1]))
                        dframe.to_csv(fpath, index=False)
                    # set_trace()
                    dframe['timeOpen'] = klass
            except IndexError:
                pass

    set_trace()


if __name__ == "__main__":
    make_new_groups()
