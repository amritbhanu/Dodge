from __future__ import print_function, division

import pandas as pd
import os
cwd = os.getcwd()
import_path=os.path.abspath(os.path.join(cwd, '..'))
import sys
sys.path.append(import_path)

data_path = os.path.join(cwd, "..","..", "data","smell")

file_dic = {"dataclass":     ["DataClass.csv"],\
        "featureenvy":  ["FeatureEnvy.csv"],\
        "godclass":     ["GodClass.csv"],\
        "longmethod": ["LongMethod.csv"]
            }

file_inc = {"DataClass": 0, "FeatureEnvy": 1, "GodClass":  2, "LongMethod":3}

if __name__ == '__main__':
    for i in file_dic.values():
        df=pd.read_csv("../../data/smell/" + i[0])
        print(i, df.info())