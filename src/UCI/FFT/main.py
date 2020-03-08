from __future__ import print_function, division

import sys
from demo import cmd

#sys.dont_write_bytecode = True
from collections import OrderedDict
import os
from random import seed
import numpy as np
from ML import DT, SVM, RF, FFT1
from sklearn.model_selection import StratifiedKFold
import pickle
import pandas as pd

ROOT = os.getcwd()
MLS = [DT, RF, SVM,FFT1]

MLS_para_dic = [OrderedDict([("min_samples_split", 2), ("min_impurity_decrease", 0.0), ("max_depth", None),
                             ("min_samples_leaf", 1)]),
                OrderedDict([("min_samples_split", 2), ("max_leaf_nodes", None), ("min_samples_leaf", 1),
                            ("min_impurity_decrease", 0.0), ("n_estimators", 10)]),
                OrderedDict([("C", 1.0), ("kernel", 'linear'),
                             ("degree", 3)]), OrderedDict()]

#metrics = ['accuracy', 'recall', 'precision', 'false_alarm','Dist2Heaven']

# file_inc = {"adult": 0, "cancer": 1, "covtype":  2, "diabetic":3, "optdigits":4, "pendigits":5
#             , "satellite":6, "shuttle":7, "waveform":8,"annealing":9,"audit":10,"autism":11,
#             "bank":12,"bankrupt":13,"biodegrade":14,"blood-transfusion":15,"car":16,
#             "cardiotocography":17,"cervical-cancer":18, "climate-sim":19,"contraceptive":20,
#             "credit-approval":21,"credit-default":22,"crowdsource":23,"drug-consumption":24,
#             "electric-stable":25,"gamma":26,"hand":27,"hepmass":28,"htru2":29,"image":30,
#             "kddcup":31,"liver":32,"mushroom":33,"phishing":34,"sensorless-drive":35,"shop-intention":36}

file_inc = {"annealing":9,"audit":10,"autism":11,
            "bank":12,"bankrupt":13,"biodegrade":14,"blood-transfusion":15,"car":16,
            "cardiotocography":17,"cervical-cancer":18, "climate-sim":19,"contraceptive":20,
            "credit-approval":21,"credit-default":22,"crowdsource":23,"drug-consumption":24,
            "electric-stable":25,"gamma":26,"hand":27,"hepmass":28,"htru2":29,"image":30,
            "kddcup":31,"liver":32,"mushroom":33,"phishing":34,"sensorless-drive":35,"shop-intention":36}

def readfile(filename=''):
    df=pd.read_csv(filename,header=None)
    return df


def _test(res=''):
    seed(1)
    np.random.seed(1)

    df = readfile("../../../data/UCI/"+res+".csv")
    temp = {}

    for i in range(4):
        df = df.sample(frac=1).reset_index(drop=True)
        dict, labels = df[df.columns[:-1]], df[df.columns[-1]]
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        for train_index, test_index in skf.split(dict, labels):
            train_data, test_data = dict[dict.index.isin(train_index.tolist())], dict[dict.index.isin(test_index.tolist())]
            train_labels, test_labels = labels[labels.index.isin(train_index.tolist())], labels[labels.index.isin(test_index.tolist())]
            for j, le in enumerate(MLS):
                if le.__name__ not in temp:
                    temp[le.__name__] = []
                _,val = MLS[j](MLS_para_dic[j], train_data.values, train_labels.values, test_data.values, test_labels.values, 'Dist2Heaven')
                temp[le.__name__].append(val[0]['Dist2Heaven'])

    with open('dump/fft_d2h' + res + '.pickle', 'wb') as handle:
        pickle.dump(temp, handle)

if __name__ == '__main__':
    eval(cmd())