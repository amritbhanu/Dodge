from __future__ import print_function


import pandas as pd
import os
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
import arff
import math
import pickle

cwd=os.getcwd()
data_path=os.path.join(cwd,"..","..","data", "UCI")
file_inc = {"adult": 0, "cancer": 1, "covtype":  2, "diabetic":3, "optdigits":4, "pendigits":5
            , "satellite":6, "shuttle":7, "waveform":8,"annealing":9,"audit":10,"autism":11,
            "bank":12,"bankrupt":13,"biodegrade":14,"blood-transfusion":15,"car":16,
            "cardiotocography":17,"cervical-cancer":18, "climate-sim":19,"contraceptive":20,
            "credit-approval":21,"credit-default":22,"crowdsource":23,"drug-consumption":24,
            "electric-stable":25,"gamma":26,"hand":27,"hepmass":28,"htru2":29,"image":30,
            "kddcup":31,"liver":32,"mushroom":33,"phishing":34,"sensorless-drive":35,"shop-intention":36}

final = {"adult": {"pd":[0.605,0.614,0.597,0.602],"pf":[0.075,0.08,0.071,0.074]},
         "cancer": {"pd":[0.934,0.934,0.934,0.939],"pf":[0.014,0.017,0.025,0.008]},
         "covtype":  {"pd":[1.0,0.99,1,0,0.99],"pf":[0.0,0.0,0.0,0.0]},
         "diabetic":{"pd":[0.638,0.635,0.655,0.664],"pf":[0.302,0.274,0.313,0.302]},
         "optdigits":{"pd":[0.99,0.993,0.986,0.988],"pf":[0.014,0.011,0.07,0.014]},
         "pendigits":{"pd":[0.999,1.0,1.0,1.000],"pf":[0.001,0.0,0.001,0.001]},
         "satellite":{"pd":[0.989,0.987,0.987,0.984],"pf":[0.006,0.003,0.003,0.003]},
         "shuttle":{"pd":[0.999,0.999,0.999,0.999],"pf":[0.0,0.0,0.0,0.0]},
         "waveform":{"pd":[0.925,0.93,0.925,0.923],"pf":[0.097,0.104,0.104,0.098]}}

# pd=recall
# pf=fpr
# score = all_metrics[-FPR] ** 2 + (1 - all_metrics[-REC]) ** 2
#         score = math.sqrt(score) / math.sqrt(2)
np.random.seed(1)
random.seed(1)

def spit_datasets(filename=''):
    path1="../../data/UCI/" + filename + ".csv"
    df=pd.read_csv(path1,header=None)
    df[df.columns[-1]]=df[df.columns[-1]].apply(lambda x: True if x==1 else False)
    count=1
    for i in range(4):
        df = df.sample(frac=1).reset_index(drop=True)
        #dict, labels = df[df.columns[:-1]], df[df.columns[-1]]
        #skf = StratifiedKFold(n_splits=5, shuffle=False)
        arff.dump(data_path + "/CHIRP/Train/" + filename + str(count) + ".arff"
                  , df.values, relation='name', names=df.columns)
        # for train_index, test_index in skf.split(dict, labels):
        #     X_train, X_test = dict[dict.index.isin(train_index.tolist())], dict[dict.index.isin(test_index.tolist())]
        #     y_train, y_test = labels[labels.index.isin(train_index.tolist())], labels[labels.index.isin(test_index.tolist())]
        #     X_train["class"]=y_train
        #     X_test["class"]=y_test
        #     df = pd.concat([X_train, X_test], ignore_index=True)

            # arff.dump(data_path+"/CHIRP/Train/"+filename+str(count)+".arff"
            #           , df.values, relation='name', names=df.columns)
            # arff.dump(data_path + "/CHIRP/Test/" + filename + str(count) + ".arff"
            #           , X_test.values, relation='name', names=X_test.columns)

            #path2=data_path+"/CHIRP/Train/"+filename+str(count)+".csv"
            #path3=data_path+"/CHIRP/Test/"+filename+str(count)+".csv"
            #X_train.to_csv(path2,index=False)
            #X_test.to_csv(path3, index=False)
        count+=1


def dump_files(f=''):
    with open("../../dump/UCI/d2h_" + f+".pickle", 'rb') as handle:
        final = pickle.load(handle)
    return final

def dump_files1(f=''):
    with open("../../dump/UCI/FFT/fft_d2h" + f+".pickle", 'rb') as handle:
        final = pickle.load(handle)
    final["FFT-Dist2Heaven"]=final["FFT1"]
    del final['FFT1']
    return final

if __name__ == '__main__':
    # for i in file_inc.keys():
    #     spit_datasets(i)

    d2h={}
    for i in final.keys():
        d2h[i]=[]
        pd=final[i]["pd"]
        pf=final[i]["pf"]
        for x,y in zip(pd,pf):
            score=y**2 + (1-x)**2
            score=math.sqrt(score)/math.sqrt(2)
            d2h[i].append(round(score,3))
    for i in d2h:
        d2h[i]=sorted(d2h[i])

    temp_fi = {}
    temp_fft={}
    print(d2h)
    for j in file_inc.keys():
        dic = dump_files(j)
        temp_fft[j]=dump_files1(j)
        temp_fi[j] = sorted(dic["counter_full"][0.2][29])


    with open('../../dump/UCI/dodge.pickle', 'wb') as handle:
        pickle.dump(temp_fi, handle)

    with open('../../dump/UCI/fft.pickle', 'wb') as handle:
        pickle.dump(temp_fft, handle)


