from __future__ import print_function, division

import os
cwd = os.getcwd()
import_path=os.path.abspath(os.path.join(cwd, '..'))
import sys
sys.path.append(import_path)

from helper.transformation import *
from random import seed
from helper.utilities import _randchoice, unpack
from helper.ML import *
from itertools import product
from sklearn.metrics import auc
#from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import StratifiedShuffleSplit
from helper.demos import *
import time
import pickle
from collections import OrderedDict
from operator import itemgetter


metrics=["d2h"]

# file_inc = {"adult": 0, "cancer": 1, "covtype":  2, "diabetic":3, "optdigits":4, "pendigits":5
#             , "satellite":6, "shuttle":7, "waveform":8,"annealing":9,"audit":10,"autism":11,
#             "bank":12,"bankrupt":13,"biodegrade":14,"blood-transfusion":15,"car":16,
#             "cardiotocography":17,"cervical-cancer":18, "climate-sim":19,"contraceptive":20,
#             "credit-approval":21,"credit-default":22,"crowdsource":23,"drug-consumption":24,
#             "electric-stable":25,"gamma":26,"hand":27,"hepmass":28,"htru2":29,"image":30,
#             "kddcup":31,"liver":32,"mushroom":33,"phishing":34,"sensorless-drive":35,"shop-intention":36}

# image, liver, phishing

file_inc = {"annealing":9,"audit":10,"autism":11,
            "bank":12,"bankrupt":13,"biodegrade":14,"blood-transfusion":15,"car":16,
            "cardiotocography":17,"cervical-cancer":18, "climate-sim":19,"contraceptive":20,
            "credit-approval":21,"credit-default":22,"crowdsource":23,"drug-consumption":24,
            "electric-stable":25,"gamma":26,"hand":27,"hepmass":28,"htru2":29,"image":30,
            "kddcup":31,"liver":32,"mushroom":33,"phishing":34,"sensorless-drive":35,"shop-intention":36}

def readfile(filename=''):
    df=pd.read_csv(filename,header=None)
    neg=df[df[df.columns[-1]]==0]
    pos=df[df[df.columns[-1]]==1]
    cut_pos=int(pos[0].count()* 0.8)
    cut_neg=int(neg[0].count()* 0.8)
    pos_1, pos_2=pos.iloc[:cut_pos, :], pos.iloc[cut_pos:, :]
    neg_1, neg_2 = neg.iloc[:cut_neg, :], neg.iloc[cut_neg:, :]
    df = pd.concat([pos_1, neg_1, pos_2, neg_2], ignore_index=True)
    dict,labels=df[df.columns[:-1]],df[df.columns[-1]]
    return np.array(dict), np.array(labels)

def _test(res=''):


    raw_data,labels = readfile("../../data/UCI/"+res+".csv")

    metric="d2h"
    final = {}
    final_auc={}
    e_value = [0.05, 0.1, 0.2]
    start_time=time.time()
    dic={}
    dic_func={}

    for mn in range(500+file_inc[res]*10,521+file_inc[res]*10):
        for e in e_value:
            np.random.seed(mn)
            seed(mn)
            preprocess = [standard_scaler, minmax_scaler, maxabs_scaler, [robust_scaler] * 20, kernel_centerer,
                          [quantile_transform] * 200
                , normalizer, [binarize] * 100]  # ,[polynomial]*5
            MLs = [NB, [KNN] * 20, [RF] * 50, [DT] * 30, [LR] * 50]  # [SVM]*100,
            preprocess_list = unpack(preprocess)
            MLs_list = unpack(MLs)
            combine = [[r[0], r[1]] for r in product(preprocess_list, MLs_list)]

            if e not in final_auc.keys():
                final_auc[e]=[]
                dic[e] = {}


            func_str_dic = {}
            func_str_counter_dic = {}
            lis_value = []
            dic_auc={}
            for i in combine:
                scaler, tmp1 = i[0]()
                model, tmp2 = i[1]()
                string1 = tmp1 + "|" + tmp2
                func_str_dic[string1] = [scaler, model]
                func_str_counter_dic[string1] = 0

            counter=0
            while counter!=200:
                if counter not in dic_func.keys():
                    dic_func[counter]=[]

                keys = [k for k, v in func_str_counter_dic.items() if v == 0]
                key = _randchoice(keys)

                cut_off=int(len(raw_data)*0.8)
                scaler, model = func_str_dic[key]
                df1 = transform(raw_data, scaler)
                df1.loc[:, "bug"] = labels

                train_data, test_data = df1.iloc[:cut_off,:], df1.iloc[cut_off:,:]
                measurement = run_model(train_data, test_data, model, metric,training=-1)

                if all(abs(t - measurement) > e for t in lis_value):
                    lis_value.append(measurement)
                    func_str_counter_dic[key] += 1
                else:
                    func_str_counter_dic[key] += -1

                if counter not in dic[e].keys():
                    dic[e][counter] = []
                    dic_func[counter]=[]
                if e == 0.05:
                    dic_func[counter].append(key)
                dic[e][counter].append(min(lis_value))
                dic_auc[counter]=min(lis_value)

                counter+=1

            dic1 = OrderedDict(sorted(dic_auc.items(), key=itemgetter(0))).values()
            area_under_curve=round(auc(list(range(len(dic1))), dic1), 3)
            final[e]=dic_auc
            final_auc[e].append(area_under_curve)
    total_run=time.time()-start_time
    final_auc["temp"]=final
    final_auc["time"] = total_run
    final_auc["counter_full"]=dic
    final_auc["settings"]=dic_func
    print(final_auc)
    with open('dump/d2h_' + res + '.pickle', 'wb') as handle:
        pickle.dump(final_auc, handle)

if __name__ == '__main__':
    eval(cmd())