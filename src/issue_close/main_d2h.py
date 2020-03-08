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
data_path = os.path.join(cwd, "..","..", "data","issue_close_time")

directories= ["1 day", "7 days", "14 days", "30 days", "90 days", "180 days", "365 days"]

file_inc = {"camel": 0, "cloudstack": 1, "cocoon":  2, "deeplearning":3,"hadoop":4, "hive":5,"node":6
            , "ofbiz":7, "qpid":8}

def readfile(filename=''):
    df=pd.read_csv(filename)
    dict,labels=df[df.columns[:-1]],df[df.columns[-1]]
    labels=map(lambda x: 1 if x == True else 0, labels)
    return np.array(dict), np.array(labels)

def _test(res=''):

    for x in directories:
        raw_data,labels = readfile("../../data/issue_close_time/"+x+"/"+res+".csv")

        metric="d2h"
        final = {}
        final_auc={}
        e_value = [0.025, 0.05, 0.1, 0.2]
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
        with open('dump/d2h_' + x+"_"+res + '.pickle', 'wb') as handle:
            pickle.dump(final_auc, handle)

if __name__ == '__main__':
    eval(cmd())