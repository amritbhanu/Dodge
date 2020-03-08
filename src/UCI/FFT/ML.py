from __future__ import print_function, division

import sys

#sys.dont_write_bytecode = True
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix
import numpy as np
import pandas as pd
from FFT import FFT
import math
from helpers import get_score

metrics=['Dist2Heaven']

metrics_dic={'accuracy':-2,'recall':-6,'precision':-7,'false_alarm':-4}
PRE, REC, SPEC, FPR, NPV, ACC, F1 = 7, 6, 5, 4, 3, 2, 1

def DT(k,train_data,train_labels,test_data,test_labels, metric):

    model = DecisionTreeClassifier(**k)
    model.fit(train_data, train_labels)
    prediction=model.predict(test_data)
    dic = {}
    print(train_data,train_labels)
    for i in metrics:
        dic[i] = round(evaluation(i, prediction, test_labels),3)

    return dic[metric], [dic, model.feature_importances_]

def RF(k,train_data,train_labels,test_data,test_labels, metric):
    model = RandomForestClassifier(**k)
    model.fit(train_data, train_labels)
    prediction = model.predict(test_data)
    dic = {}
    for i in metrics:
        dic[i] = round(evaluation(i, prediction, test_labels),3)
    return dic[metric], [dic, model.feature_importances_]


def SVM(k,train_data,train_labels,test_data,test_labels, metric):
    from sklearn.preprocessing import MinMaxScaler
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(train_data)
    train_data = scaling.transform(train_data)
    test_data = scaling.transform(test_data)
    model = SVC(cache_size=20000,**k)
    model.fit(train_data, train_labels)
    #print(model.coef_)
    prediction = model.predict(test_data)
    dic = {}
    for i in metrics:
        dic[i] = round(evaluation(i, prediction, test_labels),3)
    return dic[metric], [dic, []]


def FFT1(k,train_data,train_labels,test_data,test_labels, metric):
    dic={}
    dic1={}
    for i in metrics:
        fft = FFT(max_level=5)
        fft.criteria=i
        #fft.print_enabled=True
        train_labels=np.reshape(train_labels,(-1,1))
        test_labels = np.reshape(test_labels, (-1, 1))

        training=np.hstack((train_data, train_labels))
        testing = np.hstack((test_data, test_labels))
        training_df = pd.DataFrame(training)
        testing_df = pd.DataFrame(testing)
        training_df.rename(columns={training_df.columns[-1]: "bug"},inplace=True)
        testing_df.rename(columns={testing_df.columns[-1]: "bug"},inplace=True)

        fft.target = "bug"
        fft.train, fft.test = training_df, testing_df
        fft.build_trees()  # build and get performance on TEST data
        t_id = fft.find_best_tree()  # find the best tree on TRAIN data
        fft.eval_tree(t_id)  # eval all the trees on TEST data

        description=fft.print_tree(t_id)
        if i!='Dist2Heaven':
            dic[i]=fft.performance_on_test[t_id][metrics_dic[i]]
        else:
            dic["Dist2Heaven"]=get_score("Dist2Heaven", fft.performance_on_test[t_id][:4])
        dic1[i] = description
    return dic[metric], [dic, dic1]

def get_performance(prediction, test_labels):
    tn, fp, fn, tp = confusion_matrix(test_labels,prediction).ravel()
    pre = 1.0 * tp / (tp + fp) if (tp + fp) != 0 else 0
    rec = 1.0 * tp / (tp + fn) if (tp + fn) != 0 else 0
    spec = 1.0 * tn / (tn + fp) if (tn + fp) != 0 else 0
    fpr = 1 - spec
    npv = 1.0 * tn / (tn + fn) if (tn + fn) != 0 else 0
    acc = 1.0 * (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) != 0 else 0
    return [round(x, 3) for x in [pre, rec, spec, fpr, npv, acc, f1]]

def evaluation(measure, prediction, test_labels, class_target=1):
    tn, fp, fn, tp = confusion_matrix(test_labels, prediction).ravel()
    pre, rec, spec, fpr, npv, acc, f1 = get_performance(test_labels, prediction)
    all_metrics = [tp, fp, tn, fn, pre, rec, spec, fpr, npv, acc, f1]
    if measure == "accuracy":
        score = -all_metrics[-ACC]
    elif measure == "recall":
        score = -all_metrics[-REC]
    elif measure == "precision":
        score = -all_metrics[-PRE]
    elif measure == "false_alarm":
        score = -all_metrics[-FPR]
    elif measure == "f1":
        score = -all_metrics[-F1]
    elif measure == "Dist2Heaven":
        score = all_metrics[-FPR] ** 2 + (1 - all_metrics[-REC]) ** 2
        score = math.sqrt(score) / math.sqrt(2)
    return score