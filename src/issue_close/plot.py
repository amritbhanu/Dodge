from __future__ import print_function, division


import matplotlib.pyplot as plt
import os
import pickle
import plotly
import plotly.plotly as py
#from helper.stats import rdivDemo
import numpy as np
from collections import OrderedDict
from operator import itemgetter

e_value=[0.2,0.1, 0.05]
files=["camel.csv", "cloudstack.csv", "cocoon.csv", "deeplearning.csv","hive.csv" ,"node.csv", "ofbiz.csv", "qpid.csv", "hadoop.csv"]
folders = ["1 day"] + map(lambda x: str(x) + " days", [7, 14, 30, 90, 180, 365])

ROOT=os.getcwd()

def dump_files(folder='',file=''):
    # for _, _, files in os.walk(ROOT + "/../dump/defect/"):
    #     for file in files:
    #         if f in file:
    f=file.split(".")[0]
    with open("../../dump/issue_close/d2h_" +folder+"_"+ f+".pickle", 'rb') as handle:
        final = pickle.load(handle)
    return final


def draw(dic,f):
    font = {'size': 70}
    plt.rc('font', **font)
    paras = {'lines.linewidth': 20, 'legend.fontsize': 70, 'axes.labelsize': 80, 'legend.frameon': True,
                  'figure.autolayout': True,'axes.linewidth':8}
    plt.rcParams.update(paras)
    colors = ['red', 'green', 'blue', 'orange']
    markers=["o","*","v","D"]
    fig = plt.figure(figsize=(80, 60))
    for x,i in enumerate(e_value):
        li=dic[i].values()
        li=[y+(0.01*(x+1)) for y in li]
        ## li = [y - (0.01 * (x + 1)) for y in li]
        plt.plot(li,color=colors[x],label=str(i)+" epsi")

    plt.ylabel("Max Popt20 Score")
    plt.ylim(0,1.2)

    plt.xlabel("No. of iterations")
    plt.legend(bbox_to_anchor=(0.7, 0.5), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../../results/popt20/"+f+ ".png")
    plt.close(fig)

def draw_iqr(dic,f):
    font = {'size': 90}
    plt.rc('font', **font)
    paras = {'lines.linewidth': 10, 'legend.fontsize': 90, 'axes.labelsize': 90, 'legend.frameon': True,
                  'figure.autolayout': True,'axes.linewidth':8}
    plt.rcParams.update(paras)
    colors = ['red', 'green', 'blue', 'orange']
    markers=["o","*","v","D"]
    fig = plt.figure(figsize=(50, 40))
    for x,i in enumerate(e_value):
        li = dic[i].values()
        # temp1 = [z for y,z in enumerate(li) if y==100]
        # print(max(temp1[0]))
        med = [round(np.median(y),3) for y in li]
        print(i, min(li[29]))
        iqr = [round((np.percentile(y,75)-np.percentile(y,25)), 3) for y in li]

        #print(i, max(med[:100]))
        plt.plot(med,color=colors[x],label="median "+str(i)+" epsi")
        plt.plot(iqr, color=colors[x],linestyle='-.', label="iqr "+str(i) + " epsi")

    plt.ylabel("Min D2h Score")
    plt.ylim(0,1)
    plt.xlabel("No. of iterations")
    plt.title(f + ' Dataset')
    plt.legend(bbox_to_anchor=(0.8, 0.9), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../../results/d2h/"+f+ "_iqr.png")
    plt.close(fig)

def draw_boxplot(dic,f):
    font = {'size': 70}
    plt.rc('font', **font)
    paras = {'lines.linewidth': 70, 'legend.fontsize': 70, 'axes.labelsize': 80, 'legend.frameon': True,
             'figure.autolayout': True, 'axes.linewidth': 8}
    plt.rcParams.update(paras)

    boxprops = dict(linewidth=9, color='black')
    colors = ['red', 'green', 'blue', 'purple']
    whiskerprops = dict(linewidth=5)
    medianprops = dict(linewidth=8, color='firebrick')

    dic1 = OrderedDict(sorted(dic.items(), key=itemgetter(0)))

    fig1, ax1 = plt.subplots(figsize=(80, 60))
    bplot = ax1.boxplot(dic1.values(), showmeans=False, showfliers=False, medianprops=medianprops, capprops=whiskerprops,
                       flierprops=whiskerprops, boxprops=boxprops, whiskerprops=whiskerprops)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set(color=color)
    ax1.set_xticklabels(dic1.keys())
    #ax1.set_ylim([0, 1])
    ax1.set_xlabel("Epsilon Values")
    ax1.set_ylabel("AUC of Popt20 (20 repeats)", labelpad=30)
    plt.savefig("../../results/popt20/" + f + "_auc.png")
    plt.close(fig1)



def para_samples(perf, settings, file):
    dic = {'RandomForestClassifier': {'n_estimators': {75: 0, 100: 0, 125: 0, 150: 0},
                                      'min_samples_split': {0.25: 0, 0.5: 0, 0.75: 0, 1: 0}},
           'DecisionTreeClassifier': {0.25: 0, 0.5: 0, 0.75: 0, 1: 0},
           'KNeighborsClassifier': {6: 0, 12: 0, 18: 0, 24: 0},
           'LogisticRegression': {120: 0, 240: 0, 360: 0, 480: 0},
           'MaxAbsScaler': 0, 'MinMaxScaler': 0, 'StandardScaler': 0, 'Normalizer': 0,
           'KernelCenterer': 0,
           'Binarizer': {25: 0, 50: 0, 75: 0, 100: 0},
           'QuantileTransformer': {250: 0, 500: 0, 750: 0, 1000: 0},
           'RobustScaler': {1: 0, 2: 0, 3: 0, 4: 0}
           }
    for x in range(20):
            for j in perf.keys()[:100]:
                preprocess, learner = settings[j][x].split("|")
                temp=preprocess.split("_")
                temp1=learner.split("_")
                if temp1[-1]=="RandomForestClassifier":
                    if int(temp1[0]) <=75: dic[temp1[-1]]['n_estimators'][75]+=1
                    if 75 < int(temp1[0]) <= 100: dic[temp1[-1]]['n_estimators'][100] += 1
                    if 100 < int(temp1[0]) <= 125: dic[temp1[-1]]['n_estimators'][125] += 1
                    if 125 < int(temp1[0]) <= 150: dic[temp1[-1]]['n_estimators'][150] += 1
                    if float(temp1[2]) <=0.25: dic[temp1[-1]]['min_samples_split'][0.25]+=1
                    if 0.25 < float(temp1[2]) <= 0.5: dic[temp1[-1]]['min_samples_split'][0.5] += 1
                    if 0.5 < float(temp1[2]) <= 0.75: dic[temp1[-1]]['min_samples_split'][0.75] += 1
                    if 0.75 < float(temp1[2]) <= 1.0: dic[temp1[-1]]['min_samples_split'][1] += 1
                if temp1[-1]=="DecisionTreeClassifier":
                    if float(temp1[0]) <=0.25: dic[temp1[-1]][0.25]+=1
                    if 0.25 < float(temp1[0]) <= 0.5: dic[temp1[-1]][0.5] += 1
                    if 0.5 < float(temp1[0]) <= 0.75: dic[temp1[-1]][0.75] += 1
                    if 0.75 < float(temp1[0]) <= 1.0: dic[temp1[-1]][1] += 1
                if temp1[-1]=="KNeighborsClassifier":
                    if int(temp1[0]) <=6: dic[temp1[-1]][6]+=1
                    if 6 < int(temp1[0]) <= 12: dic[temp1[-1]][12] += 1
                    if 12 < int(temp1[0]) <= 18: dic[temp1[-1]][18] += 1
                    if 18 < int(temp1[0]) <= 24: dic[temp1[-1]][24] += 1
                if temp1[-1]=="LogisticRegression":
                    if int(temp1[2]) <=120: dic[temp1[-1]][120]+=1
                    if 120 < int(temp1[2]) <= 240: dic[temp1[-1]][240] += 1
                    if 240 < int(temp1[2]) <= 360: dic[temp1[-1]][360] += 1
                    if 360 < int(temp1[2]) <= 480: dic[temp1[-1]][480] += 1
                if temp[-1] in ['MaxAbsScaler','MinMaxScaler','StandardScaler','Normalizer',
                    'KernelCenterer']:
                    dic[temp[-1]]+=1
                if temp[-1]=="Binarizer":
                    if float(temp[0]) <= 25.0: dic[temp[-1]][25]+=1
                    if 25.0 < float(temp[0]) <= 50.0 : dic[temp[-1]][50] += 1
                    if 50.0 < float(temp[0]) <= 75.0: dic[temp[-1]][75] += 1
                    if 75.0 < float(temp[0]) <= 100.0: dic[temp[-1]][100] += 1
                if temp[-1]=="QuantileTransformer":
                    if int(temp[0]) <=250: dic[temp[-1]][250]+=1
                    if 250 < int(temp[0]) <= 500 : dic[temp[-1]][500] += 1
                    if 500 < int(temp[0]) <= 750: dic[temp[-1]][750] += 1
                    if 750 < int(temp[0]) <= 1000: dic[temp[-1]][1000] += 1
                if temp[-1]=="RobustScaler":
                    if int(temp[0]) <=25 and 50<= int(temp[1])< 75: dic[temp[-1]][1]+=1
                    if int(temp[0]) <= 25 and 75 <= int(temp[1]) <= 100: dic[temp[-1]][2] += 1
                    if 25 < int(temp[0]) <= 50 and 50 <= int(temp[1]) < 75: dic[temp[-1]][3] += 1
                    if 25 < int(temp[0]) <= 50 and 75 <= int(temp[1]) <= 100: dic[temp[-1]][4] += 1
    return dic


if __name__ == '__main__':

    temp_fi={}
    for i in folders:
        temp={}
        for j in files:
            dic=dump_files(i,j)
            temp[j]=sorted(dic["counter_full"][0.2][29])
        temp_fi[i]=temp
        # print(dic["settings"])
        # draw(dic['temp'],i)

        #draw_iqr(dic['counter_full'], i)

        # dic_settings=para_samples(dic["counter_full"][0.05],dic["settings"],i)
        # print(dic_settings)

        # del dic["temp"]
        # del dic["time"]
        # del dic["counter_full"]
        # del dic["settings"]
        # l=[]
        # for x in dic.keys():
        #     l.append([str(x)]+dic[x])
        # rdivDemo(l)

        # draw_boxplot(dic,i)

    with open('../../data/issue_close_time/dodge.pickle', 'wb') as handle:
        pickle.dump(temp_fi, handle)


