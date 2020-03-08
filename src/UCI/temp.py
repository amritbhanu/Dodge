from __future__ import print_function, division

import pandas as pd
import os
cwd = os.getcwd()
import_path=os.path.abspath(os.path.join(cwd, '..'))
import sys
sys.path.append(import_path)

data_path = os.path.join(cwd, "..","..", "data","UCI")

file_inc = {"adult": 0, "cancer": 1, "covtype":  2, "diabetic":3, "optdigits":4, "pendigits":5
            , "satellite":6, "shuttle":7, "waveform":8,"annealing":9,"audit":10,"autism":11,
            "bank":12,"bankrupt":13,"biodegrade":14,"blood-transfusion":15,"car":16,
            "cardiotocography":17,"cervical-cancer":18, "climate-sim":19,"contraceptive":20,
            "credit-approval":21,"credit-default":22,"crowdsource":23,"drug-consumption":24,
            "electric-stable":25,"gamma":26,"hand":27,"hepmass":28,"htru2":29,"image":30,
            "kddcup":31,"liver":32,"mushroom":33,"phishing":34,"sensorless-drive":35,"shop-intention":36}

if __name__ == '__main__':
    for i in file_inc.keys():
        df=pd.read_csv("../../data/UCI/" +i+".csv",header=None)
        pos=df[df[df.columns[-1]]==1][0].count()
        total=df[0].count()
        print("&", i, "&", total, "&",len(df.columns)-1,"&", int(round(pos/total,2)*100),"\\\hline")
