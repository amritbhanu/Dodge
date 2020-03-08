from __future__ import print_function, division

import pandas as pd
import os
from scipy.io import arff

cwd=os.getcwd()
data_path=os.path.join(cwd,"..","..","data", "UCI")

def preprocess_adult():
    path=os.path.join(data_path,"adult")
    df1=pd.read_csv(path+"/adult.data",header=None)
    df2=pd.read_csv(path+"/adult.test",header=None)
    df=pd.concat([df1, df2], ignore_index=True)
    cat_columns = [1, 3, 5, 6, 7, 8, 9, 13]

    df[cat_columns] = df[cat_columns].astype(str)
    #print(df[df[1].astype(str).str.contains('\?')].head())
    for i in cat_columns:
        df.drop(df[df[i].astype(str).str.contains('\?')].index, inplace=True)
    df.dropna(inplace=True)

    df[cat_columns]=df[cat_columns].astype('category')
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    df[14]=df[14].apply(lambda x: x.split(".")[0])
    df[14] = df[14].apply(lambda x: 0 if x==' <=50K' else 1)
    df.to_csv(data_path+"/adult.csv",index=False,header=False)

def preprocess_waveform():
    # waveform.data actual data, waveform+noise, have some noises as well with same class labels
    path=os.path.join(data_path,"waveform")
    df1 = pd.read_csv(path + "/waveform.data", header=None)
    df=df1[df1[21]!=2]
    df.to_csv(data_path + "/waveform.csv", index=False, header=False)

def preprocess_shuttle():
    path = os.path.join(data_path, "statlog_shuttle")
    df1 = pd.read_csv(path + "/shuttle.trn", header=None,sep=" ")
    df2 = pd.read_csv(path + "/shuttle.tst", header=None,sep=" ")
    df = pd.concat([df1, df2], ignore_index=True)
    df=df[df[9].isin([1,4])]
    df[9]=df[9].apply(lambda x: 1 if x==4 else 0)
    df.to_csv(data_path + "/shuttle.csv", index=False, header=False)

def preprocess_satellite():
    path = os.path.join(data_path, "statlog_satimage")
    df1 = pd.read_csv(path + "/sat.trn", header=None, sep=" ")
    df2 = pd.read_csv(path + "/sat.tst", header=None, sep=" ")
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df[36].isin([1, 4])]
    df[36] = df[36].apply(lambda x: 1 if x == 4 else 0)
    df.to_csv(data_path + "/satellite.csv", index=False, header=False)

def preprocess_pendigits():
    path = os.path.join(data_path, "pendigits")
    df1 = pd.read_csv(path + "/pendigits.tra", header=None)
    df2 = pd.read_csv(path + "/pendigits.tes", header=None)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df[16].isin([2, 4])]
    df[16] = df[16].apply(lambda x: 1 if x == 4 else 0)
    df.to_csv(data_path + "/pendigits.csv", index=False, header=False)

def preprocess_optdigits():
    path = os.path.join(data_path, "optdigits")
    df1 = pd.read_csv(path + "/optdigits.tra", header=None)
    df2 = pd.read_csv(path + "/optdigits.tes", header=None)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df[64].isin([1, 3])]
    df[64] = df[64].apply(lambda x: 1 if x == 3 else 0)
    df.to_csv(data_path + "/optdigits.csv", index=False, header=False)

def preprocess_covtype():
    path = os.path.join(data_path, "covtype")
    df = pd.read_csv(path + "/covtype.data", header=None)
    df = df[df[54].isin([5, 4])]
    df[54] = df[54].apply(lambda x: 1 if x == 4 else 0)
    df.to_csv(data_path + "/covtype.csv", index=False, header=False)

def preprocess_cancer():
    path = os.path.join(data_path, "breast-cancer-wisconsin")
    df1 = pd.read_csv(path + "/wdbc.data", header=None)

    df1["class"]=df1[1].apply(lambda x: 1 if "M" in x else 0)
    df1.drop(labels=[0,1], axis=1, inplace=True)
    df1.to_csv(data_path + "/cancer.csv", index=False, header=False)

def preprocess_diabetic():
    path = os.path.join(data_path, "Diabetic")
    df1 = pd.read_csv(path + "/diabetic.csv")
    df1.to_csv(data_path + "/diabetic.csv", index=False, header=False)

def preprocess_annealing():
    path=os.path.join(data_path,"annealing")
    df1=pd.read_csv(path+"/anneal.data",header=None)
    df2=pd.read_csv(path+"/anneal.test",header=None)
    df=pd.concat([df1, df2], ignore_index=True)
    drop_columns=[0,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,35,37]
    df.drop(labels=drop_columns,axis=1, inplace=True)
    for i in df.columns:
        df.drop(df[df[i].astype(str).str.contains('\?')].index, inplace=True)

    cat_columns = [1, 2, 31]
    df[cat_columns]=df[cat_columns].astype('category')
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    df = df[df[38].isin(['3','2'])]
    df[38]=df[38].apply(lambda x: 0 if x=='3' else 1)
    df.dropna(inplace=True)
    df.to_csv(data_path+"/annealing.csv",index=False,header=False)

def preprocess_autism():
    path=os.path.join(data_path,"autism")
    data=arff.loadarff(path+"/autism.arff")
    df=pd.DataFrame(data[0])
    df.drop('age_desc', axis=1, inplace=True)
    for i in df.columns:
        df.drop(df[df[i].astype(str).str.contains('\?')].index, inplace=True)
        df[i]=df[i].astype('float64',errors='ignore')

    cat_columns = ['gender', 'ethnicity', 'jundice', 'austim','contry_of_res','used_app_before','relation']
    df[cat_columns] = df[cat_columns].astype('category')
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    df['Class/ASD']=df['Class/ASD'].apply(lambda x: 0 if x=='NO' else 1)
    df.dropna(inplace=True)
    df.to_csv(data_path+"/autism.csv",index=False,header=False)

def preprocess_audit():
    path = os.path.join(data_path, "audit")
    df1 = pd.read_csv(path + "/audit_risk.csv")
    df1.drop('LOCATION_ID', axis=1, inplace=True)
    df1.dropna(inplace=True)
    df1.to_csv(data_path + "/audit.csv", index=False, header=False)

def preprocess_bank():
    path = os.path.join(data_path, "bank")
    df = pd.read_csv(path + "/bank-additional.csv",sep=';')
    for i in df.columns:
        df.drop(df[df[i].astype(str).str.contains('unknown')].index, inplace=True)
    cat_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact','month',
                   'day_of_week','poutcome']
    df[cat_columns] = df[cat_columns].astype('category')
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    df['y'] = df['y'].apply(lambda x: 0 if x == 'no' else 1)
    df.dropna(inplace=True)
    df.to_csv(data_path + "/bank.csv", index=False, header=False)

def preprocess_blood_transfusion():
    path = os.path.join(data_path, "blood-transfusion")
    df = pd.read_csv(path + "/transfusion.data")
    df.dropna(inplace=True)
    df.to_csv(data_path + "/blood-transfusion.csv", index=False, header=False)

def preprocess_car():
    path = os.path.join(data_path, "car")
    df = pd.read_csv(path + "/car.data",header=None)
    cat_columns=[0,1,2,3,4,5]
    df[cat_columns] = df[cat_columns].astype('category')
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    df[6] = df[6].apply(lambda x: 0 if x == 'unacc' else 1)
    df.to_csv(data_path + "/car.csv", index=False, header=False)

def preprocess_cardiotocography():
    path = os.path.join(data_path, "cardiotocography")
    df = pd.read_csv(path + "/ctg.csv")
    df['NSP'] = df['NSP'].apply(lambda x: 0 if x == 1 else 1)
    df.to_csv(data_path + "/cardiotocography.csv", index=False, header=False)

def preprocess_cervical_cancer():
    path = os.path.join(data_path, "cervical-cancer")
    df = pd.read_csv(path + "/risk_factors_cervical_cancer.csv")
    for i in df.columns:
        df.drop(df[df[i].astype(str).str.contains('\?')].index, inplace=True)
    df.to_csv(data_path + "/cervical-cancer.csv", index=False, header=False)

def preprocess_climate_sim():
    path = os.path.join(data_path, "climate-sim")
    l=[]
    with open(path + "/pop_failures.dat", 'r') as f:
        for doc in f.readlines():
            l.append(doc.strip().split())
    df=pd.DataFrame(l[1:],columns=None)
    df.drop([0,1],axis=1,inplace=True)
    df.to_csv(data_path + "/climate-sim.csv", index=False, header=False)

def preprocess_contraceptive():
    path = os.path.join(data_path, "contraceptive")
    df = pd.read_csv(path + "/cmc.data",header=None)
    df[9]=df[9].apply(lambda x: 0 if x==1 else 1)
    df.to_csv(data_path + "/contraceptive.csv", index=False, header=False)

def preprocess_credit_approval():
    path = os.path.join(data_path, "credit-approval")
    df = pd.read_csv(path + "/crx.data", header=None)
    for i in df.columns:
        df.drop(df[df[i].astype(str).str.contains('\?')].index, inplace=True)
    cat_columns=[0,3,4,5,6,8,9,11,12]
    df[cat_columns] = df[cat_columns].astype('category')
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    df[15]=df[15].apply(lambda x: 1 if x=='+' else 0)
    df.to_csv(data_path + "/credit-approval.csv", index=False, header=False)

def preprocess_crowdsource():
    path = os.path.join(data_path, "crowdsource")
    df = pd.read_csv(path + "/training.csv")
    df = df[df['class'].isin(['grass', 'farm'])]
    df['label']=df['class'].apply(lambda x: 0 if x=='farm' else 1)
    df.drop('class', axis=1, inplace=True)
    df.to_csv(data_path + "/crowdsource.csv", index=False, header=False)

def preprocess_sensorless_drive():
    path = os.path.join(data_path, "sensorless-drive")
    df = pd.read_csv(path + "/Sensorless_drive_diagnosis.txt",header=None,sep=' ')
    df = df[df[48].isin([1, 2])]
    df[48] = df[48].apply(lambda x: 0 if x == 1 else 1)
    df.to_csv(data_path + "/sensorless-drive.csv", index=False, header=False)

def preprocess_credit_default():
    path = os.path.join(data_path, "credit-default")
    df = pd.read_csv(path + "/credit.csv")
    df.to_csv(data_path + "/credit-default.csv", index=False, header=False)

def preprocess_drug_consumption():
    path = os.path.join(data_path, "drug-consumption")
    df = pd.read_csv(path + "/drug_consumption.data", header=None)
    df.drop(0,axis=1, inplace=True)
    cat_columns=list(range(13,31))
    df[cat_columns] = df[cat_columns].astype('category')
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    df[31]=df[31].apply(lambda x: 0 if x=='CL0' else 1)
    df.to_csv(data_path + "/drug-consumption.csv", index=False, header=False)

def preprocess_electric_stable():
    path = os.path.join(data_path, "electric-stable")
    df = pd.read_csv(path + "/electric.csv")
    df.drop("stab", axis=1, inplace=True)
    df["stabf"]=df["stabf"].apply(lambda x: 0 if x=="unstable" else 1)
    df.to_csv(data_path + "/electric-stable.csv", index=False, header=False)

def preprocess_htru2():
    path = os.path.join(data_path, "htru2")
    df = pd.read_csv(path + "/HTRU_2.csv",header=None)
    df.to_csv(data_path + "/htru2.csv", index=False, header=False)

def preprocess_hepmass():
    path = os.path.join(data_path, "hepmass")
    df = pd.read_csv(path + "/1000_test.csv")
    df1=df[df['# label']==1.0]
    df2 = df[df['# label'] == 0.0]
    df1=df1.sample(n=1000)
    df2=df2.sample(n=1000)
    df=pd.concat([df1, df2], ignore_index=True)
    df.dropna(inplace=True)
    df['class']=df['# label'].apply(lambda x: 0 if x==0.0 else 1)
    df.drop(['# label'],axis=1, inplace=True)
    df.to_csv(data_path + "/hepmass.csv", index=False, header=False)

def preprocess_liver():
    path = os.path.join(data_path, "liver")
    df = pd.read_csv(path + "/liver.csv",header=None)
    cat_columns = [1]
    df[cat_columns] = df[cat_columns].astype('category')
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    df[10]=df[10].apply(lambda x: 0 if x==2 else 1)
    df.dropna(inplace=True)
    df.to_csv(data_path + "/liver.csv", index=False, header=False)

def preprocess_image():
    path = os.path.join(data_path, "image")
    df1 = pd.read_csv(path + "/segmentation.data")
    df2 = pd.read_csv(path + "/segmentation.csv")
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df['label'].isin(["WINDOW", "PATH"])]
    df['class'] = df['label'].apply(lambda x: 0 if x == "WINDOW" else 1)
    df.drop(["label"],axis=1,inplace=True)
    df.to_csv(data_path + "/image.csv", index=False, header=False)

def preprocess_kddcup():
    path = os.path.join(data_path, "kddcup")
    df = pd.read_csv(path + "/kddcup.data_10_percent",header=None)
    cat_columns = [1,2,3]
    df[cat_columns] = df[cat_columns].astype('category')
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    df = df[df[41].isin(["normal.", "back."])]
    df1 = df[df[41] == "normal."]
    df2 = df[df[41] == "back."]
    df1 = df1.sample(n=1000)
    df = pd.concat([df1, df2], ignore_index=True)
    df[41] = df[41].apply(lambda x: 0 if x == "normal." else 1)
    df.to_csv(data_path + "/kddcup.csv", index=False, header=False)

def preprocess_gamma():
    path = os.path.join(data_path, "gamma")
    df = pd.read_csv(path + "/magic04.data", header=None)
    df[10] = df[10].apply(lambda x: 0 if x == "g" else 1)
    df.to_csv(data_path + "/gamma.csv", index=False, header=False)

def preprocess_hand():
    path = os.path.join(data_path, "hand")
    df = pd.read_csv(path + "/hand.csv")
    for i in df.columns:
        df.drop(df[df[i].astype(str).str.contains('\?')].index, inplace=True)
        df[i] = df[i].astype('float64', errors='ignore')
    df=df[df["Class"].isin([5,1])]
    df['label'] = df["Class"].apply(lambda x: 0 if x == 5 else 1)
    df.drop('Class',axis=1,inplace=True)
    df.to_csv(data_path + "/hand.csv", index=False, header=False)

def preprocess_mushrooom():
    path = os.path.join(data_path, "mushroom")
    df = pd.read_csv(path + "/mushroom.csv",header=None)
    for i in df.columns:
        df.drop(df[df[i].astype(str).str.contains('\?')].index, inplace=True)
    cat_columns=list(range(1,23))
    df[cat_columns] = df[cat_columns].astype('category')
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    df['label'] = df[0].apply(lambda x: 0 if x == 'e' else 1)
    df.drop([0], axis=1, inplace=True)
    df.to_csv(data_path + "/mushroom.csv", index=False, header=False)

def preprocess_shop_intention():
    path = os.path.join(data_path, "shop-intention")
    df = pd.read_csv(path + "/online_shoppers_intention.csv")
    cat_columns = ['Month','VisitorType','Weekend']
    df[cat_columns] = df[cat_columns].astype('category')
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    df['Revenue']=df['Revenue'].apply(lambda x: 0 if x==False else 1)
    df.to_csv(data_path + "/shop-intention.csv", index=False, header=False)

def preprocess_phishing():
    path = os.path.join(data_path, "phishing")
    data = arff.loadarff(path + "/Dataset.arff")
    df = pd.DataFrame(data[0])
    df['Result']=df['Result'].apply(lambda x: 0 if x=="-1" else 1)
    df.to_csv(data_path + "/phishing.csv", index=False, header=False)

def preprocess_bankrupt():
    path = os.path.join(data_path, "bankrupt")
    data = arff.loadarff(path + "/4year.arff")
    df = pd.DataFrame(data[0])
    df.dropna(inplace=True)
    df.to_csv(data_path + "/bankrupt.csv", index=False, header=False)

def preprocess_biodegrade():
    path = os.path.join(data_path, "biodegrade")
    df = pd.read_csv(path + "/biodeg.csv",header=None,sep=';')
    df[41]=df[41].apply(lambda x: 0 if x=='NRB' else 1)
    df.to_csv(data_path + "/biodegrade.csv", index=False, header=False)

def tsting():
    df=pd.read_csv(data_path+"/diabetic.csv",header=None)
    neg = df[df[df.columns[-1]] == 0]
    pos = df[df[df.columns[-1]] == 1]
    cut_pos = int(pos[0].count() * 0.8)
    cut_neg = int(neg[0].count() * 0.8)
    pos_1, pos_2 = pos.iloc[:cut_pos, :], pos.iloc[cut_pos:, :]
    neg_1, neg_2 = neg.iloc[:cut_neg, :], neg.iloc[cut_neg:, :]
    df = pd.concat([pos_1, neg_1, pos_2, neg_2], ignore_index=True)


if __name__ == '__main__':
    # preprocess_biodegrade()
    # preprocess_bankrupt()
    # preprocess_phishing()
    # preprocess_shop_intention()
    # preprocess_mushrooom()
    # preprocess_hand()
    # preprocess_gamma()
    # preprocess_kddcup()
    # preprocess_image()
    preprocess_liver()
    # preprocess_hepmass()
    # preprocess_htru2()
    # preprocess_electric_stable()
    # preprocess_drug_consumption()
    # preprocess_credit_default()
    # preprocess_sensorless_drive()
    # preprocess_crowdsource()
    # preprocess_credit_approval()
    # preprocess_contraceptive()
    # preprocess_climate_sim()
    # preprocess_cervical_cancer()
    # preprocess_cardiotocography()
    # preprocess_car()
    # preprocess_blood_transfusion()
    # preprocess_bank()
    # preprocess_audit()
    # preprocess_autism()
    # preprocess_adult()
    # preprocess_waveform()
    # preprocess_shuttle()
    # preprocess_satellite()
    # preprocess_pendigits()
    # preprocess_optdigits()
    # preprocess_covtype()
    # preprocess_cancer()
    # preprocess_diabetic()
    # preprocess_annealing()


