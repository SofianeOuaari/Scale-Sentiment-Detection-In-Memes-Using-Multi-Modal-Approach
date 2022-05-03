from pycaret.classification import *
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,VotingClassifier,ExtraTreesClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE,KMeansSMOTE,BorderlineSMOTE,SVMSMOTE
import pandas as pd 
import json



d={}
rbs=[1,2]
for rb in rbs:
    for k in [1,3,5]:
        d[k]={}
        all_humours=["humour","motivational","offensive","sarcasm"]

        for H in all_humours:
            dict_stats={}
            d[k][H]={}
            df_train=pd.read_csv(f"./residual_block_{rb}_{H}_train.csv")
            df_test=pd.read_csv(f"./residual_block_{rb}_{H}_test.csv")

            model=OneVsRestClassifier(KNeighborsClassifier(k,n_jobs=-1))
            fea_train=df_train[df_train.columns[:-1]]
            y_train=df_train[df_train.columns[-1]]
            fea_test=df_test[df_test.columns[:-1]]
            y_test=df_test[df_test.columns[-1]]

            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
            lst_f1_stratified = []

            for train_index, test_index in skf.split(fea_train, y_train):
                x_train_fold, x_test_fold = fea_train.iloc[train_index], fea_train.iloc[test_index]
                y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
                model.fit(x_train_fold, y_train_fold)
                y_pred=model.predict(x_test_fold)
                f1=f1_score(y_test_fold,y_pred,average="macro")
                lst_f1_stratified.append(f1)

            lst_f1_stratified=np.array(lst_f1_stratified)
            print(lst_f1_stratified)
            print(f"Mean: {lst_f1_stratified.mean()}")
            print(f"Std {lst_f1_stratified.std()}")
            model.fit(fea_train,y_train)
            y_pred=model.predict(fea_test)

            print(f1_score(y_test,y_pred,average="macro"))

            d[k][H]["Min"]=lst_f1_stratified.min()
            d[k][H]["Max"]=lst_f1_stratified.max()
            d[k][H]["Mean"]=lst_f1_stratified.mean()
            d[k][H]["Std"]=lst_f1_stratified.std()
            d[k][H]["test"]=f1_score(y_test,y_pred,average="macro")
            json_dict = open(f"RB_{rb}_{k}_{H}_stats.json", "w")
            json.dump(d,json_dict)
            json_dict.close()

