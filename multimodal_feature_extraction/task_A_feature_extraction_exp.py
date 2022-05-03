import json
import pandas as pd 
import numpy as np
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






file_names_train=["train_latent_autoencoder_3.csv","emb_avg_train.csv","residual_block_1_1_train.csv","residual_block_2_1_train.csv"]
file_names_test=["test_latent_autoencoder_3.csv","emb_avg_test.csv","residual_block_1_1_test.csv","residual_block_2_1_test.csv"]



dict_models={"knn1":KNeighborsClassifier(1),"knn3":KNeighborsClassifier(3),"knn5":KNeighborsClassifier(5),"linear_svc":LinearSVC(C=500,random_state=0),"rf":RandomForestClassifier(2000,n_jobs=-1),"gb":GradientBoostingClassifier(2000),"mlp":MLPClassifier(hidden_layer_sizes=(100,100,50))}


for file_name_train,file_name_test in zip(file_names_train,file_names_test): 
    df_train=pd.read_csv(file_name_train)
    df_test=pd.read_csv(file_name_test)


    for model_name,m in dict_models.items():
        dict_stats={}
        fea_train=df_train[df_train.columns[:-1]]
        y_train=df_train[df_train.columns[-1]]

        fea_test=df_test[df_train.columns[:-1]]
        y_test=df_test[df_train.columns[-1]]

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
        lst_f1_stratified = []
        model=OneVsRestClassifier(m)

        for train_index, test_index in skf.split(fea_train, y_train):
            x_train_fold, x_test_fold = fea_train.iloc[train_index], fea_train.iloc[test_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
            model.fit(x_train_fold, y_train_fold)
            y_pred=model.predict(x_test_fold)
            f1=f1_score(y_test_fold,y_pred,average="macro")
            lst_f1_stratified.append(f1)
        
        lst_f1_stratified=np.array(lst_f1_stratified)
        dict_stats={"cross_std":lst_f1_stratified.std(),"cross_macro_f1":lst_f1_stratified.mean()}

        y_pred=model.predict(fea_test)
        dict_stats["test_macro_f1"]=f1_score(y_test,y_pred,average="macro")

        root_name=file_name_train.replace(".csv","")
        json_dict = open(f"{root_name}_{model_name}.json", "w")
        json.dump(dict_stats,json_dict)
        json_dict.close()