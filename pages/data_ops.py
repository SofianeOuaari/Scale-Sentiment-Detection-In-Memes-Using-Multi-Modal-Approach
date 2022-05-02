import base64
import math
import time
import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.feature_selection import SelectKBest,chi2,f_regression,f_classif,RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier ,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier,XGBRegressor


class FileDownloader(object):
	
	def __init__(self, data,filename='myfile',file_ext='txt'):
		super(FileDownloader, self).__init__()
		self.data = data
		self.filename = filename
		self.file_ext = file_ext

	def download(self):
		b64 = base64.b64encode(self.data.encode()).decode()
		new_filename = "{}_{}_.{}".format(self.filename,time.strftime("%Y%m%d-%H%M%S"),self.file_ext)
		href = f'Result <a href="data:file/{self.file_ext};base64,{b64}" download="{new_filename}">Download File</a>'
		st.markdown(href,unsafe_allow_html=True)



def app():
    st.markdown("<html><body><center><h1>Data Processing Operations</h1></center></body></html>",unsafe_allow_html=True)
    st.markdown("<html><body><center><p>In any data science/analysis and machine learning pipeline implementation, preprocessing the data is considered a core and essential step. In this section of the application, we offer to the user a set of automated tools to perform different type of processing operations applied on numerical data such as scaling/normalization, dimensionality reduction and selecting the best features for a given supervised task.</p></center></body></html>",unsafe_allow_html=True)
    st.markdown("<html><body><h3>Please Upload your dataset to proceed</h3></body></html>",unsafe_allow_html=True)
    
    data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])

    if 'activation_button' not in st.session_state: 
        st.session_state.activation_button=False
    
    def callback(): 
        st.session_state.activation_button=True
    
    if data is not None:
        df=pd.read_csv(data)
        col1,col2,col3=st.columns(3)
        with col1:
            label_name=st.selectbox("Select the main targeted label",tuple(df.columns))
        with col2:
            pass
        with col3: 
            task=st.selectbox("Supervised Task",("Classification","Regression"))

        col1_1,col2_1,col3_1,col4_1,col5_1=st.columns(5)

        with col1_1:
            pass
        with col2_1:
            pass
        with col3_1: 
            pass
            
        with col4_1: 
            pass
        with col5_1: 
            pass
               

        if (st.button("Access Operations",on_click=callback) or st.session_state.activation_button):
            st.session_state.activation_button=True
            label=df[label_name]
            label_encoder=LabelEncoder()
            y=label_encoder.fit_transform(label)
            df_filter=df[set(df.columns)-set([label_name])]
            st.dataframe(df_filter)

            st.markdown("<html><body><h3>Data Scaling/Normalization</h3></body></html>",unsafe_allow_html=True)

            col1,col2,col3=st.columns(3)
            with col1:
                pass
            with col2:
                scaling_technique=st.selectbox("Select normalization technique",("Min/Max Scaling","Standard Scaling"))
                scaler_types={"Min/Max Scaling":MinMaxScaler(),"Standard Scaling":StandardScaler()}
                scaler=scaler_types[scaling_technique]

            with col3: 
                pass
            if 'scaling_button' not in st.session_state: 
                st.session_state.scaling_button=False
            
            def callback_scaling(): 
                st.session_state.scaling_button=True
            
            if (st.button("Run Scaling",on_click=callback_scaling) or st.session_state.scaling_button):
                df_scaled=scaler.fit_transform(df_filter)

                df_scaled=pd.DataFrame(df_scaled,columns=[column+"_scaled" for column in df_filter.columns])
                df_scaled["Label"]=y
                st.dataframe(df_scaled)
                download = FileDownloader(df_scaled.to_csv(),filename="scaled_data",file_ext='csv').download()






            st.markdown("<html><body><h3>Feature Selection</h3></body></html>",unsafe_allow_html=True)
            

            st.markdown("<html><body><h5>SelectKBest Method</h5></body></html>",unsafe_allow_html=True)
            col1_2, col2_2,col3_2 = st.columns(3)
            with col1_2:
                pass
            with col2_2:
                feature_selection=st.slider("How many features do you want to select?",min_value=2,max_value=len(df_filter.columns),on_change=callback)
            with col3_2: 
                pass
            if task=="Classification": 
                score_function=f_classif
            elif task=="Regression": 
                score_function=f_regression
            

            if 'feature_selection_button' not in st.session_state: 
                st.session_state.feature_selection_button=False
            
            def callback_feature_selection(): 
                st.session_state.feature_selection_button=True

            if (st.button("Run Feature Selection",on_click=callback_feature_selection) or st.session_state.feature_selection_button):
                select = SelectKBest(score_func=score_function, k=feature_selection)
                print(y)
                selector=select.fit(df_filter,y)

                cols = selector.get_support(indices=True)
                features_df_new = df_filter.iloc[:,cols]
                features_df_new[label_name]=y
                st.dataframe(features_df_new)
                download = FileDownloader(features_df_new.to_csv(),filename=f"selected_{feature_selection}",file_ext='csv').download()
            

            
            st.markdown("<html><body><h5>Tree-Based Feature Selection</h5></body></html>",unsafe_allow_html=True)
            if task=="Classification":
                tree_type=st.selectbox("Select the model",["Random Forest Classifier","Decision Tree Classifier","XGBoostClassifier"])
                tree_models={"Random Forest Classifier":RandomForestClassifier(),"Decision Tree Classifier":DecisionTreeClassifier(),"XGBoostClassifier":XGBClassifier()}
            elif task=="Regression": 
                tree_type=st.selectbox("Select the model",["Decision Tree Regressor","Random Forest Regressor","XGBoostRegressor"])
                tree_models={"Random Forest Classifier":RandomForestRegressor(),"Decision Tree Regressor":DecisionTreeRegressor(),"XGBoostClassifier":XGBRegressor()}

            tree_model=tree_models[tree_type]
            tree_model.fit(df_filter,y)
            f_i = list(zip(df_filter.columns,tree_model.feature_importances_))
            f_i.sort(key = lambda x : x[1])
            fig, ax = plt.subplots()
            ax.barh([x[0] for x in f_i],[x[1] for x in f_i])
            st.pyplot(fig)

            st.markdown("<html><body><h5>Recursive Feature Elimination(RFE) </h5></body></html>",unsafe_allow_html=True)
            st.markdown("<html><body><p>Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through any specific attribute or callable. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.</p></body></html>",unsafe_allow_html=True)
            
            col1_2, col2_2,col3_2 = st.columns(3)
            with col1_2:
                pass
            with col2_2:
                feature_selection_rfe=st.slider("How many features do you want to select?",min_value=2,max_value=len(df_filter.columns),on_change=callback,key="rfe_slider")
            with col3_2: 
                pass
            

            if 'feature_selection_button_rfe' not in st.session_state: 
                st.session_state.feature_selection_button_rfe=False
            
            def callback_feature_selection_rfe(): 
                st.session_state.feature_selection_button_rfe=True

            if (st.button("Run Feature Selection RFE",on_click=callback_feature_selection_rfe) or st.session_state.feature_selection_button_rfe):
                rfe = RFE(LogisticRegression(), n_features_to_select=feature_selection_rfe, step=1)
                selector_rfe=rfe.fit(df_filter,y)

                cols = selector_rfe.get_support(indices=True)
                features_df_new = df_filter.iloc[:,cols]
                features_df_new[label_name]=y
                st.dataframe(features_df_new)
                download = FileDownloader(features_df_new.to_csv(),filename=f"selected_rfe_{feature_selection}",file_ext='csv').download()

            st.markdown("<html><body><h2>Dimensionality Reduction</h2></body></html>",unsafe_allow_html=True)
            col1, col2,col3,col4 = st.columns(4)

            with col1: 
                pass
            with col2: 
                dim_choice=st.selectbox("Select Dimensionality Reduction Technique",("PCA","Truncated SVD","t-SNE"))
            with col3:
                n_dimension=st.number_input("Select Reduced Number of Dimensions",2,len(df_filter.columns)-1)
            with col4: 
                pass
            pca_pipeline=Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=n_dimension))])
            svd_pipeline=Pipeline([('scaling', StandardScaler()), ('pca', TruncatedSVD(n_components=n_dimension))])
            tsne_pipeline=Pipeline([('scaling', StandardScaler()), ('pca', TSNE(n_components=n_dimension))])
            dim_models={"PCA":pca_pipeline,"Truncated SVD":svd_pipeline,"t-SNE":tsne_pipeline}
            dim_model=dim_models[dim_choice]

            if 'dimensionality_reduction' not in st.session_state: 
                st.session_state.dimensionality_reduction=False
            
            def callback_dimensionality_reduction(): 
                st.session_state.dimensionality_reduction=True
            
            if (st.button("Run Dimensionality Reduction",on_click=callback_dimensionality_reduction) or st.session_state.dimensionality_reduction):
                df_reduced=pd.DataFrame(dim_model.fit_transform(df_filter))
                df_reduced[label_name]=y
                st.dataframe(df_reduced) 
                download = FileDownloader(features_df_new.to_csv(),filename=f"reduced_dimensions_{n_dimension}",file_ext='csv').download()
