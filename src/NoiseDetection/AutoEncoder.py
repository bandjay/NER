# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 15:35:50 2017

@author: M179100
"""

"""
REMOVING NOISE by the assumptions
1. Entities of same Type will fall in to a cluster based on word embeddings
2. Entities that belong to different type will fit into a single cluster based on METAPATH using word embeddings 
"""

"""
##################################################################
############### Auto encoder for noise detection #################
##################################################################
"""
import h2o
import os
import pandas as pd
import numpy as np
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator

class AutoEncoder:
    def __init__(self,w2v_df_File,intermediate_data_path,size):
        print "Running Auto Encoder"
        self.w2v_df_File=w2v_df_File
        self.intermediate_data_path=intermediate_data_path
        self.size=size
        self.detect_anomaly()
        print "Anomalies !!done"
        
    def detect_anomaly(self): 
        os.chdir(self.intermediate_data_path)               
        wv_df=pd.read_csv(self.w2v_df_File,header=None,index_col=False) 
        ty=pd.unique(wv_df[self.size+1])
        h2o.init(max_mem_size = 6) 
        wv=h2o.import_file(self.w2v_df_File)
        anomaly_list=[]
        for t in range(len(ty)):
            w=wv[wv['C302']==str(ty[t])]  
            predictors = w.col_names[0:300]
            train = w[predictors]
            dl_model = H2ODeepLearningEstimator(hidden=[100,20,100], epochs = 100,activation="Tanh",autoencoder=True)
            dl_model.train(x=w.col_names[0:300],training_frame= train)
            e=dl_model.anomaly(train,per_feature=False)
            w['errors']=e
            w2 = w.as_data_frame(use_pandas=True)
            anomaly=np.array(w2[w2['errors']>w2['errors'].quantile(q=0.99)]['C301'])
            anomaly_list.append([anomaly,ty[t]])
        
        with open("Anomaly_by_type.txt","w") as wfile:
            for i in range(len(anomaly_list)):
                wfile.write(str(",".join(anomaly_list[i][0]))+str("\t")+str(anomaly_list[i][1])+str("\n"))

"""                
os.chdir(intermediate_data_path)               
wv_df=pd.read_csv("WE.csv",header=None,index_col=False) 
ty=pd.unique(wv_df[300+1])
h2o.init(max_mem_size = 6) 
wv=h2o.import_file("WE.csv")
anomaly_list=[]
for t in range(len(ty)):
            w=wv[wv['C302']==ty[t]]  
            predictors = w.col_names[0:300]
            train = w[predictors]
            dl_model = H2ODeepLearningEstimator(hidden=[500,100,500], epochs = 100,activation="Tanh",autoencoder=True)
            dl_model.train(x=w.col_names[0:300],training_frame= train)
            e=dl_model.anomaly(train,per_feature=False)
            w['errors']=e
            w2 = w.as_data_frame(use_pandas=True)
            anomaly=np.array(w2[w2['errors']>w2['errors'].quantile(q=0.95)]['C301'])
            anomaly_list.append([anomaly,ty[t]])
        
with open("Anomaly.txt","w") as wfile:
    for i in range(len(anomaly_list)):
        wfile.write(str(",".join(anomaly_list[i][0]))+str("\t")+str(anomaly_list[i][1])+str("\n"))

"""