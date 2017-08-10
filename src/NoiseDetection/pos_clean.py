# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 15:20:31 2017

@author: M179100
"""
import nltk
import pandas as pd
import os

class cleanEntity:
    
    def __init__(self,seed_file_path):
        print "cleaning mentions"
        self.seed_file_path=seed_file_path
        #self.seedpath=seedpath
    
    def pos_tagging(self):
        #os.chdir(self.seedpath)
        seedfile=open(self.seed_file_path)
        seeds=seedfile.readlines()
        Entity=[]
        Type=[]
        for s in seeds:
            type_split=s.split("\t")
            Entity.append(type_split[0])
            Type.append(type_split[1])
        pos_tag=[nltk.pos_tag(nltk.word_tokenize(x))[0] for x in Entity]
        pos_tag_df=pd.DataFrame.from_records(pos_tag)
        pos_tag_df.columns=["Entity","POS"]
        pos_tag_df["Type"]=Type
        #pos_list=list(pos_tag_df["POS"].unique())
        #for p in pos_list:
        #    sub_df=pos_tag_df[pos_tag_df["POS"]==p]
        #    print(sub_df.head(n=5))
        #    print("There are " ,sub_df.shape[0], " entities which are ",p)
        """ filtering out pos tags IN,DT,PRP$ """
        pos_tag_df=pos_tag_df[~pos_tag_df["POS"].isin(["IN","DT","PRP$"])]
        
        return pos_tag_df        