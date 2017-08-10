# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:20:42 2017

@author: M179100
"""

import os
import sys
#from cleanEntity import cleanEntity
from AutoEncoder import AutoEncoder
from wordEmbeddings import wordEmbeddings

class outlier: 
    def __init__(self,raw_text_path,intermediate_data_path):
        self.raw_text_path=raw_text_path
        self.intermediate_data_path=intermediate_data_path
        #seedFile="seed_file.txt"
        #rawTextFile="train_raw.txt"
        #os.chdir("D:/CS512/src")
        #print "cleaning entities"
        #CE=cleanEntity(self.seed_file)
        #clean_df=CE.pos_tagging()
        
        print "Generating word embeddings"
        WE=wordEmbeddings(rawTextFile,rawTextPath)
        w2v=WE.w2v_model(size=300)
        #w2v_vocab=w2v.wv.vocab
        WE.get_word_embeddings_df(seedFile,seedPath)
        
        print " AutoEncoder in progress"
            
        AE=AutoEncoder("WE.csv",dataPath,size=300)
        AE.detect_anomaly()
