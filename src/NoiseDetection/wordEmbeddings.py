# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:03:00 2017

@author: M179100

"""

""" word 2 vec training class """
import os
import re
import numpy as np
import pandas as pd
import logging
from gensim.models import word2vec,Phrases
from nltk.corpus import stopwords


class wordEmbeddings:    
    def __init__(self,raw_text_path,seeds_processed_file,intermediate_data_path,size):
        print "Generating word Embeddings"
        #self.rawTextFile=rawTextFile
        self.raw_text_path=raw_text_path
        self.seeds_processed_file=seeds_processed_file
        self.intermediate_data_path=intermediate_data_path
        self.size=size
        self.w2v_model()
        self.get_word_embeddings_df()
        
        print "word embeddings done"
    
    """ Only letters,digits,space,period(.) are valid characters and words are converted to lower """  
                                                          
    def clean_rawText(self) : 
        #os.chdir(self.rawTextPath)
        rawFile=open(self.raw_text_path)
        lines=rawFile.readlines()
        stopword_list = set(stopwords.words("english"))
        clean_text_w2v=[]
        for l in lines:
            raw_sentence=l.split("\t")[1].strip()
            clean_sentence= re.sub("[^a-zA-Z0-9. ]", "", raw_sentence)          
            words=clean_sentence.split()
            clean_words=[w.lower() for w in words if not w in stopword_list] 
            clean_text_w2v.append(clean_words)
        return clean_text_w2v        
        
        
    def w2v_model(self):
        print "word2vec training" 
        clean_sentences= self.clean_rawText()
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        phrases = Phrases(clean_sentences)
        bigram = Phrases(phrases[clean_sentences])
        trigram= Phrases(bigram[clean_sentences])
        self.w2v_model = word2vec.Word2Vec(trigram[bigram[clean_sentences]], size= self.size)
        os.chdir(self.intermediate_data_path)
        self.w2v_model.save("w2v_model")
        return self.w2v_model
    
    def get_word_embeddings_df(self):
        np_arr=[]
        seedf=open(self.seeds_processed_file)
        seeds=seedf.readlines()
        type_dict={0:"/time/event",1:"/location/location",2:"/people/person",3:"/organization/organization"}
        for s in seeds:
            e=s.split("\t")[0].lower()
            ty=int(s.split("\t")[1].strip("\n"))
            if e in self.w2v_model.wv.vocab:
                np1=np.array(self.w2v_model[e])
                np2=np.append(e,type_dict[ty])
                arr1=np.append(np1,np2)
                np_arr.append(arr1)    
        embed_df=pd.DataFrame.from_records(np_arr)
        os.chdir(self.intermediate_data_path)
        embed_df.to_csv("WE.csv",header=False,index=False)
        
        
"""
we=wordEmbeddings(raw_text_path,seed_file_path,intermediate_data_path,300)
w2v=we.w2v_model()
np_arr=[]
seedf=open(seeds_processed_file)
type_dict={0:"/time/event",1:"/location/location",2:"/people/person",3:"/organization/organization"}
seeds=seedf.readlines()
for s in range(len(seeds)):
    e=seeds[s].split("\t")[0].lower()
    type=int(seeds[s].split("\t")[1].strip("\n"))
    print s,e,t
    if e in w2v.wv.vocab:
        np1=np.array(w2v[e])
        np2=np.append(e,type_dict[type])
        arr1=np.append(np1,np2)
        np_arr.append(arr1)    
embed_df=pd.DataFrame.from_records(np_arr)
os.chdir(intermediate_data_path)
embed_df.to_csv("WE.csv",header=False,index=False)


"""        
    
       






