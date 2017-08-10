# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:49:28 2017

@author: M179100
"""

from gensim.models import Phrases
from nltk.corpus import stopwords
import re
import os
import math
from multiprocessing.dummy import Pool as ThreadPool
import cPickle as pickle


''' Extracting entities and relation '''
class collocation:
    
    def __init__(self,Entity_file_path,intermediate_data_path,num_lines,score):
        print "start"
        self.score=score
        self.num_lines=num_lines
        e_file=open(Entity_file_path)
        e_lines=e_file.readlines()
        #e_lines=e_lines[0:5]
        pwd=os.getcwd()
        os.chdir(intermediate_data_path)
        self.ngram_dictionary=pickle.load(open( "ngram_count_dict.pkl", "rb" )) 
        os.chdir(pwd)
        pool = ThreadPool(8) 
        if self.score=="PMI":
            doc_lines= pool.map(self.pmi_process, e_lines)
        elif self.score=="t-score":
            doc_lines= pool.map(self.t_process, e_lines)
        op_path=os.path.split(Entity_file_path)[0]
        with open(os.path.join(op_path,"segment.txt"),"w") as segf:
            for li in doc_lines:
                segf.write(li+"\n")

    def pmi_process(self,line):
        #line=e_lines[0]
        #line="0	In:RP,Cyprus:EP,it,ppv:RP,no,longer,safe,to,keep your money in:RP,a,bank,and,so,it,ppv,soon,ppv impossible:RP,to,get:RP,a,loan,from:RP,a,bank,there"
        doc_num,text=line.split("\t")
        words=text.split(",")
        entities=[re.sub(":EP","",w).strip("\n") for w in words if w.__contains__("EP")]
        entities_ngram=[en for en in entities if len(en.split(" "))>1]
        scored_entities=[en for en in entities if len(en.split(" "))==1]
        ''' before scoring run pos based filtering 
            POS function here'''
        for e in entities_ngram:
            if self.pmi_score_entity(e) > float(5):
                scored_entities.append(e)
        #scored_entities=["Cyprus"]
        words_temp=[re.sub(":EP","",w).strip("\n") for w in words]
        words_temp=[ w+":EP" if w in scored_entities else w for w in words_temp]
        words_temp=",".join(words_temp)
        return str(doc_num)+"\t"+words_temp

            

    def pmi_score_entity(self,entity,min_count=1):
        #entity="South Korea"
        words_e=entity.split(" ")
        #entity in ngram_dictionary.keys()
        word_freq=self.ngram_dictionary[entity]
        if word_freq>=min_count:
            freq_arr=[]
            for w1 in words_e:
                freq_arr.append(self.ngram_dictionary[w1])
            return self.get_pmi_score(word_freq,freq_arr)
        else: 
            return float(0)
        
    def get_pmi_score(self,w,w_arr):
        #w_arr=freq_arr
        #w=word_freq
        try:
            denom=reduce(lambda x, y: float(x) * float(y), w_arr, 1)
            return math.log(float(w)*float(self.num_lines)/denom,2) 
        except:
            return float(0)
        
    ''' using t-score '''
    def t_process(self,line):
        #line=e_lines[10]
        #line="0	In:RP,Cyprus:EP,it,ppv:RP,no,longer,safe,to,keep your money in:RP,a,bank,and,so,it,ppv,soon,ppv impossible:RP,to,get:RP,a,loan,from:RP,a,bank,there"
        doc_num,text=line.split("\t")
        words=text.split(",")
        entities=[re.sub(":EP","",w).strip("\n") for w in words if w.__contains__("EP")]
        entities_ngram=[en for en in entities if len(en.split(" "))>1]
        scored_entities=[en for en in entities if len(en.split(" "))==1]
        ''' before scoring run pos based filtering 
            POS function here'''
        for e in entities_ngram:
            if self.t_score_entity(e) :
                scored_entities.append(e)
        #scored_entities=["Cyprus"]
        words_temp=[re.sub(":EP","",w).strip("\n") for w in words]
        words_temp=[ w+":EP" if w in scored_entities else w for w in words_temp]
        words_temp=",".join(words_temp)
        
        return str(doc_num)+"\t"+words_temp

            

    def t_score_entity(self,entity,min_count=1):
        #entity="Julian Markham House"
        words_e=entity.split(" ")
        #entity in ngram_dictionary.keys()
        word_freq=self.ngram_dictionary[entity]
        if word_freq>min_count:
            freq_arr=[]
            for w1 in words_e:
                freq_arr.append(self.ngram_dictionary[w1])            
            return self.get_t_score(word_freq,freq_arr)
        else: 
            return False
    
    def get_t_score(self,w,w_arr):
        try:
        #t-test based approach to measure independence
        #p_w1=w1/n,p_w2 =w2/n
        #t=(x-mu)/sqrt(s^2/N) , x=w1w2/N ,mu=p_w1*pw_2 ,s^2=x(1-x)
        #N.H = independent(t>0.01) A.H=dependent(t<0.01)
            N=float(self.num_lines)
            #entity=bigram_entity[1]
            x_bar=float(w)/N
            s_square=float(x_bar*(1-x_bar))
            prob_words=[float(wi)/N for wi in w_arr]
            mu=reduce(lambda x, y: x * y, prob_words, 1)
            t_score=float((x_bar-mu)/math.sqrt(s_square/N))
            if t_score < 2.576 and t_score> -2.576:
                return True
            else :
                return False
        except:
            return False
        

     
    
    


"""
import re
import math
def get_entity(full_record):
    entity=[]
    entity_record=full_record[1]
    for en in entity_record:
        entity.append(re.sub(":EP","",en))
    return entity
all_entities=pool.map(get_entity,doc_entity_relation)
all_entities=[e for i in all_entities for e in i]
len(all_entities)
unique_entity_list=list(set(all_entities))
len(unique_entity_list)
unigram_entity=[]
bigram_entity=[]
for u in unique_entity_list:
    en_split=u.split(" ")
    if len(en_split)==1:
        unigram_entity.append(u.strip("\n"))
    elif len(en_split)==2:
        bigram_entity.append(u.strip("\n"))

def measure_collocation(entity):
    try:
        #t-test based approach to measure independence
        #p_w1=w1/n,p_w2 =w2/n
        #t=(x-mu)/sqrt(s^2/N) , x=w1w2/N ,mu=p_w1*pw_2 ,s^2=x(1-x)
        #N.H = independent(t>0.05) A.H=dependent(t<0.05)
        N=float(29834) 
        #entity=bigram_entity[1]
        x_bar=float(all_entities.count(entity))/N
        s_square=x_bar*(1-x_bar)
        words=entity.split(" ")
        prob_words=[]
        for w in words:
            prob_words.append(float(all_entities.count(w))/N)
            #print w
            #print float(all_entities.count(w))
        mu=reduce(lambda x, y: x * y, prob_words, 1)
        t_score=(x_bar-mu)/math.sqrt(s_square/N)
    except:
        print entity
    return t_score

bi_gram_scores=pool.map(measure_collocation,bigram_entity)
        

import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
finder = BigramCollocationFinder.from_words(nltk.corpus.genesis.words('english-web.txt'))
finder.nbest(bigram_measures.pmi, 10)        


tweet_phrases = "I love iphone . I am so in love with iphone . iphone is great . samsung is great . iphone sucks. I really really love iphone cases. samsung can never beat iphone . samsung is better than apple"
from nltk.collocations import *
import nltk
finder = BigramCollocationFinder.from_words(tweet_phrases.split(), window_size = 3)
finder1 = BigramCollocationFinder.from_words(tweet_phrases.split(), window_size = 3)
finder1.apply_freq_filter(2)
bigram_measures = nltk.collocations.BigramAssocMeasures()
for k,v in finder.ngram_fd.items():
  print(k,v)


import nltk
from nltk.collocations import *
from nltk.tokenize import word_tokenize

text = "this is a foo bar bar black sheep  foo bar bar black sheep foo bar bar black sheep shep bar bar black sentence"

trigram_measures = nltk.collocations.TrigramAssocMeasures()
finder = TrigramCollocationFinder.from_words(word_tokenize(text))

for i in finder.score_ngrams(trigram_measures.pmi):
    print i

"""


























