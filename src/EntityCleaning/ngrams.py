# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 16:59:26 2017

@author: M179100
"""

"""
n gram frequency counts based on gensim
"""
import re
import os
import cPickle as pickle
from gensim.models import phrases
from nltk.corpus import stopwords

class ngrams:
    def __init__(self,raw_text_path,inter_data_path):
        self.raw_text_path=raw_text_path
        self.inter_data_path=inter_data_path
        clean_sentences=self.clean_rawText()
        print "generating pkl dictionaries for ngram counts"
        self.ngram_counts(clean_sentences)
    def clean_rawText(self) : 
        rawFile=open(self.raw_text_path)
        lines=rawFile.readlines()
        stopword_list = set(stopwords.words("english"))
        clean_text_w2v=[]
        for l in lines:
            raw_sentence=l.split("\t")[1].strip()
            clean_sentence= re.sub("[^a-zA-Z. ]", "", raw_sentence)          
            words=clean_sentence.split()
            clean_words=[w.lower() for w in words if not w in stopword_list] 
            clean_text_w2v.append(clean_words)
        return clean_text_w2v        
       
    def ngram_counts(self,clean_sentences): 
        ''' threshold is for PMI score and min_count is for word counts in order to not miss any 
            person that is seldom make the min_count =1 '''
        phrase = phrases.Phrases(clean_sentences,min_count=1,threshold=2,delimiter=' ')
        #len(phrase.vocab.keys())
        bigrams = phrases.Phrases(phrase[clean_sentences],min_count=1,threshold=2,delimiter=' ')
        #len(bigrams.vocab.keys())
        trigrams= phrases.Phrases(bigrams[clean_sentences],min_count=1,threshold=2,delimiter=' ')
        #len(trigrams.vocab.keys())
        
        #unigram_count_dict={}
        #bigram_count_dict={}
        #trigram_count_dict={}
        #ngram_count_dict=trigrams.vocab
        """
        i=0
        for k in trigrams.vocab.keys():
            if i%100000==0:
                print i,"done"
                

            if len(k.split("_"))>=4:
                quadgram_count_dict[k]=trigrams.vocab[k]
            elif len(k.split("_"))==3:
                trigram_count_dict[k]=trigrams.vocab[k]
            elif len(k.split("_"))==2:
                bigram_count_dict[k]=trigrams.vocab[k]
            else:
                unigram_count_dict[k]=trigrams.vocab[k]

            i=i+1 
        
        """
        
        os.chdir(self.inter_data_path)    
        pickle.dump( trigrams.vocab, open( "ngram_count_dict.pkl", "wb" ) )
        





"""        
def get_PMI_score(gram):
    words=gram.split(" ")
    freqs=
    
     



len(required_phrases.vocab.keys())
len(required_phrases.vocab.values())
bigram = phrases.Phraser(required_phrases)
keys=bigram.phrasegrams
print(bigram[clean_sentences[0]])

tech_companies = [
    Company('Apple', 114.18), Company('Google', 908.60), Company('Microsoft', 69.18)
]
save_object(tech_companies, 'tech_companies.pkl')

        
required_phrases= phrases.Phrases(clean_sentences,min_count=5, threshold=100)
bigram = phrases.Phraser(required_phrases)
len(bigram.phrasegrams)
trigram_phrase= phrases.Phrases(bigram[clean_sentences],min_count=1,threshold=100)
trigram = phrases.Phraser(trigram_phrase)
len(trigram.phrasegrams)
"""