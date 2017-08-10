# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:52:36 2017

@author:Jay
"""

"""
Assign raw,seed and type file names and keep these three files in the data directory of software
"""
import time
import os 
import pickle
start=time.time()

raw_text_file="nyt13_110k.txt" 
seed_file="seed_nyt.txt"
type_file="type_tid.txt"
collocation_measure="t-score" # PMI /t-score

significance="2"
capitalize=1
maxLength=4 # maximal phrase length
minSup=100 # minimal support for phrases in candidate generationSegmentOutFile='result/yelp/segment.txt'
NumRelationPhraseClusters='4000' # number of relation phrase clusters

''' creating all necessary paths '''
os.chdir("C:/CT")
pwd=os.getcwd()
os.popen("mkdir results")                 ## creating results directory
data_path=os.path.join(pwd,"data")
intermediate_data_path=os.path.join(data_path,"Intermediate")
src_file_path=os.path.join(pwd,"src")
raw_text_path=os.path.join(data_path,raw_text_file) 
seed_file_path=os.path.join(data_path,seed_file)
type_file_path= os.path.join(data_path,type_file)
results_files_path=os.path.join(pwd,"results")
SegmentOutFile=os.path.join(results_files_path,"segment.txt")
DataStatsFile=os.path.join(results_files_path,"data_model_stats.txt") # data statistics on graph construction
ResultFile=os.path.join(results_files_path,"results.txt") # typed entity mentions
ResultFileInText=os.path.join(results_files_path,"resultsInText.txt") # typed mentions annotated in segmented text
sentences_path=os.path.join(intermediate_data_path,"sentences.txt")
full_sentence_path=os.path.join(intermediate_data_path,"full_sentences.txt")
pos_path=os.path.join(intermediate_data_path,"pos.txt")
full_pos_path=os.path.join(intermediate_data_path,"full_pos.txt")
frequent_patterns_path=os.path.join(intermediate_data_path,"frequentPatterns.pickle")
out_path = os.path.join(results_files_path,"segment.txt")
gt_path=os.path.join(pwd,"results")
Entity_file_path=os.path.join(results_files_path,"segment.txt")
results_file_path=os.path.join(data_path,"results")
seeds_processed_file=os.path.join(intermediate_data_path,"seed_file.txt")
anomaly_path=os.path.join(os.path.join(intermediate_data_path,"Anomaly_by_type.txt"))


os.chdir(data_path)
dir_comm=str("mkdir Intermediate")     ## creating intermediate data directory
os.popen(dir_comm)

"""Data Preprocess"""
os.chdir(os.path.join(src_file_path,"DataPreprocessing"))
filen=open(raw_text_path)
lines=filen.readlines()
num_lines=len(lines)
from Clean import Clean
C = Clean(raw_text_path)
print "Start candidate generation..."
C.clean_and_tag()

"""FP mine"""
os.chdir(os.path.join(src_file_path,"FrequentPhraseMining"))
from FrequentPatternMining import FrequentPatternMining
documents = []
print "FP started"
phrase_segments_file=os.path.join(intermediate_data_path,"phrase_segments.txt")
with open(phrase_segments_file,'r') as f:
    for line in f:
        documents.append(line.strip())
FPM = FrequentPatternMining(documents, maxLength, minSup)
FrequentPatterns = FPM.mine_patterns()
FP_pickle_file=os.path.join(intermediate_data_path,"frequentPatterns.pickle")
pickle.dump(FrequentPatterns, open(FP_pickle_file, "w"))


""" Entity Extraction """
os.chdir(os.path.join(src_file_path,"EntityExtraction"))
from EntityRelation import EntityRelation
print "candidate generation started"
ER = EntityRelation(sentences_path,full_sentence_path,pos_path,full_pos_path,frequent_patterns_path,significance,out_path,capitalize)
ER.extract()
print 'Candidate generation done.'

""" Graph """
os.chdir(os.path.join(src_file_path,"src"))
### step 0
from step0 import step0
step0(out_path,DataStatsFile,data_path)

""" Word counts """
os.chdir(os.path.join(src_file_path,"EntityCleaning"))
from ngrams import ngrams
ngrams(raw_text_path,intermediate_data_path)


""" RUN with either PMI or t-score because segement file will be updated based on collocation score """
""" cleaning entity mentions based on collocation measure(PMI) """
if collocation_measure=="PMI":
    os.chdir(os.path.join(src_file_path,"EntityCleaning"))
    from collocation import collocation
    collocation(Entity_file_path,intermediate_data_path,num_lines,"PMI")

if collocation_measure=="t-score":
    """ cleaning entity mentions based on collocation measure(t-score) """
    os.chdir(os.path.join(src_file_path,"EntityCleaning"))
    from collocation import collocation
    collocation(Entity_file_path,intermediate_data_path,num_lines,"t-score")

""" outlier detection in seed file by type """
os.chdir(os.path.join(src_file_path,"NoiseDetection"))
from get_seed_file import get_seed_file
from wordEmbeddings import wordEmbeddings
from AutoEncoder import AutoEncoder
get_seed_file(seed_file_path,type_file_path,intermediate_data_path)
wordEmbeddings(raw_text_path,seeds_processed_file,intermediate_data_path,300)
''' to reduce autoencoder running time change model parameters 
in AutoEncoder.py in ~/src/NoiseDetection
parameters == "hidden=[1000,500,20,500,1000], epochs = 800" '''
AutoEncoder("WE.csv",intermediate_data_path,300)


""" Running final step in graph construction with cleaned seed file and entity mentions """
os.chdir(os.path.join(src_file_path,"src"))
from step1 import step1
step1(out_path,seed_file_path,type_file_path,NumRelationPhraseClusters,ResultFile,ResultFileInText,data_path,anomaly_path)
end=time.time()

''' printing results '''
res_file=os.path.join(results_file_path,"results.txt")
gt_file=os.path.join(gt_path,"gt_nyt.txt")
os.chdir(os.path.join(src_file_path,"src"))
from print_results import print_results
print_results(gt_file,res_file)


print "Finished in ",float(end-start)/float(60*60),"hours"
                     

                     

