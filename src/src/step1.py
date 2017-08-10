from collections import defaultdict
from operator import itemgetter
from math import log, sqrt
from data_model import *
from algorithm import *
from clustype import *
from evaluation import *
from sklearn.preprocessing import normalize
import sys
import os.path

class step1:
    def __init__(self,out_path,seed_file,type_file,NumRelationPhraseClusters,ResultFile,ResultFileInText,data_path,anomaly_path):
        ### Parameter setting #############################################################
        segment_path = out_path
        ground_truth_path = seed_file # seed file
        type_path = type_file
        K = int(NumRelationPhraseClusters) # #RP clusters, K < 1000 
        annotation_path = ResultFile
        annotationInText_path = ResultFileInText
        data_path=data_path
        anomaly_path=anomaly_path
        
        if not os.path.isfile(ground_truth_path):
        	print 'Seed entity file no found! Please download from github or run entity linking module to generate.'
        	exit(1)
        
        confidence_score = 0.9 # to get seeds: PercentageOfRank < 1-conf^2
        TRAIN_RATIO = 1.0 # percentage of seeds used
        VERBOSE = False
        
        gamma = 0.001 # info from mention graph S_M
        mu = 10.0 # supervision from Y_0
        alpha = 0.5 # consistency with cluster consensus
        lambda_O = 1.0 # information from Y
        lambda_L = 1.0 # information from RP clustering
        
        clustype_ITER = 5 # max iterations for our full-model
        INNER_ITER = 200 #200  # max #iters for MultiNMF inner loop--perViewNMF
        OUTER_ITER = 5 # maxIter for MultiNMF outer loop
        ITER = 40 # max iter
        tol = 1e-6 # tolerance for convergence: perViewNMF, MultiNMF, full-model
        
        
        cid_path = data_path+'\\tmp\\candidate_cid.txt'
        mid_path = data_path+'\\tmp\\mention_mid.txt' # Y_0 = (mention, mid, initial_score)
        pid_path = data_path+'\\tmp\\RP_pid.txt'
        string_wid_path = data_path+'\\tmp\\string_wid.txt'
        context_wid_path = data_path+'\\tmp\\context_wid.txt'
        PiC_path = data_path+'\\tmp\\PiC.txt'
        PiL_path = data_path+'\\tmp\\PiL.txt'
        PiR_path = data_path+'\\tmp\\PiR.txt'
        mention_graph_path = data_path+'\\tmp\\W_M.txt'
        F_context_path = data_path+'\\tmp\\F_context.txt'
        F_string_path = data_path+'\\tmp\\F_string.txt'
        
        ### data preparation' #############################################################
        print 'Data loading...'
        type_tid, tid_type, T = load_type_file(type_path)
        cid_candidate, n = load_id(cid_path)
        mid_mention, m = load_id(mid_path)
        doc_mid = get_doc_mid(mid_mention) # doc_id - mid mapping
        pid_RP, l = load_id(pid_path)
        wid_string, ns = load_id(string_wid_path)
        wid_context, nc = load_id(context_wid_path)
        PiC = load_graph(PiC_path, m, n)
        PiL = load_graph(PiL_path, m, l)
        PiR = load_graph(PiR_path, m, l)
        PiC = normalize(PiC, norm='l2', axis=0) # column-normalize PiC to ensure PiC.T*PiC = I
        PiCC = PiC.T * PiC
        PiL = normalize(PiL, norm='l2', axis=0) # column-normalize PiC to ensure PiC.T*PiC = I
        PiLL = PiL.T*PiL
        PiR = normalize(PiR, norm='l2', axis=0) # column-normalize PiC to ensure PiC.T*PiC = I
        PiRR = PiR.T*PiR
        S_L = normalize_graph(PiC.T * PiL)
        S_R = normalize_graph(PiC.T * PiR)
        print '#links in S_L:', len(S_L.nonzero()[0]), ', #links in S_R:', len(S_R.nonzero()[0])
        W_M = load_graph(mention_graph_path, m, m)
        S_M = normalize_graph(W_M)
        print 'S_M dims', S_M.shape[0], S_M.shape[1], '#links in S_M:', len(S_M.nonzero()[0])
        del W_M
        F_context = load_graph(F_context_path, l, nc)
        F_string = load_graph(F_string_path, l, ns)
        print 'Loading DONE'
        
        ### Partition seed mentions by ratio #############################################################
        seedMention_tid_score = load_ground_truth(ground_truth_path, type_path,anomaly_path, confidence_score)
        seed_mid, _ = partition_train_test(mid_mention, TRAIN_RATIO)
        Y0 = set_Y(seed_mid, seedMention_tid_score, mid_mention, m, T).todense()
        print 'seeding DONE'
        
        ### run methods #############################################################
        ### target type set
        target_tid_set = set()
        for type in set(type_tid.keys()) - set(['NIL']):
        	target_tid_set.add(type_tid[type])
        
        print 'start ClusType...'
        Y, C, PL, PR = clustype_appx(S_L, S_R, S_M, PiC, PiL, PiR, Y0, lambda_O, gamma, mu, T, ITER, K)
        write_output(annotation_path, Y, doc_mid, mid_mention, tid_type)
        write_output_intext(segment_path, annotationInText_path, Y, doc_mid, mid_mention, tid_type)
        """
        print 'Start ClusType-Full... Clusters:', K 
        Y_full = clustype_inexact(S_L, S_R, S_M, PiC, PiL, PiR, F_context, F_string, Y0, lambda_O, gamma, mu, \
        	lambda_L, alpha, T, K, clustype_ITER, INNER_ITER, tol, Y, C, PL, PR, VERBOSE)
        """
        print 'ClusType done!'
        
        