# -*- coding: utf-8 -*-
"""
Created on Mon Aug 07 16:26:56 2017

@author: M179100
"""

from collections import defaultdict
class print_results:
    def __init__(self,gt,res):
        #gt="C:/CT/results/gt_nyt.txt" 
        #results="C:/CT/data/results/results_train.txt" 
        self.gt=gt
        self.res=res
        result = self.reader(self.res)
        ground_truth = self.reader(self.gt)
        self.evaluate(ground_truth, result)
    def reader(self,file_path):
        data = defaultdict(set)
        with open(file_path) as f:
            for line in f:
                if line:
                    did, name, label = line.strip().split('\t')
                    data[did].add((name.lower(), label))
        #print 'load', len(data), 'docs'
        return data
    def evaluate(self,ground_truth,result):
        gt_dict=ground_truth
        pred_dict=result
        overlap_size = 0.0
        pred_size = 0.0
        gt_size = 0.0
    
        for did in gt_dict:
            #print did
            gt_size += len(gt_dict[did])
            #print gt_sizes
            if did in pred_dict:
                #print did
                pred_size += len(pred_dict[did])
                overlap_size += len(gt_dict[did] & pred_dict[did])
    
        print 'Precision = ', overlap_size / pred_size
        print 'Recall = ', overlap_size / gt_size
        print 'F1 = ', 2*(overlap_size / pred_size * overlap_size / gt_size) / (overlap_size / pred_size + overlap_size / gt_size)


