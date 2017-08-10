# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:35:54 2017

@author: M179100
"""
import os
from collections import defaultdict
from string import punctuation

class get_seed_file:
    def __init__(self,seed_file_path,type_file_path,intermediate_data_path):
        """ getting mentions and types """
        print "GENERATING SEED FILE"
        self.seed_file_path=seed_file_path
        self.type_file_path=type_file_path
        mention_type_dict= self.get_mention_type(0.9) 
        len(mention_type_dict.keys())
        mentions=list(set([m.split("_")[1]+'!!'+str(mention_type_dict[m][0]) for m in mention_type_dict.keys()])) 
        len(mentions)
        #i=0
        
        os.chdir(intermediate_data_path)
        with open ("seed_file.txt","w") as sf:
            for m in mentions:
                #print i
                values='\t'.join(m.split("!!"))
                sf.write(values+"\n")
                #i=i+1
        print "SEED FILE !! DONE"

    def load_type_file(self):
        type_tid = defaultdict(int)
        tid_type = defaultdict(str)
        with open(self.type_file_path) as f:
            for line in f:
                entry = line.strip().split('\t')
                if len(entry) == 2:
                    type_tid[entry[0]] = int(entry[1])
                    tid_type[int(entry[1])] = entry[0]
        return (type_tid, tid_type, len(type_tid))

    def get_mention_type(self,confidence_threshold):
        type_tid, tid_type, T = self.load_type_file()    
        mention = defaultdict(tuple)
        score_set = set()
        """
        anomaly_dict={}
        anomaly_path="C:\CT\data\Intermediate\Anomaly_by_type.txt"
        anomaly_file=open(anomaly_path)
        anomaly_lines=anomaly_file.readlines()
        for an in anomaly_lines:
            en,types=an.split("\t")
            anomaly_dict[types.strip("\n")]=set(en.split(","))    
        """
        
        with open(self.seed_file_path) as f:
            fl=f.readlines()
            for line in range(len(fl)):                
                '''
                filen=open(file_name)
                lines=filen.readlines()
                line=lines[0]
                '''
                entry = fl[line].strip().split('\t')
                if len(entry) == 7:
                    doc_id = entry[0]
                    mention_string = entry[1].lower().strip(punctuation) # strip punctuation
                    mention_notableType = entry[2].strip(";")
                    mention_type = entry[3].strip(";")
                    # filter
                    percentOfSecondRank = float(entry[6])
                    simScore = float(entry[5])
                    if percentOfSecondRank == -1:
                        percentOfSecondRank = 0.0
                    score_set.add(percentOfSecondRank)
                    if simScore > 0.05 and percentOfSecondRank <= (1 - confidence_threshold*confidence_threshold):
                        if mention_type in type_tid: # target type (positive examples)
                            mention[doc_id + '_' + mention_string] = (type_tid[mention_type], simScore)
                        """
                        try:
                            
                                if mention_string not in anomaly_dict[mention_type]:
                                    mention[doc_id + '_' + mention_string] = (type_tid[mention_type], simScore)
                                else:
                                    print "FOUND ANOMALY",mention_string,mention_type
                        except:
                            pass
                        """
        return mention

        
        
        
        
        
        
        
        
        
        
"""        
def load_type_file(type_file_path):
    type_tid = defaultdict(int)
    tid_type = defaultdict(str)
    with open(type_file_path) as f:
            for line in f:
                entry = line.strip().split('\t')
                if len(entry) == 2:
                    type_tid[entry[0]] = int(entry[1])
                    tid_type[int(entry[1])] = entry[0]
    return (type_tid, tid_type, len(type_tid))

type_tid, tid_type, T = load_type_file(type_file_path)


    
    





import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

# Example settings
n_samples = 200
outliers_fraction = 0.25
clusters_separation = [0, 1, 2]

# define two outlier detection tools to be compared
classifiers = {
    "One-Class SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,
                                     kernel="rbf", gamma=0.1),
    "Robust covariance": EllipticEnvelope(contamination=outliers_fraction),
    "Isolation Forest": IsolationForest(max_samples=n_samples,
                                        contamination=outliers_fraction,
                                        random_state=rng)}

# Compare given classifiers under given settings
xx, yy = np.meshgrid(np.linspace(-7, 7, 500), np.linspace(-7, 7, 500))
n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction * n_samples)
ground_truth = np.ones(n_samples, dtype=int)
ground_truth[-n_outliers:] = -1

# Fit the problem with varying cluster separation
for i, offset in enumerate(clusters_separation):
    np.random.seed(42)
    # Data generation
    X1 = 0.3 * np.random.randn(n_inliers // 2, 2) - offset
    X2 = 0.3 * np.random.randn(n_inliers // 2, 2) + offset
    X = np.r_[X1, X2]
    # Add outliers
    X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]

    # Fit the model
    plt.figure(figsize=(10.8, 3.6))
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        # fit the data and tag outliers
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        threshold = stats.scoreatpercentile(scores_pred,
                                            100 * outliers_fraction)
        y_pred = clf.predict(X)
        n_errors = (y_pred != ground_truth).sum()
        # plot the levels lines and the points
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        subplot = plt.subplot(1, 3, i + 1)
        subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                         cmap=plt.cm.Blues_r)
        a = subplot.contour(xx, yy, Z, levels=[threshold],
                            linewidths=2, colors='red')
        subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                         colors='orange')
        b = subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='white')
        c = subplot.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='black')
        subplot.axis('tight')
        subplot.legend(
            [a.collections[0], b, c],
            ['learned decision function', 'true inliers', 'true outliers'],
            prop=matplotlib.font_manager.FontProperties(size=11),
            loc='lower right')
        subplot.set_title("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
        subplot.set_xlim((-7, 7))
        subplot.set_ylim((-7, 7))
    plt.subplots_adjust(0.04, 0.1, 0.96, 0.92, 0.1, 0.26)

plt.show()














































import numpy as np
import matplotlib.pyplot as plt
from LocalOutlierFactor import LocalOutlierFactor

np.random.seed(42)

# Generate train data
X = 0.3 * np.random.randn(100, 2)
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X + 2, X - 2, X_outliers]

# fit the model
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(X)
y_pred_outliers = y_pred[200:]

# plot the level sets of the decision function
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Local Outlier Factor (LOF)")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

a = plt.scatter(X[:200, 0], X[:200, 1], c='white',
                edgecolor='k', s=20)
b = plt.scatter(X[200:, 0], X[200:, 1], c='red',
                edgecolor='k', s=20)
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a, b],
           ["normal observations",
            "abnormal observations"],
           loc="upper left")
plt.show()
"""