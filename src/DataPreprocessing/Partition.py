__author__ = 'ahmed'
from StopWords import StopWords
import os
class Partition:
    def __init__(self, punctuation,data_inter_path,data_path):
        self.punctuation = set(punctuation)
        self.num_words = 0
        self.sw = StopWords(data_path)
        os.chdir(data_inter_path)
        self.f = open('phrase_segments.txt','w')
        

    def split(self, sentence):
        new_sent = [None]*len(sentence)
        for i in xrange(len(sentence)):
            if sentence[i] in self.punctuation:
                new_sent[i] = ','
            else:
                new_sent[i] = sentence[i]
        mining_sentence = "".join(new_sent).lower().split(',')
        sentence = sentence.split(",")

        for seg in mining_sentence:
            seg = seg.split()
            new_set = []
            for word in seg:
                if not self.sw.isStopWord(word):
                    new_set.append(word)
            seg = " ".join(new_set)
            seg = seg.strip()
            if seg:
                self.f.write(seg+"\n")
        return sentence




