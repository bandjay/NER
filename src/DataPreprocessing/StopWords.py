__author__ = 'ahmed'
import os
class StopWords:
    def __init__(self,data_path):
        os.chdir(os.path.join(data_path,"stopwords"))
        path = 'en.txt'
        f = open(path, 'r')
        self.stop_words = set([line.strip() for line in f])
    def isStopWord(self, word):
        if word in self.stop_words:
            return True
        return False