import sys
from collections import defaultdict
import numpy as np
from pyspark import SparkContext
import csv
import re
import string


stopWords = [ "a", "i", "it", "am", "at", "on", "in", "to", "too", "very", \
              "of", "from", "here", "even", "the", "but", "and", "is", "my", \
              "them", "then", "this", "that", "than", "though", "so", "are" ]
stemEndings = [ "-s", "-es", "-ed", "-er", "-ly" "-ing", "-'s", "-s'" ]
punctuation = ".,:;!?"
remove_spl_char_regex = re.compile('[%s]' % re.escape(string.punctuation)) # regex to remove special characters
remove_num = re.compile('[\d]+')



def remove_punctuation(input_string):
    for item in punctuation:
        input_string = input_string.replace(item, '')
    return input_string


def tokenize(line):
    #global vocab
    a1 = line.strip().lower()
    a2 = remove_spl_char_regex.sub(" ",a1)  # Remove special characters
    a3 = remove_num.sub("", a2)  #Remove numbers
    a4 = remove_punctuation(a3)
    ws = a3.split()
    x = defaultdict(float)
    for w in ws:
                #vocab.setdefault(w, len(vocab))
                x[w] += 1
                
    return x

    

if __name__ == "__main__":
    vocab = {}
    sc = SparkContext(appName="PythonKMeans")
    filename = "/home/hduser/Downloads/2013-01.csv"
    lines = sc.textFile(filename)

    
    #data = lines.map(lambda x: len(x))
    data = lines.map(lambda x: x.split(",")[0])
    tok=data.map(tokenize)
    xs=[]

    for ws in tok.collect():
        x = defaultdict(float)
        for w in ws:
                vocab.setdefault(w, len(vocab))
                x[vocab[w]] += 1
        xs.append(x.items())
    print(len(vocab))        
