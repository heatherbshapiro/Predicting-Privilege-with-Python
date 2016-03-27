import pandas as pd
import numpy as np
import os
import re
from nltk.sentiment.util import *
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

path = "C:/Users/heathes/Desktop/all transcripts"

list = []
users = []

remove_list = ["interviewer","interviewee""shapiro","inaudible","heather","castingwords","par","line","silence", "course","coursera"]
 
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.txt'):
            f = open(path + '/' + file)
            content = f.readlines()
            content = ''.join([i for i in content if i not in remove_list])

            content = ''.join([i if ord(i) < 128  else ' ' for i in content])

            content = content.decode('utf-8', 'ignore').encode('utf-8')
            content = re.sub("/", " ", content)
            content = re.sub("@", " ", content)
            content = re.sub("\\|", " ", content)
            content = re.sub("\n","", content)
            content = content.lower()
            content = re.sub("interviewer","",content)
            content = re.sub("interviewee","",content)
            content = re.sub("massive open online courses", "MOOCs", content)
            content = re.sub("massive open online course", "MOOC",content)
            content = re.sub("silence","",content)
            content = re.sub("á","",content)
            content = content.strip()
            list.append(content)
            
            userName = re.sub(".txt","",file)
            if userName == 'S10\xa0':
                userName = "S10"
               
            users.append(userName)

classification = pd.read_csv("C:/Users/heathes/Dropbox/HS-CourseraInterviews/data/Classification Assignments-People.csv")

df = pd.DataFrame()

df['transcript'] = list
df['number'] = users


df = pd.merge(df, classification, on = "number")
df = df.drop("Name",1)

sentences = []

from vaderSentiment.vaderSentiment import sentiment as vaderSentiment 


for index, row in df.iterrows():
    print index
    sentences = tokenize.sent_tokenize(row["transcript"])
    sumComp = 0
    sumPos = 0
    sumNeg = 0
    tot = len(sentences)
    pos = 0
    neg = 0
    neu = 0
    compPos = 0
    compNeg = 0
    compNeutral = 0
    for sentence in sentences: 
        vs = vaderSentiment(sentence)
        sumComp += vs['compound']
        sumPos += vs['pos']
        sumNeg += vs['neg']
        if vs['pos'] > .25:
            pos +=1
        if vs['neg'] > .25:
            neg += 1
        if vs['neu'] > .5:
            neu +=1
        if vs['compound'] > .25:
            compPos +=1
        if vs['compound'] < -.25:
            compNeg +=1
        if -.25<= vs['compound'] <.25:
            compNeutral +=1
    df.loc[index,'compound'] = sumComp / tot
    df.loc[index,'compNeutral'] = compNeutral
    df.loc[index, 'comppos'] = compPos
    df.loc[index, 'compneg'] = compNeg
    df.loc[index,'percentPos'] = sumPos / tot
    #print row

df.to_csv("dataset2.csv")
#print df['number'][0],sumPos/tot, sumNeg/tot, sumComp/tot

print df[df['Age'] == '18-24']['compound'].mean()
