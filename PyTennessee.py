
# coding: utf-8

# # Pandas

# ### How we did it in the past

# In[1]:

import csv
numbers=[]
gender = []

f = open("C:/Users/heathes/Documents/Visual Studio 2015/Projects/PyTennesse/PyTennesse/dataset.csv")
csv_f = csv.reader(f)
for  row in csv_f:
    numbers.append(row[2])
    gender.append(row[3])
    
print numbers, gender


# In[2]:

lines = []
with open('C:/Users/heathes/Documents/Visual Studio 2015/Projects/PyTennesse/PyTennesse/dataset.csv') as f:
    reader = csv.reader(f)
    your_list = list(reader)

# print your_list

for nested_list in your_list:
    print '\t'.join(nested_list)


# ### Setting up the Environment

# In[1]:

import pandas as pd
import numpy as np
import os
import re
from nltk import tokenize
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment
import matplotlib.pyplot as plt


# ### Importing the Transcripts and removing StopWords

# In[3]:

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



# ### Creating and Merging DataFrames

# In[6]:

classification = pd.read_csv("C:/Users/heathes/Dropbox/HS-CourseraInterviews/data/Classification Assignments-People.csv")

df = pd.DataFrame()

df['transcript'] = list
df['number'] = users


df = pd.merge(df, classification, on = "number")
df = df.drop("Name",1)
    


# ## Sentiment Analysis GUI

# ### Setting up the environment

# In[8]:

from vaderSentiment.vaderSentiment import sentiment as vaderSentiment 
from Tkinter import *


# ### Creating the Class

# In[9]:

class Application(Frame):
    def __init__(self,master):
        Frame.__init__(self,master)
        self.grid()
        self.create_widgets()

    def create_widgets(self):
        self.instruction = Label(self,text ="Find the Sentiment Score of your sentence!")
        self.instruction.grid(row=0, column=0, columnspan = 2, sticky= W)

        self.sentence = Label(self, text = "Enter a sentence: ")
        self.sentence.grid(row=1, column=0, sticky=W)

        self.entry = Entry(self)
        self.entry.grid(row = 1, column=1,sticky=W)

        self.submit_button= Button(self,text="submit", command=self.reveal)
        self.submit_button.grid(row=2, column=0, sticky = W)

        self.text = Text(self,width=35,height = 5, wrap = WORD)
        self.text.grid(row=3, column = 0, columnspan = 2, sticky = W)

        self.clear_button = Button(self, text = "Clear text", command =self.clear_text)
        self.clear_button.grid(row=2, column=1, sticky=W)
    def reveal(self):
        self.text = Text(self,width=35,height = 5, wrap = WORD)
        self.text.grid(row=3, column = 0, columnspan = 2, sticky = W)
        sentences = self.entry.get()
       
        message = vaderSentiment(sentences)
       
        self.text.insert(0.0,"This is the sentiment score\n" + str(message)+ "\n ")
            
    def clear_text(self):
        self.text = Text(self,width=35,height = 5, wrap = WORD)
        self.text.grid(row=3, column = 0, columnspan = 2, sticky = W)


# ### Starting up the GUI

# In[10]:

root = Tk()
root.title("Buttons")

app = Application(root)

root.mainloop()


# ### Finding the average sentiment for every transcript

# In[11]:

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
    if row["Education"]=="Some College" or row["Education"]=="Less than HS" or row["Education"]=="Associate":
        df.loc[index,"Education"]="Less than Bachelors"
    if row["Age"]=="18-24" or row["Age"]=="25-34":
        df.loc[index, "Age"] = "18-34"
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
    df.loc[index,'Total'] = tot


# In[12]:

df.to_csv("C:/Users/heathes/Documents/Visual Studio 2015/Projects/PyTennesse/PyTennesse/dataset2.csv")


# In[16]:

df["compound"].describe()


# ## Analysis

# In[17]:

df = pd.read_csv("C:/Users/heathes/Documents/Visual Studio 2015/Projects/PyTennesse/PyTennesse/dataset.csv")

#print df

dfFull = pd.read_csv("C:/Users/heathes/Documents/Visual Studio 2015/Projects/PyTennesse/PyTennesse/dataset2.csv")


# In[18]:

dfFull =dfFull.drop('Unnamed: 0',1)
dfFull


# In[19]:

fig, axes = plt.subplots(2, 2)
axes[0,0].hist(dfFull['compound'], color="#3F5D7D")
axes[0,0].set_title("Average Compound Sentiment")
# axes[0,0].annotate(dfFull['compound'].describe(), xy=(1, 0), xycoords='axes fraction', fontsize=14,
#                 xytext=(-5, 140), textcoords='offset points',
#                 ha='right', va='bottom')
axes[0,0].spines["top"].set_visible(False)  
axes[0,0].spines["right"].set_visible(False) 



axes[0,1].hist(dfFull['comppos'], color="#3F5D7D")
axes[0,1].set_title("Positive Compound Sentiments")
# axes[0,1].annotate(dfFull['comppos'].describe(), xy=(1, 0), xycoords='axes fraction', fontsize=14,
#                 xytext=(-5, 140), textcoords='offset points',
#                 ha='right', va='bottom')
axes[0,1].spines["top"].set_visible(False)  
axes[0,1].spines["right"].set_visible(False)

axes[1,0].hist(dfFull['compneg'], color="#3F5D7D")
axes[1,0].set_title("Negative Compound Sentiments")
# axes[1,0].annotate(dfFull['compneg'].describe(), xy=(1, 0), xycoords='axes fraction', fontsize=14,
#                 xytext=(-5, 140), textcoords='offset points',
#                 ha='right', va='bottom')

axes[1,0].spines["top"].set_visible(False)  
axes[1,0].spines["right"].set_visible(False) 

axes[1,1].hist(dfFull['compNeutral'], color="#3F5D7D")
axes[1,1].set_title("Neutral Compound Sentiments")
# axes[1,1].annotate(dfFull['compNeutral'].describe(), xy=(1, 0), xycoords='axes fraction', fontsize=14,
#                 xytext=(-5, 140), textcoords='offset points',
#                 ha='right', va='bottom')
axes[1,1].spines["top"].set_visible(False)  
axes[1,1].spines["right"].set_visible(False)

# plt.setp([a.get_xticklabels() for a in axes[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axes[:, 1]], visible=False)
plt.xticks(fontsize=14)  


plt.show()


# In[ ]:




# In[20]:

dfFull.ix[:,-5:-2].plot(kind='hist', alpha = 0.5, stacked=False)
plt.title("Frequency of Sentiment Scores")
plt.show()


# In[21]:

dfFull.ix[:,-5:-1].hist()

plt.show()


# In[62]:

from statsmodels.graphics.mosaicplot import mosaic

mosaic(dfFull,['Age','compound'])


# In[23]:

dfFull["Continent"]= dfFull["Continent"].astype("category", categories = ["America", "Africa","Asia", "Europe"], ordered = True)


# In[24]:

dfFull["Income"]=dfFull["Income"].astype("category", categories = ["20000-34999","<20000","35000-49999", "50000-74999", "75000-99999","100000+"], ordered = True)


# In[25]:

import statsmodels.formula.api as smf

# create a fitted model in one line
lm = smf.ols(formula='comppos~Total+C(Age)+C(Gender)+C(Education)+C(Income)+C(Continent)', data=dfFull).fit()

# print the coefficients
lm.params.keys
# lm.params.values

x = lm.params.keys().tolist()
y = lm.params.values
upper = lm.conf_int()[1].tolist()
lower =lm.conf_int()[0].tolist()

dfLM = pd.DataFrame()

dfLM["x"]=x
dfLM['y']=y
dfLM["upper"]= upper
dfLM["lower"]= lower

dfLM2= dfLM.sort_values(by="x", ascending=False)

dfLM2['y']= dfLM2['y']
dfLM2=dfLM2[dfLM2.x !="Intercept"]

y_pos = np.arange(len(x)-1)
error = (dfLM2["upper"]+dfLM2["lower"])
dfLM2["error"] = error

# plt.barh(y_pos, dfLM2['y'], xerr=error, align='center', alpha=0.4)
plt.plot(dfLM2['y'],y_pos, "o")
plt.errorbar(dfLM2['y'],y_pos,xerr=error.tolist(),fmt = 'o')
plt.yticks(y_pos, dfLM2['x'])
plt.ylabel('Demographic')
plt.title('Regression Coefficients')
plt.axvline(0, color='k', linestyle='--')


plt.show()


# In[ ]:




# # Word Cloud Tutorial

# ### Setting up the environment

# In[16]:

from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from wordcloud import WordCloud, STOPWORDS
import matplotlib

#d = path.dirname(__file__)
d = "C:/Users/heathes/Documents/Visual Studio 2015/Projects/PythonApplication1/PythonApplication1/"
# Read the whole text.
#text = open("C:/Users/heathes/Desktop/all transcripts/C1.txt").read()



# In[17]:

df = pd.read_csv("C:/Users/heathes/Documents/Visual Studio 2015/Projects/PyTennesse/PyTennesse/dataset2.csv")
text = " ".join(df['transcript'].astype(str))
remove = ["interviewer","interviewee""shapiro","inaudible","heather","castingwords","par","line","silence","course","coursera","courses","lot", "like"]
STOPWORDS.add("said")
STOPWORDS.add("course")
STOPWORDS.add("courses")
STOPWORDS.add("coursera")
STOPWORDS.add("really")
STOPWORDS.add("one")
text = ' '.join(filter(lambda x: x.lower() not in remove,  text.split()))
# read the mask image
# taken from
# http://www.stencilry.org/stencils/movies/alice%20in%20wonderland/255fk.jpg


# In[22]:

alice_mask = np.array(Image.open(path.join(d, "stormtrooper_mask.png")))

wc = WordCloud(background_color="white", max_words=2000, mask=alice_mask, 
               stopwords=STOPWORDS.add("said"))
# generate word cloud
wc.generate(text)


# store to file
wc.to_file(path.join(d, "storm.png"))

# show
plt.imshow(wc)
plt.axis("off")
# plt.figure()
# plt.imshow(alice_mask, cmap=plt.cm.gray)
# plt.axis("off")
plt.show()


# In[ ]:




# In[ ]:



