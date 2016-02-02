
# coding: utf-8

# In[1]:

from vaderSentiment.vaderSentiment import sentiment as vaderSentiment 
from Tkinter import *


# In[2]:

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


# In[3]:

root = Tk()
root.title("Buttons")

app = Application(root)

root.mainloop()

