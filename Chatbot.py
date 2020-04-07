# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:10:55 2020

@author: Shinu.Valappila
"""

# import packages
from newspaper import Article
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
import warnings

# ignore warning messages
warnings.filterwarnings('ignore')

# download packages from NLTK
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Get article URL
article = Article(
    'https://www.weforum.org/agenda/2020/03/covid-19-explained-virology-expert/')

# download the article and parse it
article.download()
article.parse()
article.nlp()
corpus = article.text

#print(corpus)

# Tokenization
text = corpus
sent_tokens = nltk.sent_tokenize(text) # convert text into list of sentences

# print the list of sentences
#print(sent_tokens)

# create a dictionary (key:value) pair to remove punctuations
remove_punct_dict = dict( (ord(punct) ,None) for punct in string.punctuation)

# print punctuations
#print(string.punctuation)
# print dictionary 
#print(remove_punct_dict)

# create a function to return a list of lenmatized lower case words after removing punctuations

def LenNormalize(text):
    return nltk.word_tokenize(text.lower().translate(remove_punct_dict))

# print tokenization text
#print(LenNormalize(text))

# keyword matching
# list of possible greeting inputs by user
GREETING_INPUTS = ["hi", "hello", "hola", "greetings", "wassup", "hey"]
# list of greeting responses
GREETING_RESPONSES = ["hi", "hello", "hola", "howdy", "hey", "what's good", "hey there"]

# function to generate random greeting response to a user's greeting
def greeting(sentence):
    # if the user's input is a greeting, then return a randomly chosen greeting response
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# generate response
def response(user_response):

    # setup user response/query and change it to lower case
    #user_response = 'what is Acute respiratory distress syndrome'
    user_response = user_response.lower()
    #print(user_response)

    # set chatbot response to an empty string
    bot_response = ''

    # append user's response to the sentence list (sent_tokens)
    sent_tokens.append(user_response)
    #print(sent_tokens)
    # create a TfidfVectorizer object
    TfidfVec = TfidfVectorizer(tokenizer = LenNormalize, stop_words='english')

    # convert text to matrix of TF-IDF features
    tfidf = TfidfVec.fit_transform(sent_tokens)
    #print(tfidf)

    # get the measure of similarity (similarity scores 0-1)
    vals = cosine_similarity(tfidf[-1], tfidf)
    #print(vals)

    # get the index of the most similar text/sentence to the user response (remember vals lists of lists)
    idx = vals.argsort()[0][-2]

    # dimentionality reductions of vals (make it one list)
    flat = vals.flatten()

    # sort the list in ascending order
    flat.sort()

    # get the most similar score of the users repsonse
    score = flat[-2] # -1 is user response; we want the one prior to it
    #print(score)

    # if the variable score is 0 then there is no text similar to user's response
    if (score == 0):
        bot_response = bot_response+ "I apologize, I dont understand."
    else:
        bot_response = bot_response+ sent_tokens[idx]     

    # print chatbot response
    #print(bot_response)

    # remove user response
    sent_tokens.remove(user_response)
    # return bot response
    return bot_response


flag = True
print("DOCBot: I am Doctor Bot or DOCBot for short. I will answer your queries regarding COVID Virology. If you want to quit, type Bye!")

while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'bye'):
        if(user_response == "thanks" or user_response == "thank you"):
            flag = False
            print("DOCBot: You are most welcome !")
        else:
            if(greeting(user_response) != None):
                print("DOCBOT: "+ greeting(user_response))
            else:
                print("DOCBot: "+ response(user_response))
    else:
        flag = False
        print("DOCBOT : Thanks for your time. Chat with you later !")