# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 09:36:22 2018

@author: aksha
"""

import nltk
import os
import html
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import operator
nltk.download('wordnet')
import pandas as pd 
import ast

os.getcwd()
os.chdir("C:\\Users\\aksha\\OneDrive\\Documents\\Purdue\\Fall Module - 2\\Unstructured Data analytics\\Project\\Code for text analytics")


df = pd.read_csv("Full_data.csv")


df.reset_index(inplace=True)
df=df.dropna()

    
#Data pre-Processing
templist = []
processeddescription  = []
pricelist = []
for i in range(0,len(df)):
    templist.append(ast.literal_eval(df.iloc[i][7])[0]['path'])
    processeddescription.append(df.iloc[i][2].replace('\r\n','').strip())
    if(df.iloc[i][3]!='None'):
        pricelist.append(int(df.iloc[i][3].split('$')[1]))
    else:
        pricelist.append(0)
        
        
df['Description'] = processeddescription
df['Price'] = pricelist
df['image_path'] = templist
df.drop(columns=['images', 'image_urls'],inplace = True)

df=df.dropna()
#Let's Preprocess and noramlize price column in whole 
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
column_names_to_normalize = ['Price']
x = df[column_names_to_normalize].values
x_scaled = min_max.fit_transform(x)
df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = df.index)
df[column_names_to_normalize] = df_temp

#Price are normalizaed at this point 



#Model 1 Text Model 
#let's do tokenization and stemming and check how the accuracy changes for both the approaches
#preprocessing the text.
TokenizedList = []
ProcessedTokenList = []
#POS Tagged Model
POS_c2 = []
for i in range(0,len(df)):
    POS_token_doc = nltk.pos_tag(nltk.word_tokenize(html.unescape(df.iloc[i][2])))
    POS_token_temp = ""
    counter = 0;
    for i in POS_token_doc:
        POS_token_temp=POS_token_temp+("".join(i[0]+i[1]))
        POS_token_temp = POS_token_temp+" ";
        counter = counter+1
    POS_c2.append(POS_token_temp.strip())
df['Description'] = POS_c2
#logit model - 53.51
#





ProcessedTokenList=[]
for i in range(0,len(df)):
    print(i)
#Step 1 code - Tokenization
    tokenizer = nltk.tokenize.WhitespaceTokenizer() 
    token_d2 =  nltk.word_tokenize(html.unescape(df.iloc[i][2]))
#Step 2 code - Lemmetization
    lemmatizer = nltk.stem.WordNetLemmatizer() 
    lemmatized_token_d2 = [lemmatizer.lemmatize(token) for token in token_d2 if token.isalpha()]  
#Step 3 code - removing punctuation and stop words
    Cleaned_String = [token for token in lemmatized_token_d2 if not token in stopwords.words('english') if token.isalpha()]
    temp = " ".join(str(x) for x in Cleaned_String)
    TokenizedList.append(token_d2)
    ProcessedTokenList.append(temp)
df['Description'] = ProcessedTokenList



#Some of the items cannot be classified properly because anything can be put into it
#So actually determine the heuristics of the model, we are removing some types of listings from the data.
df = df[df['Type']!='free stuff']
#df = df[df['Type']!='free stuff']
df.reset_index(inplace = True)


#Now, let's add the price into the array so that the model will learn about the price 
#while classification


df.dropna()

df.dropna(axis='columns',inplace=True)



from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2, random_state=0, stratify=df['Type'])
train = train.drop(columns=['level_0'])
test = test.drop(columns=['level_0'])
train = train.reset_index()
test = test.reset_index()

#Question 3 Train 3 models 
#Train a Naïve Bayes model, a Logit model, a Random Forest model, (with 10 trees), a SVM model.

vectorizer2 = TfidfVectorizer(analyzer='word',ngram_range=(1, 3),min_df = 3) 
vectorizer2.fit(train['Description'])
    

training_x = vectorizer2.transform(train['Description'])

training_y = train['Type']

test_x = vectorizer2.transform(test['Description'])
test_y = test['Type']



#Price into the array
train_price = pd.DataFrame(training_x.toarray())
train_price['Price'] = train['Price']

test_price=pd.DataFrame(test_x.toarray())
test_price['Price'] = test['Price']
training_x = train_price
test_x = test_price

#Logistic Model
from sklearn.linear_model import LogisticRegression
Logitmodel = LogisticRegression()
# training
Logitmodel.fit(training_x, training_y)


y_pred_logit = Logitmodel.predict(test_x)
# evaluation
from sklearn.metrics import accuracy_score
acc_logit = accuracy_score(test_y, y_pred_logit)
print("Logit model Accuracy:: {:.2f}%".format(acc_logit*100)) 



#NaiveBayes Model
from sklearn.naive_bayes import MultinomialNB
NBmodel = MultinomialNB()
# training
NBmodel.fit(training_x, training_y)
y_pred_NB = NBmodel.predict(test_x)
# evaluation
acc_NB = accuracy_score(test_y, y_pred_NB)
print("Naive Bayes model Accuracy:: {:.2f}%".format(acc_NB*100))


#Support Vector Machine
from sklearn.svm import LinearSVC
SVMmodel = LinearSVC()
# training
SVMmodel.fit(training_x, training_y)
y_pred_SVM = SVMmodel.predict(test_x)
# evaluation
acc_SVM = accuracy_score(test_y, y_pred_SVM)
print("SVM model Accuracy:{:.2f}%".format(acc_SVM*100))





#Random Forest 
from sklearn.ensemble import RandomForestClassifier
RFmodel = RandomForestClassifier(n_estimators=500, max_depth=100,bootstrap=True, random_state=0) ## number of trees and number of layers/depth
# training
RFmodel.fit(training_x, training_y)
y_pred_RF = RFmodel.predict(test_x)
# evaluation
acc_RF = accuracy_score(test_y, y_pred_RF)
print("Random Forest Model Accuracy: {:.2f}%".format(acc_RF*100))

#neural Network Model 

from sklearn.neural_network import MLPClassifier
DLmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(50), random_state=1)
# training
DLmodel.fit(training_x, training_y)
y_pred_DL= DLmodel.predict(test_x)  
# evaluation
acc_DL = accuracy_score(test_y, y_pred_DL)
print("Neural Network Model Accuracy: {:.2f}%".format(acc_DL*100))


#image modelling starts here 

#images are already there in the dataframe 

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 11:39:26 2018

@author: aksha
"""

import numpy as np 
import os 
import pandas as pd
#Set working directory
os.getcwd()
os.chdir("C:\\Users\\aksha\\OneDrive\\Documents\\Purdue\\Fall Module - 2\\Unstructured Data analytics\\Project\\scrapyDataextraction\\DataExtraction\\Car_Images\\full")
#Import necessary libraries
from PIL import Image
from pylab import *
#Initialize array and dataframe to save the flattened file
flattenedaggregatedimage = []
df_image = pd.DataFrame(columns = range(0,10000))

#Processing images
#Flatenning it
#greyscale
#reshape
#Normalization
normalized_flattenedaggregatedimage = []
for i in range(0,len(df)):
    print(i)
    im = Image.open(df.iloc[i][7].split('/')[1])
    crop = (75,100,180,150)
    im_crop = im.crop(crop)
    im_rs = im.resize((100,100)) #Crop the image
    im_gs = im_rs.convert('L')#Convert the omage to gray scale
    image_final = array(im_gs) # converted gray scale image to array
    im_v = image_final.flatten()#Make it into 1-D Array
    temptostore = (im_v.astype(int)).reshape(1,10000)
    df_temp = pd.DataFrame.from_records(temptostore)
    df_image = df_image.append(df_temp)    
    flattenedaggregatedimage = np.append(flattenedaggregatedimage,im_v)    
#adding the Y variable to the dataset
df_image['Type'] = df['Type']


from sklearn.model_selection import train_test_split
#Splitting the dataset into test and train
#We are doing stratified sampling on the Y column
train, test = train_test_split(df_image, test_size=0.2, random_state=0, stratify=df_image['Type'])
train = train.drop(columns=['level_0'])
test = test.drop(columns=['level_0'])
train = train.reset_index()
test = test.reset_index()

training_x = train[:-1]

training_y = train[-1]

test_x = train[:-1]
test_y = train[-1]

#Running logistic model 

#Logistic Model
from sklearn.linear_model import LogisticRegression
Logitmodel = LogisticRegression()
# training
Logitmodel.fit(training_x, training_y)


y_pred_logit = Logitmodel.predict(test_x)
# evaluation
from sklearn.metrics import accuracy_score
acc_logit = accuracy_score(test_y, y_pred_logit)
print("Logit model Accuracy:: {:.2f}%".format(acc_logit*100)) 

#Running naive Bayes model

#NaiveBayes Model
from sklearn.naive_bayes import MultinomialNB
NBmodel = MultinomialNB()
# training
NBmodel.fit(training_x, training_y)
y_pred_NB = NBmodel.predict(test_x)
# evaluation
acc_NB = accuracy_score(test_y, y_pred_NB)
print("Naive Bayes model Accuracy:: {:.2f}%".format(acc_NB*100))






