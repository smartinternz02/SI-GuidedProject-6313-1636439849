import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df=pd.read_csv('insurance.csv')

def convertSex(word):
    word_dict = {'male':0,'female':1}
    return word_dict[word]

def convertSmoker(word):
    word_dict = {'no':0,'yes':1}
    return word_dict[word]

def convertRegion(word):
    word_dict = {'southeast':2,'southwest':3,'northeast':0,'northwest':1}
    return word_dict[word]


df['sex'] = df['sex'].apply(lambda x : convertSex(x))
df['smoker'] = df['smoker'].apply(lambda x : convertSmoker(x))
df['region'] = df['region'].apply(lambda x : convertRegion(x))


x=df.iloc[:,:6].values
y=df.iloc[:,6:].values

from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=5,random_state=0)
rfr.fit(x,y)


pickle.dump(rfr, open('model.pkl','wb'))