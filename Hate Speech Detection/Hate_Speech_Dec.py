import numpy as np
import pandas as pd


datasetTrain = pd.read_csv('train.csv')
datasetTest = pd.read_csv('test.csv')

X_trainD = datasetTrain.iloc[0:12000,2:3]
y_trainD = datasetTrain.iloc[0:12000,1]
#X_test = datasetTest.iloc[:,0:2]

#X_train['Train-Test'] = 1   
#X_test['Train-Test'] = 0   

#X = pd.concat([X_train,X_test], ignore_index= True)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []


for i in range(0,12000):
    tweet = re.sub('[^a-zA-Z]', ' ', str(X_trainD['tweet'][i]))
    tweet = tweet.lower()
    tweet = tweet.split()
    
    ps = PorterStemmer()
    
    #tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    temp = []
    
    all_stopwords = set(stopwords.words('english'))
    all_stopwords.add("u")
    all_stopwords.add("ur")
    all_stopwords.remove('not')
    
    for word in tweet:
        if not word in all_stopwords:
            temp.append(ps.stem(word))
        tweet = temp
            
    tweet = " ".join(tweet)
    corpus.append(tweet)       
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()  
X_bow = cv.fit_transform(corpus).toarray()

#len(X_trainD[0])


#train = X[X["Train-Test"] == 1 ]
#train = train.drop(columns = ['Train-Test'])
#test = X[X["Train-Test"] == 0 ]
#test = test.drop(columns = ['Train-Test'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_bow,y_trainD)


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

nb.score(X_test, y_test)
y_pred = nb.predict(X_test)



from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()
forest.fit(X_train, y_train)

forest.score(X_test, y_test)
y_pred2 = forest.predict(X_test)

#Confusion Matrix
import sklearn
sklearn.metrics.confusion_matrix(y_test,y_pred2)
