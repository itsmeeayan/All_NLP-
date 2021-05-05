import numpy as np
import pandas as pd


datasetF = pd.read_csv('Fake1.csv')
datasetR = pd.read_csv('True1.csv')



datasetR['real_or_fake'] = 1    
datasetF['real_or_fake'] = 0   
combd = pd.concat([datasetR, datasetF],ignore_index=True)

#Fake = combd[combd["fake"] == 0 ]
#real = combd[combd["real"] == 1 ]


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0,979):
   
   title = re.sub('[^a-zA-Z]',' ', str(combd['title'][i] ) )         #IMP
   text = re.sub('[^a-zA-Z]',' ', str(combd['text'][i] ) )

   title = title.lower()
   text = text.lower()
   
   title = title.split()
   text = text.split()
   
   ps = PorterStemmer()
   
   all_stopwords = stopwords.words('english')
   all_stopwords.remove('not')
   
   title = [ps.stem(word) for word in title if not word in set(all_stopwords)]       # Stemming
   title = ' '.join(title)
   
          
   text = [ps.stem(word) for word in text if not word in set(all_stopwords)]         # Stemming
   text = ' '.join(text)
   
   conc = []
   conc.append(title)
   conc.append(text)
   conc = ' '.join(conc)
   corpus.append(conc)
   
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=10000)  
X = cv.fit_transform(corpus).toarray()
#X.merge(X,)

y = combd.iloc[:,-1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.1)


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

nb.score(X_test, y_test)
y_pred = nb.predict(X_test)
