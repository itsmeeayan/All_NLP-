import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import tensorflow as tf


dataset = pd.read_csv('BankCustomers.csv')

dataset.info()

#PLOTTING

plt.plot(dataset['Balance'])

#Scatter Plot(Account Balance - Estimated Salary)
plt.scatter(dataset['Balance'],dataset['EstimatedSalary'])
plt.show()



plt.show()

#GROUPED DATA (Based on Age)

gd = dataset.groupby('Age')
gd.describe()

gd_max = gd.max()
gd_min = gd.min()
gd_mean = gd.mean()

#BarGraphs showing Age-Salary of each age group.
plt.bar(gd_max.index, gd_max['EstimatedSalary'])
plt.show()

plt.bar(gd_mean.index , gd_mean['EstimatedSalary'] )

plt.bar(gd_min.index , gd_min['EstimatedSalary'] )


#Data Pre-Processing

X = dataset.iloc[:,3:-1]
y = dataset.iloc[:,-1] 

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X.iloc[:,2] = lab.fit_transform(X.iloc[:,2:3])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder = 'passthrough' )
X  = np.array(ct.fit_transform(X))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

#Feature Scaling
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# *****Building the ANN******

#Initializing ANN
ann = tf.keras.models.Sequential()

#1st Hidden layer
ann.add(tf.keras.layers.Dense(units = 7, activation= 'relu'))

#2nd Hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation= 'relu'))

#Output Layer
ann.add(tf.keras.layers.Dense(units = 1, activation= 'sigmoid'))

# ****Training the ANN****
#Compile ANN
ann.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics = ['accuracy'])

#Training ANN
ann.fit(X_train, y_train, batch_size=32, epochs=30)

print(ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]])) > 0.5)

#






