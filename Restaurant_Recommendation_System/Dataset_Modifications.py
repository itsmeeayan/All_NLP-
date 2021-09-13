import pandas as pd
import numpy as np
import re

df = pd.read_csv('zomato.csv')

#Dropping irrelevant columns
df = df.drop(columns = ['url', 'address', 'location', 'phone','listed_in(city)' ])
df.info()
df.isnull().sum()

df.head()

#MAKING a dataset with string values of features from original dataset

#df['dish_liked']
df_string = df['rest_type']+ ' '  + df['cuisines'] + ' ' + df['approx_cost(for two people)'] + ' '  + df['menu_item'] + ' '  + df['listed_in(type)']
df_string = pd.DataFrame(df_string)

#Including rest of the columns
df_string['Name'] = df['name']
df_string.columns = ['corr_string', 'Name']
df_string['rest_type'] = df['rest_type']
df_string['cuisines'] = df['cuisines']
df_string['approx_cost(for two people)'] = df['approx_cost(for two people)']
df_string['menu_item'] = df['menu_item']
df_string['listed_in(type)'] = df['listed_in(type)']



df_string.isnull().sum()

#Drop Null values
df_string = df_string.dropna()

#Cleaning text

def CleanText(sentence):
    sentence = re.sub('\\[]', '', str(sentence))
    sentence = re.sub(',', '', str(sentence))
    return sentence


df_string['corr_string'] = df_string['corr_string'].apply(CleanText)

#Shortening the dataset (as per as the processing capacity of the system)
short_df = df_string[:10000]
#_______Making an ID column from index
short_df['id'] = short_df.index


#_________________________________________________________________________________________________________________
###Saving the newly augmented dataset short_df in csv format
short_df.to_csv('D:/WORKSPACE/DATASETS/1 KAGGLE/Restaurant Recommendation/Zomato Bangalore Restaurants/df_short.csv')
#____________________________________________________________________________________________________________________

















