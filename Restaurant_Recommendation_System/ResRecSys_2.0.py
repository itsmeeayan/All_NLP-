import pandas as pd
import numpy as np

df = pd.read_csv('zomato.csv')

#df = pd.read_csv('zomato_all.csv')


df = df.drop(columns = ['url', 'address', 'location', 'phone','listed_in(city)' ])
df.info()
df.isnull().sum()

df.head()

#MAKING a dataset with string values of features from original dataset

#df['dish_liked']
df_string = df['rest_type']+ ' '  + df['cuisines'] + ' ' + df['approx_cost(for two people)'] + ' '  + df['menu_item'] + ' '  + df['listed_in(type)']
df_string = pd.DataFrame(df_string)

#Adding the Name column
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
import re
def CleanText(sentence):
    sentence = re.sub('\\[]', '', str(sentence))
    sentence = re.sub(',', '', str(sentence))
    return sentence


df_string['corr_string'] = df_string['corr_string'].apply(CleanText)

short_df = df_string[:10000]
#_______Making an ID column from index
short_df['id'] = short_df.index

#Tokenizing
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 10000)
vectorized = cv.fit_transform(short_df['corr_string']).toarray()

words = cv.get_feature_names()

#COSINE_SIMILARITY
from sklearn.metrics.pairwise import cosine_similarity
cos_s = cosine_similarity(vectorized)

#____________
#cos_s[:10]
#_________________


#Inserting name of restaurand and retrieving its ID
name = "Timepass Dinner"
rest_id = short_df[short_df.Name == name]['id'].values[0]


#Creating list of enumaration for similarity score

score = list(enumerate(cos_s[rest_id]))

#Sort List
sorted_score = sorted(score, key = lambda x:x[1], reverse = True)
sorted_score = sorted_score[1:]

#Output The name of the Restaurants
name_list = []
rest_type_list = []
cuisines_list = []
approx_cost_list = []
menu_item_list = []
listed_in_list = []


counter = 0
for r_id in sorted_score:
    rest = short_df[ r_id[0] == short_df.id]['Name'].values[0]
    name_list.append(rest)
    
    rest = short_df[ r_id[0] == short_df.id]['rest_type'].values[0]
    rest_type_list.append(rest)
    
    rest = short_df[ r_id[0] == short_df.id]['cuisines'].values[0]
    cuisines_list.append(rest)
    
    rest = short_df[ r_id[0] == short_df.id]['approx_cost(for two people)'].values[0]
    approx_cost_list.append(rest)
    
    rest = short_df[ r_id[0] == short_df.id]['menu_item'].values[0]
    menu_item_list.append(rest)
    
    rest = short_df[ r_id[0] == short_df.id]['listed_in(type)'].values[0]
    listed_in_list.append(rest)
    
    if counter >= 9:
        break
    counter = counter + 1


#creating a df
df_output = pd.DataFrame({'Name': name_list,
                          'rest_type': rest_type_list,
                          'cuisines': cuisines_list,
                          'approx_cost(for two)': approx_cost_list,
                          'menu_item': menu_item_list,
                          'listed_in': listed_in_list})




#_________________________________________________________________________________________________________________
###
short_df.to_csv('D:/WORKSPACE/DATASETS/1 KAGGLE/Restaurant Recommendation/Zomato Bangalore Restaurants/df_short.csv')
#____________________________________________________________________________________________________________________

















