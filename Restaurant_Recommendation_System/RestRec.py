import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

class recommend:


    df = pd.read_csv('D:/WORKSPACE/DATASETS/1 KAGGLE/Restaurant Recommendation/Zomato Bangalore Restaurants/df_short.csv')
    
    def rec(res_name, short_df = df):
        #Tokenizing
        
        cv = CountVectorizer(max_features = 10000)
        vectorized = cv.fit_transform(short_df['corr_string']).toarray()
        
        #COSINE_SIMILARITY
        
        cos_s = cosine_similarity(vectorized)
        
        #Inserting name of restaurand and retrieving its ID
        name = res_name
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
        
        return df_output


