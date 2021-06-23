# coding=utf-8
import pandas as pd
import numpy as np
import json
import jieba
from os import listdir
from os.path import join, basename, dirname
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import time
import datetime


def read_data():
    data = []
    with open('tripadvisor_restaurant_infos.json', 'r',encoding="utf-8") as f:
        data = json.load(f)
    return data
    

def file2csv():

    data = read_data()
    
    id_list = []
    name_list = []
    description_orig_list = []
    description_list = []
    
    total_doc = len(data)
    
    count = 0
    for _id, restaurant_data in enumerate(data,1):
        count += 1
        print(f'progress:{count}/{total_doc}={np.around(count/total_doc*100,2)}%')
        
        id_list += [_id]
        name_list += [restaurant_data['name']]
        
        str1 = restaurant_data['name'] + '，' #21工房天然手工涼麵（酒泉店，
        str2 = restaurant_data['location'] + '，' #103 臺灣 大同臺北 酒泉街10巷25號，
        str3 = "評分 "+restaurant_data['rating']['overall']+ '，' #評分4.0，
        str4 = ' '.join(str(e) for e in restaurant_data['characteristics']) + '，' #菜系 日式料理 亞洲料理 餐點 午餐, 晚餐，
        text = str1+str2+str3+str4
        description_orig_list += [text]
        
        sent_words = list(jieba.cut(text))
        document = " ".join(sent_words)
        description_list += [document]
        
        
        
    pd.DataFrame({'id':id_list, 'name':name_list, 'description_orig':description_orig_list, 'description':description_list}).to_csv('tripadvisor_data.csv',index = False, encoding='utf-8-sig')


# use self-self similarity to rank restaurants: index searching
def main_self_rank():
    
    def get_item_name(id):
        return df.loc[df['id'] == id]['name'].tolist()[0]
    
    def get_item_descr(id):
        return df.loc[df['id'] == id]['description_orig'].tolist()[0]
    
    # Just reads the results out of the dictionary.
    def recommend(item_id, num):
        
        print("Recommending " + str(num) + " products similar to " + get_item_name(item_id) + "... /description: ", get_item_descr(item_id))
        print("="*160)
        recs = results[item_id][:num]
        for order, rec in enumerate(recs,1):
            print("Recommended%d: "%order + get_item_name(rec[1]) + " (score:" + str(np.around(rec[0],6)) + ")" + " /description: " + get_item_descr(rec[1]))
            print("-"*160)
    
    # file2csv()
    
    df = pd.read_csv('tripadvisor_data.csv',encoding = 'utf-8-sig')
    
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0, max_df = 0.9,stop_words=["是","的"],token_pattern=r"(?u)\b\w+\b").fit(df['description']) #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    tfidf_matrix = tf.transform(df['description'])
    # print(tf.vocabulary_) # print the feature indices
    
    
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    # print(cosine_similarities)
    
    
    results = {}

    for idx, row in df.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1] # top-100 : the most similar
        similar_items = [(cosine_similarities[idx][i], df['id'][i]) for i in similar_indices] # [(0.220379, 19), (0.169389, 494),(0.167694, 18),...]
        results[row['id']] = similar_items[1:] # exclude itself
        

    recommend(item_id=11, num=10)
    print('done!')




def chinese_preprocess(input_str):
    
    sent_words = list(jieba.cut(input_str))
    document = " ".join(sent_words)
    
    return document
        


# use input string to rank restaurant: query searching
def main_input_rank():
    
    def get_item_name(id):
        return df.loc[df['id'] == id]['name'].tolist()[0]
    
    def get_item_descr(id):
        return df.loc[df['id'] == id]['description_orig'].tolist()[0]
    
    
    def recommend(input_str, q_count, num):
        
        print("Recommending " + str(num) + " products based on query: " + input_str)
        print("="*160)
        recs = results[q_count][:num]
        for order, rec in enumerate(recs,1):
            print("Recommended%d: "%order + get_item_name(rec[1]) + " (score:" + str(np.around(rec[0],6)) + ")" + " /description: " + get_item_descr(rec[1]))
            print("-"*160)
    
    # file2csv()
    
    df = pd.read_csv('tripadvisor_data.csv',encoding = 'utf-8-sig')
    
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0, max_df = 1.0,stop_words=["是","的"],token_pattern=r"(?u)\b\w+\b").fit(df['description']) #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    
    tfidf_matrix = tf.fit_transform(df['description'])
    # print(tf.vocabulary_) # print the feature indices
    
    
    results = {}
    query_count = 0
    
    
    input_str = input("find your favorite restaurant! or enter \"e\" to exit:  ")
    input_str_pre = chinese_preprocess(input_str)
    
    time_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_data')
    df_input = pd.DataFrame({'q_count':[query_count],'datetime':[time_now],'description':[input_str], 'description_preprocess':[input_str_pre]})
    print('the query info:\n',df_input)
    print('\n')
    
    
    tfidf_matrix_input = tf.transform(df_input['description_preprocess'])
    cosine_similarities = linear_kernel(tfidf_matrix_input, tfidf_matrix)

    
    similar_indices = cosine_similarities[0].argsort()[:-100:-1] # top-100 : the most similar
    similar_items = [(cosine_similarities[0][i], df['id'][i]) for i in similar_indices]
    results[query_count] = similar_items
    
    
    recommend(input_str = input_str, q_count = query_count, num=10)
    print('done!')
    


if __name__ == "__main__":
    # main_self_rank()
    main_input_rank()
    
