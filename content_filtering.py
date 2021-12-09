# coding=utf-8
import pandas as pd
import numpy as np
import json
import jieba
from os.path import join, basename, dirname, exists
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import time
import datetime
from preprocess import preprocess_query, tokenize, json2csv



def recommend_restaurants(input_str="牛肉麵", top_n=10):
    
    def get_item_name(id):
        return df.loc[df['id'] == id]['name'].tolist()[0]
    
    def get_item_descr(id):
        return df.loc[df['id'] == id]['description_orig'].tolist()[0]
        # return df.loc[df['id'] == id]['description'].tolist()[0]
        
    def get_item_url(id):
        return df.loc[df['id'] == id]['url'].tolist()[0]
    
    
    def recommend(input_str, top_n):
        res_list=[]
        print("Recommending top " + str(top_n) + " products based on query: " + input_str)
        print("="*160)
        for order, rec in enumerate(similar_items,1):
            res_list+=[(rec[1], rec[0], get_item_url(rec[1]))] # element: (id,score,url)
            print("Recommended%d: "%order + get_item_name(rec[1]) + " (score:" + str(np.around(rec[0],6)) + ")" + " /description: " + get_item_descr(rec[1]))
            print("-"*160)
        
        return res_list
    
    if not exists('restaurant_embedding.csv'):
        print('cannot find the local file restaurant_embedding.csv!')
        print('query database......')
        json2csv()
    
    df = pd.read_csv('restaurant_embedding.csv',encoding = 'utf-8-sig')
    
    # tf = TfidfVectorizer(analyzer='word', max_features=200 , ngram_range=(1, 1), min_df=0, max_df = 0.6,stop_words=["是","的","了"],token_pattern=r"(?u)\b\w+\b").fit(df['description'])
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, max_df = 1.0,stop_words=["是","的","了"],token_pattern=r"(?u)\b\w+\b").fit(df['description'])
    
    tfidf_matrix = tf.fit_transform(df['description'])
    
    input_str_pre = preprocess_query(input_str)
    time_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_data')
    df_input = pd.DataFrame({'datetime':[time_now],'description':[input_str], 'description_preprocess':[input_str_pre]})
    # print('the query info:\n',df_input)
    # print('\n')
    
    
    tfidf_matrix_input = tf.transform(df_input['description_preprocess'])
    cosine_similarities = linear_kernel(tfidf_matrix_input, tfidf_matrix)

    
    similar_indices = cosine_similarities[0].argsort()[:-(top_n+1):-1] # top-n : the most similar
    similar_items = [(cosine_similarities[0][i], df['id'][i]) for i in similar_indices]
    
    
    return input_str, input_str_pre, recommend(input_str = input_str, top_n=top_n)
    




if __name__ == "__main__":
    input_str = input("find your favorite restaurant! input query:  ") # do query: e.g. 牛肉麵 or 燒肉烤肉CP值高 or 火鍋吃到飽 or 便宜水餃店 or 壽司...
    query, query_pre, res_list = recommend_restaurants(input_str = input_str, top_n = 20)
    print("-"*80,"Summary","-"*80)
    print("The input query: {0}\nThe preprocessed query: {1}".format(query, query_pre))
    print(f"the recommended restaurants from top-1 to top-{len(res_list)}:\n\n",res_list)
    print("-"*80,"End","-"*80)
    