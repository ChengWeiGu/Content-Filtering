# coding=utf-8
import pandas as pd
import numpy as np
import requests
import json
import jieba
from os import listdir
from os.path import join, basename, dirname, exists
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import time
import datetime
from db_query import get_all_restaurant_info



def read_data():
    data = []
    with open('restaurant.json', 'r',encoding="utf-8") as f:
        data = json.load(f)
    return data
    

def generate_stop_words():
    stop_word_list = []
    with open("stopwords1893_cn.txt",'r', encoding = 'utf-8-sig') as f:
        stop_word_list = [line.strip('\n') for line in f.readlines()]
        # print(stop_word_list)
        f.close()
    return stop_word_list


def tokenize(text, stop_word_list):
    cut_words = list(jieba.cut(text))
    new_cut_words = []
    for word in cut_words:
        if word not in stop_word_list:
            new_cut_words.append(word)
        else:
            pass
            
    return new_cut_words



def preprocess_query(input_str):
    stop_word_list=generate_stop_words()
    sent_words = tokenize(input_str,stop_word_list)
    document = " ".join(sent_words)
    
    return document
    


def json2csv():
    
    # from db query
    data = get_all_restaurant_info(cond = {"cond": ["rating","quote","summary"]}) 
    stop_word_list=generate_stop_words()
    
    id_list = []
    url_list = []
    name_list = []
    description_orig_list = []
    description_final_list = []
    
    total_doc = len(data)
    count = 0
    for _id, restaurant_data in enumerate(data,1):
        count += 1
        print(f'progress:{count}/{total_doc}={np.around(count/total_doc*100,2)}%')
        
        id_list += [restaurant_data['restaurant_id']]
        url_list += [restaurant_data['restaurant_url']]
        name_list += [restaurant_data['restaurant_name']]
        str1 = restaurant_data['restaurant_name'] + '，' #21工房天然手工涼麵（酒泉店，
        str2 = ""
        criteria = 20
        for review in restaurant_data['reviews']:
            user_quote_len = len(review['quote'])
            user_rating = int(review['rating']) if review['rating'] != '' else -1 # sometimes, user review is empty
            if user_quote_len < criteria and user_rating >= 0:
                if user_quote_len < 5 and user_rating >= 40: #refer to the summary instead of quote
                    str2 += (review['summary'] + '，')
                elif user_quote_len < 5 and user_rating <= 20: #does not include this
                    pass
                else:
                    str2 += (review['quote'] + '，') # 老字號西餐館, CP值超低, 好吃...     
            else:
                pass
                    
        
        str3 = ' '.join(str(e) for e in restaurant_data['characteristics']) + '，' #菜系 日式料理 亞洲料理 餐點 午餐, 晚餐，
        text = str1+str2+str3
        description_orig_list += [text]
        sent_words = tokenize(text,stop_word_list)
        document = " ".join(sent_words)
        description_final_list += [document]
        
        
        
    pd.DataFrame({'id':id_list, 'name':name_list, 'url':url_list, 'description_orig':description_orig_list, 'description':description_final_list}).to_csv('cf_embedding.csv',index = False, encoding='utf-8-sig')    



if __name__ == "__main__":
    json2csv()
    