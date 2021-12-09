# coding=utf-8
import pandas as pd
import numpy as np
import json
import jieba
from os.path import join, basename, dirname, exists
# from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import time
import pickle
import datetime
from preprocess import preprocess_query, tokenize, json2csv



class CONTENT_FILTER:
    
    def __init__(self):
        self.embed_filename = 'cf_embedding.csv'
        self.model_filename = 'tfidf.pkl'
        
    
    def check_embedding(self):
        update_enable = False
        if not exists(self.embed_filename):
            print('cannot find the local file cf_embedding.csv')
            print('build embedding from database......')
            update_enable = True
            json2csv()
            print('finish!')
        else:
            print('embedding already exists.')
            
        if not exists(self.model_filename) or update_enable:
            print('cannot find the model tfidf.pkl')
            self.fit_model()
        else:
            print('model already exists.')

    
    def update_embedding(self):
        print('update cf_embedding.csv......')
        print('build embedding from database......')
        json2csv()
        print('update the model......')
        self.fit_model()
        print('update finished!')
        
        
    # train
    def fit_model(self):
        print('fit model...')
        df = pd.read_csv(self.embed_filename, encoding = 'utf-8-sig')
        tfidf_model = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, max_df = 1.0,stop_words=["是","的","了"],token_pattern=r"(?u)\b\w+\b")
        tfidf_matrix = tfidf_model.fit_transform(df['description'])
        _info = {'model':tfidf_model , 'embed_matrix':tfidf_matrix}
        pickle.dump(_info, open(self.model_filename,"wb"))
        print('save model to tfidf.pkl!')
    
    
    
    # predict
    def inference(self, query = "牛肉麵" , top_n = 20):
        # load model and embedding
        self.check_embedding()
        _info = pickle.load(open(self.model_filename,'rb'))
        tfidf_model, tfidf_matrix = _info['model'], _info['embed_matrix']
        df = pd.read_csv(self.embed_filename, encoding = 'utf-8-sig')
        
        query_pre = preprocess_query(query)
        tfidf_matrix_input = tfidf_model.transform(pd.Series(query_pre))
        cosine_similarities = linear_kernel(tfidf_matrix_input, tfidf_matrix)
        similar_indices = cosine_similarities[0].argsort()[:-(top_n+1):-1] # top-n : the most similar
        similar_items = [(cosine_similarities[0][i], df['id'][i]) for i in similar_indices]
        
        res_list=[]
        print("Recommending top " + str(top_n) + " products based on query: " + query)
        print("="*160)
        for order, rec in enumerate(similar_items,1):
            _score, _id = rec[0], rec[1]
            _item_name = df.loc[df['id'] == _id]['name'].tolist()[0]
            _item_url = df.loc[df['id'] == _id]['url'].tolist()[0]
            _item_descr = df.loc[df['id'] == _id]['description_orig'].tolist()[0]
            res_list+=[(_id, _score, _item_url)]
            print("Recommended%d: "%order + _item_name + " (score:" + str(np.around(_score,6)) + ")" + " /description: " + _item_descr[:150] + '......')
            print("-"*160)
        
        
        return query, query_pre, res_list
    
        
if __name__ == "__main__":
    cf = CONTENT_FILTER()
    for itr in range(3):
        query = input("find your favorite restaurant! input query: ") # do query
        cf.inference(query,20)