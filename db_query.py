import requests
import time
import json
import datetime

# check db connection
def connect_db_check():
    url_test_connect = "http://172.21.45.8:8000/restaurant_api/test_connect"
    headers = {'Content-Type': 'application/json'}
    db2json_file = 'db2json.json'
    s = requests.session()
    print('test db conncection...')
    
    # connection test
    conn_result = False
    while conn_result == False:
        try:
            response = s.post(url = url_test_connect,
                              headers = headers,
                              data = json.dumps({}),
                              timeout = 1.5
                             )
            print(response.text)
            if "Success" in response.text:
                conn_result = True
        except Exception as e:
            time.sleep(5)
            print(e)
            continue


# download info table
def get_all_restaurant_info(cond = {"cond": ["user_reviews", "quote", "summary", "rating_date"]}):
    connect_db_check()
    url_get_all_restaurant_info = "http://172.21.45.8:8000/restaurant_api/get_all_restaurant_info" # all restaurant info
    headers = {'Content-Type': 'application/json'}
    db2json_file = 'db2json.json'
    s = requests.session()
    print('request database at {} with the cond: \n{}'.format(datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'),cond))
    response = s.post(
                        url = url_get_all_restaurant_info,
                        headers = headers,
                        data = json.dumps(cond)
                     )
    print('Finish request......')
    
    return json.loads(response.json())


# download stop-word table
def get_stop_words():
    connect_db_check()
    url_get_stopWord_file = 'http://172.21.45.8:8000/restaurant_api/get_stopWord_file'
    ### get StopWord file
    s = requests.session()
    headers = {'Content-Type': 'application/json'}
    response = s.post(
              url = url_get_stopWord_file,
              headers = headers,
              data = json.dumps({})
                     )
    print(json.loads(response.json()))


        
if __name__ == "__main__":
    # cond = {"cond": ["rating","quote","summary"]}
    # data = get_all_restaurant_info(cond)
    # print(data[:3])
    get_stop_words()

