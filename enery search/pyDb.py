#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
#@Author: Yang Xiaojun
import time
import pymongo
old_time=time.time()
# mongodb服务的地址和端口号
mongo_url = "127.0.0.1:27017"

# 连接到mongodb，如果参数不填，默认为“localhost:27017”
mongo_url='218.1.116.178:27017'
client = pymongo.MongoClient(mongo_url)

#连接到数据库myDatabase
DATABASE = "spider_maineng_energy"
db = client[DATABASE]

#连接到集合(表):myDatabase.myCollection
COLLECTION = "company_v6"
db_coll = db[COLLECTION ]

# # 在表myCollection中寻找date字段等于2017-08-29的记录，并将结果按照age从大到小排序
# queryArgs = {'date':'2017-08-29'}
# search_res = db_coll.find(queryArgs).sort('age',-1)
# for record in search_res:
#       print(f"_id = {record['_id']}, name = {record['name']}, age = {record['age']}")
queryArgs = {}# 用字典指定
projectionFields = {'company_name':True,'short_description':True, 'description':True}  # 用字典指定
searchRes = db_coll.find(queryArgs, projection = projectionFields)# 用字典指定
import csv
import etl
import timeit
new_path=r"E:\能搜\fr_mangodb.csv"
csvFile3 = open(new_path, 'w', newline='',encoding='UTF-8')  # 设置newline，否则两行之间会空一行
writer3 = csv.writer(csvFile3)
def clean_data(adb):
    xt = list(adb.values())[0]
    if xt:
        xt=etl.cut_str(xt)
        # if 'engineering' in xt:
        #     print('11:', xt)
        if len(xt)>0:
            xt=' '.join(xt)+' .'
        else:
            xt=''
    else:
        xt=''
    return xt
def fr_db():#用于从mangodb提取数据

    for xx in searchRes:
        if xx:
            # print('0:', xx['company_name'],xx['description'],xx['short_description'])
            # company_name=clean_data(xx['company_name'])
            if xx['description']:
                company_des=clean_data(xx['description'])
            else:
                company_des=''
            if xx['short_description']:
                company_short_des=clean_data(xx['short_description'])
            else:
                company_short_des=''

            if len(company_des)>0 or len(company_short_des)>0:
                writer3.writerow([xx['company_name'],company_des,company_short_des])
        # item=xx['short_description']
        # xt=list(item.values())[0]
        # xt=etl.cut_str(xt)
        # xt=' '.join(xt)+' .'
        # print('0:',xx['description'],xx['short_description'])

new_time=time.time()
print('time:',new_time-old_time)
a=timeit.timeit(fr_db)
print(a)
csvFile3.close()