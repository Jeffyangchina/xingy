#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
#@Author: Yang Xiaojun
import sqlite3
import csv
import re
import os
import numpy as np
import jieba
from sklearn.externals import joblib
# conn=sqlite3.connect(r'C:\Users\XUEJW\Desktop\兴业数据\分类用数据集\总集\Db_classify.db')
conn=sqlite3.connect(r'C:\Users\XUEJW\Desktop\兴业数据\xy_dataset\xy_zh_en.db')
def init(tb_name):
    sql_init1='drop table if EXISTS '+tb_name
    sql_init2='''create table '''+tb_name+''' (zh text PRIMARY KEY,en INT )'''
    cursor=conn.cursor()
    cursor.execute(sql_init1)
    cursor.execute(sql_init2)
    conn.commit()
def init_sp_tb():
    sql_init1 = 'drop table if EXISTS sampl_tb'
    sql_init2 = '''create table sampl_tb (src text,src_tb text,src_col text,tgt_tb text )'''
    cursor = conn.cursor()
    cursor.execute(sql_init1)
    cursor.execute(sql_init2)
    conn.commit()
    model_path = r'C:\Users\XUEJW\Desktop\兴业数据\分类用数据集\数据集\input dataset\dnn_model.csv'  # 内含所有的分词结果，作了初步优化，去单个词，及一些错误的词，在停用词表内stopword.txt，可以再优化
    csvFile2 = open(model_path, "r", encoding='UTF-8')
def out_en(astr):
    outstr = re.sub("[a-zA-Z]", "", astr)
    outstr=outstr.strip()
    return outstr
def get_en(astr):
    outstr = re.sub("[^a-zA-Z]", "", astr)
    outstr = outstr.strip().upper()
    return outstr
def rm_null(lis):
    for cel in lis:
        if len(cel)==0:
            lis.remove(cel)
def clean_input(instr):#only for new input
    r = '[’!"#$%&\－()（*+,-./:：”“。‘、，【】＋（）;<=>?@[\\]^_`{|}~]+'
    line = re.sub(r, '', instr)
    outstr = re.sub("[^\D]", "", line)
    outstr=out_en(outstr)
    return outstr
def put_in(filepath,tb_name):#file csv put into Db
    cursor=conn.cursor()
    # init(tb_name)
    csvFile = open(filepath, "r",encoding='UTF-8')
    reader = csv.reader(csvFile)  # 返回的是迭代类型
    rows1=[]
    rows2=[]
    rows3=[]
    i=2
    h=0
    for row in reader:#row is a list,row[0] is a str,rows1 is a list
        if row:
            cell=row[0].strip().upper()
            cel=out_num(cell)
            en=row[1].strip()
            sql = 'insert into '+tb_name+' VALUES(?,?) '
            try:
                cursor.execute(sql,(cel,en))

            except Exception as e :
                # print('err:',e)
                h+=1
    print('unique is {}'.format(h))
    csvFile.close()
    conn.commit()
    cursor.close()

def add_tb(tb_name,astr):
    cursor = conn.cursor()
    sql = 'insert into ' + tb_name + ' VALUES(?,?) '
    try:
        cursor.execute(sql, (astr, 1))
    except Exception as e:
        print('err:', e)

    conn.commit()
def rm_null(lis):
    for cel in lis:
        if len(cel)==0:
            lis.remove(cel)
    return lis
def out_num(instr):
    outstr = re.sub("[^\D]", "", instr)
    return outstr
def del_stop(alist):
    filepath = r'C:\Users\XUEJW\Desktop\兴业数据\分类用数据集\stopword.txt'
    def stopwordslist(filepath):
        stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
        return stopwords
    stopwords = stopwordslist(filepath)
    # print('st:',stopwords)
    tstr = []
    for word in alist:
        sw = word.strip().lower()
        if sw not in stopwords:
            if word != '\t':
                wword=word.strip().upper()
                wnstr=out_num(wword)
                tstr.append(wnstr)
    rm_null(tstr)
    return tstr#is a list
def like_fill(astr):#fenci create list只分词及一般处理
    import numpy as np
    import re
    r = '[’!"#$%&\－()（*+,-./:：”“。‘、，【】＋（）;<=>?@[\\]^_`{|}~]+'
    jfc_add = r'C:\Users\XUEJW\Desktop\兴业数据\分类用数据集\分词库\fenci.txt'
    if os.path.isfile(jfc_add):
        jieba.load_userdict(jfc_add)
    astr=out_num(astr)
    src_tb_word = re.sub(r, '', astr)
    cutlist = list(jieba.cut(src_tb_word))
    cutlist = del_stop(cutlist)

    return cutlist
def read_fr_db(tb_name,astr):#输入查询的表名和名称,返回一个np的矩阵，用了tolist最后是数组
    command='select cell from '+tb_name
    sql = command
    cursor = conn.cursor()
    res=cursor.execute(sql)
    tempx = res.fetchall()
    astr=astr.strip().upper()
    if tempx:
        lth=len(tempx)
        out_list=np.zeros(lth)
        cell_list = [x[0].strip().upper() for x in tempx]
        if astr=='':
            return out_list.tolist()
        if tb_name=='Db_src':
            astr=get_en(astr)
            if astr in cell_list:
                i = cell_list.index(astr)
                out_list[i] = 1.
        else:
            stlist=like_fill(astr)#input a str output a jieba list
            for st in stlist:
                if st in cell_list:
                    i = cell_list.index(st)
                    out_list[i] = 1.
    else:
        print('no such column')

    fi_list=out_list.tolist()
    return fi_list
def deal_new_samp(astr):#处理一个新的名称，产生一个分词后的数组，且字数大于2
    out_list=[]
    alist=like_fill(astr)
    for al in alist:
        if len(al)>1:
            out_list.append(al)
    return out_list
def add_new_samptb(source=None):#source should be a list[src,src_tb,src_col,tgt]用于新增的分类
    if source:
        if len(source)>4:
            for x in range(4):
                if x==0:
                    data=get_en(source[x])
                    add_tb('Db_src',data)
                if x==1:
                    data_list=deal_new_samp(source[x])
                    for dl in data_list:
                        add_tb('Db_tb',dl)
                if x==2:
                    data_list = deal_new_samp(source[x])
                    for dl in data_list:
                        add_tb('Db_col', dl)


def data_input(alist):#由输入的数组，根据已有的模型预测分类结果
    import numpy as np
    if len(alist)<3:
        print('You may input 3 strings as Source name,Source table name,Source table column')
        r = '[’!"#$%&\－()（*+,-./:：”“。‘、，【】＋（）;<=>?@[\\]^_`{|}~]+'
    try:
        src_word = get_en(alist[0])
        src_word = src_word.strip().upper()
        src_0=read_fr_db('Db_src',src_word)#is src list[0,0,..] 76x1
    except:
        src_0 = read_fr_db('Db_src', '')
        print('You may need Source Name')
    try:
        src_word = alist[1].strip()
        src_1 = read_fr_db('Db_tb',src_word)#is src_tb list[0,0,..] 566x1
    except:
        src_1= read_fr_db('Db_tb','')
        print('You may need Source Table Name')
    try:
        src_word = alist[2].strip()
        src_2 = read_fr_db('Db_col',src_word)
    except:
        src_2 =  read_fr_db('Db_col','')
        print('You may need Source Column Name')
    union = src_0+ src_1 + src_2
    return union
def like_data_input(alist):#由样本表建模
    import numpy as np
    if len(alist)==4:
        r = '[’!"#$%&\－()（*+,-./:：”“。‘、，【】＋（）;<=>?@[\\]^_`{|}~]+'
        try:
            src_word = get_en(alist[0])
            src_word = src_word.strip().upper()
            src_0=read_fr_db('Db_src',src_word)#is src list[0,0,..] 76x1
        except:
            src_0 = read_fr_db('Db_src', '')
        try:
            src_word = alist[1].strip()
            src_1 = read_fr_db('Db_tb',src_word)#is src_tb list[0,0,..] 566x1
        except:
            src_1= read_fr_db('Db_tb','')

        try:
            src_word = alist[2].strip()
            src_2 = read_fr_db('Db_col',src_word)# 2124
        except:
            src_2 =  read_fr_db('Db_col','')
        tgt_word = alist[3].strip().upper()

        if 'T00' in tgt_word  :
            tgt = [0]
        if 'T01' in tgt_word  :
            tgt = [1]
        if 'T02' in tgt_word  :
            tgt = [2]
        if 'T03' in tgt_word  :
            tgt = [3]
        if 'T04' in tgt_word  :
            tgt = [4]
        if 'T05' in tgt_word  :
            tgt = [5]
        if 'T06' in tgt_word  :
            tgt = [6]
        if 'T07' in tgt_word  :
            tgt = [7]
        if 'T08' in tgt_word  :
            tgt = [8]
        if 'T09' in tgt_word  :
            tgt = [9]
        if 'T10' in tgt_word  :
            tgt = [10]
        if 'T99' in tgt_word  :
            tgt = [11]
        if 'REF' in tgt_word  :
            tgt = [12]
        union = src_0+ src_1 + src_2+tgt
    return union
def samp_tb():
    command = 'select * from samp_tb'
    sql = command
    cursor = conn.cursor()
    res = cursor.execute(sql)
    tempx = res.fetchall()
    train_data=[]
    if tempx:
        for tx in tempx:#is a tuple

            train_data.append(like_data_input(list(tx)))
    print('0:',type(train_data),train_data[0])#是个列表中列表
    return train_data#列表中列表
def create_model(model_path):

    train_data=samp_tb()
    from sklearn.ensemble import RandomForestClassifier as Rfc
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    from sklearn.utils import shuffle
    shuf_data = shuffle(train_data)
    tr_x=[x[:-1] for x in shuf_data]
    # tr_x = shuf_data.iloc[:, :-1].values
    # train_y_ = shuf_data.iloc[:, -1:].values
    train_y_ = [[y[-1]] for y in shuf_data]
    print('2:',train_y_[0],type(train_y_[0]))
    ohe = OneHotEncoder()
    ohe.fit([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]])
    label_name = ['T00', 'T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09', 'T10', 'T99', 'REF']
    tr_y = ohe.transform(train_y_).toarray()#这个输入必须是ohe.fit里格式
    clf = Rfc(random_state=0)  # oob_score
    # model_path = r'C:\Users\XUEJW\Desktop\兴业数据\分类用数据集\classify model\yang_Rf_model.pkl'
    clf.fit(tr_x, tr_y)
    print('train score is {0}'.format(clf.score(tr_x, tr_y)))
    joblib.dump(clf, model_path)
def do_class(alist,model):
    union=[data_input(alist)]
    label_name = ['T00', 'T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09', 'T10', 'T99', 'REF']

    clf = joblib.load(model_path)
    pre_out=clf.predict(union)
    prob_out=clf.predict_proba(union)
    # print(pre_out)
    for item in pre_out:
        ind = np.where(item == 1)
    id=ind[0].tolist()
    if len(id)>0:
        print(label_name[id[0]])
        print('pro:',prob_out[id[0]][0][1],)
    else:
        print('no class')

# tb_name='Db_tgt'#每个属性对应一个表，表内有两列，第二列作为权重
# filepath=r"C:\Users\XUEJW\Desktop\兴业数据\分类用数据集\总集\out_one_word用于输入数据库\one_word_tgt_sum_vocab.csv"
# init(tb_name)
# put_in(r'E:\兴业银行\中文翻译英文最终数据集\合并产生字典库\from_tb_trans.csv','zh_en_0')
#先执行samp_tb()从样本表制作输入数据，create_model后生成模型,再执行do_class 可以进行输入的分类

model_path = r'E:\yang_xy_test\ten class\model\_model.pkl'
# create_model(model_path)
input=['CSK','机构表','地区代号']
# do_class(input,model_path)
conn.close()