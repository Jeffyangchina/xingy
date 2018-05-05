#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
#@Author: Yang Xiaojun
import sqlite3
import csv
import re
import jieba
import warnings
import time
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
oldtime = time.clock()
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
conn=sqlite3.connect(r'C:\Users\XUEJW\Desktop\兴业数据\xy_dataset\xy_zh_en.db')
#zh_en,zh_en_prior1,zh_en_prior2
cursor=conn.cursor()
try:
    model = Word2Vec.load(
        r'C:\Users\XUEJW\Desktop\兴业数据\分类用数据集\word2vec\word2vec\10G训练好的词向量\60维\Word60.model')

except Exception as e1:
    print('Use 122g embeddings')
    try:
        model = KeyedVectors.load_word2vec_format(
            r'C:\Users\XUEJW\Desktop\兴业数据\分类用数据集\word2vec\word2vec\中文的词向量\news_12g_baidubaike_20g_novel_90g_embedding_64.bin',
            binary=True)

    except Exception as e:
        print('err1:', e)
filepath=r'C:\Users\XUEJW\Desktop\兴业数据\xy_dataset\中对英\db\stopword.txt'
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
def rm_null(lis):
    for cel in lis:
        if len(cel)==0:
            lis.remove(cel)
def out_num(instr):
    outstr = re.sub("[^\D]", "", instr)#去除数字
    return outstr
def find_from_0(instr):
    sql = 'select en  from zh_en where (zh like ?) '
    input_data=out_num(instr).strip()
    res = cursor.execute(sql, (input_data,))
    # res = cursor.execute(sql, (wx,))
    tempx = res.fetchall()
    if len(tempx)>0:
        # print('0:',tempx,tempx[0][0],type(tempx[0][0]))
        return tempx[0][0]#is a str
    else:

        return False
def find_from_dict(instr):
    sql = 'select en  from zh_en_prior2 where (zh like ?) '
    input_data = out_num(instr).strip()
    res = cursor.execute(sql, (input_data,))
    tempx = res.fetchall()
    if len(tempx) > 0:
        # print('0:',tempx,tempx[0][0],type(tempx[0][0]))
        return tempx[0][0]#is a str
    else:
        return False

def find_from_1(instr):
    sql = 'select en  from zh_en_prior1 where (zh like ?) '
    input_data = out_num(instr).strip().upper()
    res = cursor.execute(sql, (input_data,))
    tempx = res.fetchall()
    if len(tempx) > 0:
        # print('0:',tempx,tempx[0][0],type(tempx[0][0]))
        return tempx[0][0]#is a str
    else:
        return False
def normalize(name):
    return name.capitalize()
def concat(lis):
    en_out = list(map(normalize, lis))
    en_result = '_'.join(en_out)
    return en_result
def dele_stop_word(alist):
    stopwords = stopwordslist(filepath)
    # print('st:',stopwords)
    tstr = []
    for word in alist:
        sw=word.strip().lower()
        if sw not in stopwords:
            if word != '\t':
                tstr.append(word.strip())
    rm_null(tstr)
    # print('y:',tstr)
    return tstr
def deal_data(instr):#分词
    jieba.load_userdict(r"C:\Users\XUEJW\Desktop\兴业数据\xy_dataset\中对英\训练集\补充.txt")
    r = '[’!"#$%&\'()*+,-./:：、。，（）;<=>?@[\\]^_`{|}~]+'
    data = []
    line = re.sub(r, '', instr)#去除符号
    outstr = re.sub("[^\D]", "", line)
    item=list(jieba.cut(outstr.strip()))
    rm_null(item)
    # data=dele_stop_word(item)

    return item

def find_word(instr):#no jieba because already done
     #判断是否是全字母：instr
    temp_instr=instr
    temp_instr1=instr
    result = re.sub(r'[A-Za-z]', '',temp_instr )
    result1 = re.sub(r'[0-9]', '', temp_instr1)
    if len(result)==0 or len(result1)==0:
        return instr
    if find_from_0(instr):#第一个表没有，找字典表，再没有找最后一个表
        return find_from_0(instr)
    else:
        if find_from_dict(instr):
            return find_from_dict(instr)
        else:
            if find_from_1(instr):
                return find_from_1(instr)
            else:
                # return find_from_w2v(instr)
                return ''

def deep_data(instr):
    r = '[’!"#$%&\－()（*+,-./:：”“。‘、，【】＋（）;<=>?@[\\]^_`{|}~]+'
    data = []
    line = re.sub(r, '', instr)  # 去除符号
    outstr = re.sub("[^\D]", "", line)
    item = list(jieba.cut(outstr.strip(),cut_all=True))
    rm_null(item)
    # data=dele_stop_word(item)

    return item
def rm_duplicate(alist):
    out_data=[]
    for item in alist:
        if item.strip() not in out_data:
            out_data.append(item.strip())
        # print('3:',out_data)
    rm_null(out_data)
    return out_data
def find_from_w2v(instr,model):
    index = model.wv.index2word
    out_data=[]
    if instr in index:
        rela_word = model.most_similar(instr,topn=3)#is a list
        for xw in rela_word:
            out_data.append(xw[0])
        return out_data
def single_find(instr,model):
    out_str=[]
    for char in instr:
        if find_word(char) and find_word(char) not in out_str:
            out_str.append(find_word(char).capitalize())
        else:
            rela_char=find_from_w2v(char,model)
            if rela_char:
                if len(rela_char) > 0:
                    data_out = ''
                    for xx in rela_char:
                        if find_word(xx) and find_word(xx) not in data_out:
                            data_out = find_word(xx)
                            return data_out
                    if data_out == '':
                        for xx in rela_char:
                            data_out = single_find(xx, model)
                            if len(data_out) > 0:
                                return data_out
            return ''
    return ('_'.join(out_str))
def from_w2c_next(inlist,model):
    if inlist:
        if len(inlist)>0:
            data_out=''
            for xx in inlist:
                xfw=find_word(xx)
                if xfw and xfw not in data_out:
                    data_out=find_word(xx)
                    return data_out
        # else:
        #     return ''

            if data_out=='':
                for xx in inlist:
                    data_out=single_find(xx,model)
                    if len(data_out)>0:
                        return data_out

    return ''
def move_char(alist,i):
    for n in range(i):
        lth=99
        xlh=''
        for x in alist:
            if len(x)<lth:
                lth=len(x)
                xlh=x
        alist.remove(xlh)
    return alist
def create_shor(alist):#缩短单词
    relist=[]
    for xx in alist:
        ind=alist.index(xx)
        if len(xx) > 4:
            alist[ind] = alist[ind][0:3]
    #     if len(xx) < 3:
    #         relist.append(xx)
    # for it in relist:
    #
    #     alist.remove(it)
    return alist
def get_shorter(astr):#去重复单词
    i=0
    nlist=list(astr.split('_'))
    alist=[]
    for xa in nlist:
        if xa not in alist:
            alist.append(xa)
    tpstr = '_'.join(alist)
    # print('quchong0:', tpstr)
    alist=create_shor(alist)
    nwstr = '_'.join(alist)

    # print('quchong:',nwstr)
    while len(nwstr)>30:
        i+=1
        alist = list(tpstr.split('_'))
        alist=move_char(alist,i)
        # print('mc:',alist,i)
        alist = create_shor(alist)
        # print('cs:', alist)
        nwstr = '_'.join(alist)
        # print('yang:',nwstr,i)
        if len(nwstr)<30:
            break
            # return nwstr

    return nwstr
def dsc(astr):#make sentence shorter
    alist=list(astr.split('_'))
    alist=dele_stop_word(alist)
    sql = 'select jc  from zh_en where (zh like ?) '
    for i in range(len(alist)):
        alist[i]=alist[i].capitalize()
        res = cursor.execute(sql, (alist[i],))
    # res = cursor.execute(sql, (wx,))
        tempx = res.fetchall()
        if len(tempx) > 0:
            alist[i]=tempx[0][0]
        # print('0:',tempx,tempx[0][0],type(tempx[0][0]))
    alist=rm_duplicate(alist)
    nstrn='_'.join(alist)
    # print('nrt:',nstrn)
    if len(nstrn)<30:
        return nstrn
    else:
        nstrx=get_shorter(nstrn)
        nst = list(nstrx.split('_'))
        nlst=rm_duplicate(nst)
        str_out='_'.join(nlst)
        # print('yangn:',nstrx)
        return str_out
def do_trans(instr,model=None):
    instr=out_num(instr.strip())

    result=[]
    if len(instr)>0:
        if find_from_0(instr):
            return find_from_0(instr)#先去表1找
        else:#找不到了，分词
            str_list=deal_data(instr)#is a list
            for item in str_list:
                if find_word(item)!='' and find_word(item) not in result:
                    result.append(find_word(item))
                else:
                    dp_list=deep_data(item)#is a list
                    for dpic in dp_list:
                        if find_word(dpic) != '' and find_word(dpic) not in result:
                            result.append(find_word(dpic))
                        else:
                            xsingle=single_find(dpic,model)
                            if xsingle!='' and xsingle not in result:
                                # print('xsing:',xsingle)
                                result.append(xsingle)
                            else:
                                rela_char = find_from_w2v(dpic, model)
                                xw2c=from_w2c_next(rela_char,model)
                                if xw2c!='' and xw2c not in result:
                                    # print('xw2c:',xw2c)
                                    result.append(xw2c)

        # print('6:',result)
        if len(result)>0:
            if type(result) == list:
                result=dele_stop_word(result)
                result=rm_duplicate(result)
                result = '_'.join(result)
            if type(result) == str:
                result=list(result.split('_'))
                result = dele_stop_word(result)
                result = rm_duplicate(result)
                result = '_'.join(result)
            if len(result)>29:
                result=dsc(result)

            # print('7:', result)
            return result

        else:
            return ''
        # print('_'.join(result))
    else:
        return instr



# print(do_trans(input))
def batch_do_trans(model):
    csvFile = open(r"C:\Users\XUEJW\Desktop\兴业数据\xy_dataset\中对英\db\试验测试\test1.csv", "r", encoding='UTF-8')
    reader = csv.reader(csvFile)  # 返回的是迭代类型
    csvFile2 = open(r"C:\Users\XUEJW\Desktop\兴业数据\xy_dataset\中对英\db\试验测试\new1_1.csv", 'w', newline='',
                    encoding='UTF-8')  # 设置newline，否则两行之间会空一行
    writer = csv.writer(csvFile2)
    for item in reader:

        data=[]
        if len(item) > 0:
            data.append(item[0])
            if type(do_trans(item[0],model))==str:
                # print('0:',do_trans(item[0]))
                data.append(do_trans(item[0],model))
            else:
                # print('0:', item[0])
                data+=do_trans(item[0],model)

            writer.writerow(data)

    csvFile2.close()

    csvFile.close()
def persis_input(model):
    astr=''
    while astr!='q':

        oldtime = time.clock()
        astr=input('Now the model is loaded ,please input:')
        print(do_trans(astr,model))
        print(time.clock() - oldtime)
# print(do_trans('客户号',model))
# batch_do_trans(model)
# print(time.clock()-oldtime)
persis_input(model)
cursor.close()
conn.close()