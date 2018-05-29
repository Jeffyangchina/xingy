#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
#@Author: Yang Xiaojun
import csv
import jieba
import re
import os
import logging
import time
logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.DEBUG)
def get_en(astr):
    outstr = re.sub("[^a-zA-Z]", "", astr)
    outstr = outstr.strip().lower()
    return outstr
def rm_null(lis):
    out_li=[]
    for cel in lis:
        if len(cel)>0:
            out_li.append(cel)
    return out_li
def rm_puc(line):
    from zhon.hanzi import punctuation as zh_puc
    from string import punctuation as en_puc
    line=re.sub("[%s]+" % en_puc, "", line)
    # line=re.sub("[%s]+" % zh_puc, "", line)
    return line
def cut_str(astr):
    alist=list(astr.split(' '))
    blist=rm_null(alist)
    out_li=[x.lower() for x in blist]
    new_li=[]

    for item in out_li:
        xn=get_en(item)
        xn=rm_puc(xn)
        if xn not in new_li and len(xn)>0:
            new_li.append(xn)

    return new_li
def clean_table():#去除数字\英文字段做下 常规处理 去重 单独提出表名
    csvFile = open(r"E:\能搜\no symbol.csv", "r", encoding='UTF-8')
    reader = csv.reader(csvFile)  # 返回的是迭代类型
    dataset = []
    csvFile2 = open(r"E:\能搜\cleaned_no symbol.csv", 'w', newline='',
                    encoding='UTF-8')  # 设置newline，否则两行之间会空一行
    writer2 = csv.writer(csvFile2)
    word_vocab={}
    new_list=[]
    in_list=[]
    i=0
    for item in reader:
        if len(item)>1:
            # print('0:',item)
            ind=item[0].strip()
            ind_li=list(ind.split('.'))
            fr_level=ind_li[0]

            if fr_level in word_vocab:
                # print('2:',fr_level,word_vocab[fr_level])
                for xx in range(1, len(item)):
                    cell=item[xx].strip()
                    cell_list=cut_str(cell)

                    for xc in cell_list:
                        if xc not in  word_vocab[fr_level]:
                            word_vocab[fr_level].append(xc)
            else:
                new_list=[]
                for xx in range(1, len(item)):

                    cell2=item[xx].strip()
                    # print('11:', xx,cell2)
                    cell_list2 = cut_str(cell2)
                    # print('4:',cell_list2)
                    for xc in cell_list2:
                        if xc not in new_list:
                            new_list.append(xc)
                    # print('5:',new_list)
                word_vocab[fr_level]=new_list
                # print('3:', fr_level, word_vocab[fr_level])

            # print('1:',word_vocab)
    for ite in word_vocab:
        in_list=[ite]+word_vocab[ite]
        writer2.writerow(in_list)
    csvFile.close()
    csvFile2.close()
# clean_table()
def rm_zero(alist):
    nu=len(alist)
    for i in range(len(alist)):
        if alist[i][1]<0.05:
            if i>0:
                nu=i-1
            else:
                nu=0
            break
    return alist[:nu]
def get_biggest(alist):#求一个列表最大的前n项,输出是下标的数组
    st=[]
    out={}

    for i in range(len(alist)):
        st.append((i,alist[i]))
    st.sort(key=lambda x: x[1] ,reverse = True)
    in_list=rm_zero(st)
    for i in range(len(in_list)):
        out[in_list[i][0]]=in_list[i][1]
    return out
def tf_idf(fpath):#求取关键字
    from sklearn.feature_extraction.text import TfidfVectorizer
    rootdir = fpath
    lists = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(lists)):
        path = os.path.join(rootdir, lists[i])
        new_path = os.path.join(rootdir, 'tfIdf_' + lists[i])
        csvFile = open(path, "r", encoding='UTF-8')
        reader = csv.reader(csvFile)  # 返回的是迭代类型
        csvFile3 = open(new_path, 'w', newline='',
                        encoding='UTF-8')  # 设置newline，否则两行之间会空一行
        writer3 = csv.writer(csvFile3)
        typeName=[]
        typeWord=[]

        for item in reader:
            if item:
                typeName.append(item[0])
                temp_li=item[1:]
                temp_wo=' '.join(temp_li)
                typeWord.append(temp_wo)
        tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")#表示不考虑停用词
        weight = tfidf.fit_transform(typeWord).toarray()#返回每个样本对应每行一个列表，长度为词汇表长，对应样本中词汇的权重
        word = tfidf.get_feature_names()#获取词汇表
        # print('w:', len(weight[0]),len(weight[2]))
        for i in range(len(weight)):
            keyW = []
            keyW_P=get_biggest(weight[i])#是个字典，键是下标索引，值是得分

            for x in keyW_P:
                in_d=word[x].upper()+':'+str(keyW_P[x])
                keyW.append(in_d)

            writer3.writerow([typeName[i]]+keyW)
        csvFile.close()
        csvFile3.close()
tf_idf(r"E:\能搜\tfIdf")
import nltk
import string
from nltk.corpus import stopwords#nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
def textPrecessing(text):#wit nltk better spacy
    from nltk.corpus import wordnet
    #小写化
    text = text.lower()
    #去除特殊标点
    for c in string.punctuation:
        text = text.replace(c, ' ')
    #分词
    wordLst = nltk.word_tokenize(text)#nltk.download('punkt')先下载
    #去除停用词
    filtered = [w for w in wordLst if w not in stopwords.words('english')]
    # #仅保留名词或特定POS
    # refiltered =nltk.pos_tag(filtered)
    # filtered = [w for w, pos in refiltered if pos.startswith('NN')]
    #词干化,反而很多不好
    # ps = PorterStemmer()
    # filtered = [ps.stem(w) for w in filtered]
    mark=0
    for xx in filtered:
        xw= wordnet.synsets(xx)
        if len(xw)!=0:
            mark+=1
    if mark>5:
        return " ".join(filtered)
    else:
        return ''
def deal_fr_mangodb():#去停用词，标点，词干化,为lda预处理,判断是否为英文
    path=r"E:\能搜\fr_mangodb.csv"
    new_path = r"E:\能搜\deal_fr_mangodb.csv"
    csvFile = open(path, "r", encoding='UTF-8')
    reader = csv.reader(csvFile)  # 返回的是迭代类型
    csvFile3 = open(new_path, 'w', newline='', encoding='UTF-8')  # 设置newline，否则两行之间会空一行
    writer3 = csv.writer(csvFile3)
    in_word=[]
    for item in reader:
        if item:
            temp_str=item[1]+' '+item[2]
            temp_str=textPrecessing(temp_str)
            if len(temp_str)>0:
                in_word.append([item[0],temp_str])
    for xx in in_word:
        writer3.writerow(xx)
    csvFile.close()
    csvFile3.close()

def get_docLst():
    new_path = r"E:\能搜\deal_fr_mangodb.csv"
    csvFile = open(new_path, "r", encoding='UTF-8')
    reader = csv.reader(csvFile)  # 返回的是迭代类型
    doclist=[]
    for item in reader:
        if item:
            xt=item[1].strip()
            doclist.append(xt)
    # print('0:',xt)
    csvFile.close()
    return doclist
def make_input():#文章-词语”稀疏矩阵，可以通过tf_vectorizer.get_feature_names()得到每一维feature对应的term
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.externals import joblib  # 也可以选择pickle等保存模型，请随意
    # 构建词汇统计向量并保存，仅运行首次
    n_features=2500
    docLst=get_docLst()
    # print('0:',docLst[0])
    tf_ModelPath=r'E:\能搜\tf_model.pkl'
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=n_features,
                                    stop_words='english')
    tf = tf_vectorizer.fit_transform(docLst)
    joblib.dump(tf_vectorizer, tf_ModelPath)
    # ==============================================================================
    # #得到存储的tf_vectorizer,节省预处理时间
    # tf_vectorizer = joblib.load(tf_ModelPath)
    # tf = tf_vectorizer.fit_transform(docLst)
    # tf_vectorizer.get_feature_names()
    # ==============================================================================

# make_input()#制作输入矩阵，lda需要矩阵输入
# deal_fr_mangodb()#制作公司名称和描述字段
def print_top_words(model, feature_names, n_top_words):
    #打印每个主题下权重较高的term
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic #%d:" % topic_idx)
        print (" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print (model.components_)
def train_lda():
    # from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.externals import joblib  # 也可以选择p
    tf_ModelPath = r'E:\能搜\tf_model.pkl'
    docLst = get_docLst()
    tf_vectorizer = joblib.load(tf_ModelPath)
    tf = tf_vectorizer.fit_transform(docLst)
    # xx=tf_vectorizer.get_feature_names()
    from sklearn.decomposition import LatentDirichletAllocation
    n_topics = 13
    lda = LatentDirichletAllocation(n_components =n_topics,
                                    max_iter=300,
                                    learning_method='batch')
    lda.fit(tf)  # tf即为Document_word Sparse Matrix

    n_top_words=20
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)
    print('lda:',lda.perplexity(tf))
def deal_company_name(astr):
    alist=list(astr.strip().split(' '))
    alist=rm_null(alist)
    bl=[x.lower() for x in alist]
    bstr=' '.join(alist)
    return bstr
def union_class():
    path = r"E:\能搜\人工分类.csv"
    new_path = r"E:\能搜\deal_人工分类.csv"
    csvFile = open(path, "r", encoding='UTF-8')
    reader = csv.reader(csvFile)  # 返回的是迭代类型
    csvFile3 = open(new_path, 'w', newline='', encoding='UTF-8')  # 设置newline，否则两行之间会空一行
    writer3 = csv.writer(csvFile3)
    for item in reader:
        if item:
            lth=len(item)
            compName=deal_company_name(item[0])
            classli=item[1:]
            classli=rm_null(classli)
            if len(classli)>0:
                out_class='&'.join(classli)
            else:
                out_class=''
            writer3.writerow([compName,out_class])
    csvFile.close()
    csvFile3.close()
def pick_classed():
    path = r"E:\能搜\deal_人工分类.csv"
    path1 = r"E:\能搜\deal_fr_mangodb.csv"
    new_path = r"E:\能搜\pick_人工分类.csv"
    csvFile = open(path, "r", encoding='UTF-8')
    reader = csv.reader(csvFile)  # 返回的是迭代类型
    csvFile1 = open(path1, "r", encoding='UTF-8')
    reader1 = csv.reader(csvFile1)  # 返回的是迭代类型
    csvFile3 = open(new_path, 'w', newline='', encoding='UTF-8')  # 设置newline，否则两行之间会空一行
    writer3 = csv.writer(csvFile3)
    data={}
    for item in reader:
        if item:
            cname=deal_company_name(item[0])
            if cname not in data:
                data[cname]=item[1]
    for itx in reader1:
        if itx:
            cxn=deal_company_name(itx[0].strip())
            if cxn in data:
                writer3.writerow([cxn,itx[1],data[cxn]])
    csvFile3.close()
    csvFile.close()
    csvFile1.close()
# pick_classed()
# union_class()
# train_lda()