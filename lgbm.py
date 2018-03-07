#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:26:24 2018

@author: wangjian
"""
from sklearn.model_selection import train_test_split,KFold,TimeSeriesSplit
from sklearn import model_selection, preprocessing
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import model_selection, preprocessing
import pdb

def process(train,test):
    RS=1
    np.random.seed(RS)
    ROUNDS = 1500 # 1300,1400 all works fine
    params = {
            #处理回归问题
           'objective': 'regression',
            #损失函数
            'metric': 'rmse',
            #训练方式
            'boosting': 'gbdt',
            'learning_rate': 0.01 , #small learn rate, large number of iterations
            'verbose': 0,
            'num_leaves': 2 ** 5,
            #随机选95%的数据用于bagging
            'bagging_fraction': 0.95,
            #每1次迭代都会bagging一次
            'bagging_freq': 1,
            'bagging_seed': RS,
            #每次随机选取80%的特征来训练
            'feature_fraction': 0.7,
            'feature_fraction_seed': RS,
            #最大特征分割
            'max_bin': 100,
            #最大深度
            'max_depth': 7,
            'num_rounds': ROUNDS,
        }
    #Remove the bad prices as suggested by Radar
    train=train[(train.price_doc>1e6) & (train.price_doc!=2e6) & (train.price_doc!=3e6)]
    #loc是根据dataframe的具体标签选取列
    train.loc[(train.product_type=='Investment') & (train.build_year<2000),'price_doc']*=0.9 
    train.loc[train.product_type!='Investment','price_doc']*=0.969 #Louis/Andy's magic number
    test = pd.read_csv('../input/test.csv',parse_dates=['timestamp'])

  
    id_test = test.id
    times=pd.concat([train.timestamp,test.timestamp])
    num_train=train.shape[0]#行宽
    y_train = train["price_doc"]
    #inplace=False表示要对结果显示，而True表示对结果不显示
    #按着行对齐，即两个表行名字相同的，第二个表的行放在和第一个表相同行的后边继续原来的行
    train.drop(['price_doc'],inplace=True,axis=1)
    da=pd.concat([train,test])
    #注意这里的concat，由于train和test列名字基本一样，因此前m行是train，后n行是test上下合并
    #axis=0表示的是要对横坐标操作，axis=1是要对纵坐标操作
    #统计缺省值
    da['na_count']=da.isnull().sum(axis=1)
    df_cat=None
    to_remove=[]
    for c in da.columns:
        if da[c].dtype=='object':
            oh=pd.get_dummies(da[c],prefix=c)
            if df_cat is None:
                df_cat=oh
            else:
                df_cat=pd.concat([df_cat,oh],axis=1)
            to_remove.append(c)
    da.drop(to_remove,inplace=True,axis=1)

    #Remove rare features,prevent overfitting
    to_remove=[]
    if df_cat is not None:
        sums=df_cat.sum(axis=0)
        #axis=0：对矩阵，将所有a[i][1]相加，最后是1行n列，竖着加
        #axis=1:对矩阵，将每行元素直接相加，原数组n行，则最后1行n列，横着加
        to_remove=sums[sums<200].index.values
        #difference（）可以获得差集，但是这个差集是“出现在第一个集合但不出现在第二个集合”的元素！！！
        #下边表示出现在dfcat中但没有出现在toremove中
        df_cat=df_cat.loc[:,df_cat.columns.difference(to_remove)]
        da = pd.concat([da, df_cat], axis=1)
    x_train=da[:num_train].drop(['timestamp','id'],axis=1)
    x_test=da[num_train:].drop(['timestamp','id'],axis=1)
    #Log transformation, boxcox works better.
    y_train=np.log(y_train)
    train_lgb=lgb.Dataset(x_train,y_train)
    model=lgb.train(params,train_lgb,num_boost_round=ROUNDS)
    predict=model.predict(x_test)
    predict=np.exp(predict)
    return predict,id_test
if __name__=='__main__':
    train = pd.read_csv('../input/train.csv',parse_dates=['timestamp'])
    test = pd.read_csv('../input/test.csv',parse_dates=['timestamp'])
    predict,id_test=process(train,test)
    output=pd.DataFrame({'id':id_test,'price_doc':predict})
    output.to_csv('lgb.csv',index=False)