#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:26:24 2018

@author: wangjian
"""
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import xgboost as xgb

#一：数据预处理
def process_log():
    train = pd.read_csv('../input/train.csv',parse_dates=['timestamp'])#训练数据
    test = pd.read_csv('../input/test.csv',parse_dates=['timestamp'])#测试数据
    #筛选训练数据
    train=train[(train.price_doc>1e6) & (train.price_doc!=2e6)  & (train.price_doc!=3e6)  ]
    ######计算训练数据的price.doc#
    train['price_doc']*=0.969
    ##pandas的函数reset_index：还原索引，drop为false，索引会被还原成普通列，否则会丢失！    train=train.reset_index(drop=True)
    id_test = test.id 
    #轴向连接，使得train的timestamp之后为test的timestamp（转换时间）
    times=pd.concat([train.timestamp,test.timestamp])
    #train的响应变量y设置】为train的price
    y_train = train["price_doc"]
    
    #获得行数
    num_train=len(train)
    #轴向连接，使得train的timestamp之后为test的timestamp
    times=pd.concat([train.timestamp,test.timestamp])
    #删除3个列"id", "timestamp", "price_doc"
    x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
    #删除2个列"id", "timestamp"
    x_test = test.drop(["id", "timestamp"], axis=1)
    #合并train 和test，作为总的x数据
    df_all=pd.concat([x_train,x_test])
    df_cat=None
    
    #数据预处理过程：
    for c in df_all.columns:
        if df_all[c].dtype == 'object':
            #如果c的类别是sub_area
            if c=='sub_area':
                #进行热编码：有时候特征是类别型的，而一些算法的输入必须是数值型，此时需要对其编码。
                #相当于为了使用sklearn的OneHotEncoder，先把字符型的特征转化为整型的特征
                oh=pd.get_dummies(df_all[c],prefix=c)
                ########df_cat的功能：用来判断是否把c的sub转换编码了
                if df_cat is None:#空的就赋值刚才的编码
                    df_cat=oh
                else:#非空就联结刚才的编码
                    df_cat=pd.concat([df_cat,oh],axis=1)
                #换掉c列，在原数组上直接操作
                df_all.drop([c],inplace=True,axis=1)
            else:#LabelEncoder对不连续的数字或者文本进行编号，整个三句
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(df_all[c].values))
                df_all[c] = lbl.transform(list(df_all[c].values))

    if df_cat is not None:#转换完之后就合并
        df_all = pd.concat([df_all, df_cat], axis=1)
        
        
#二：建立xboost选择线性模型

    x_train=df_all[:len(x_train)]
    x_test=df_all[len(x_train):]

    xgb_params = {
        #为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。 eta通过缩减特征的权重使提升计算过程更加保守。缺省值为0.3
        #取值范围为：[0,1]
        #通常最后设置eta为0.01~0.2
        'eta': 0.05,
        #树的最大深度
        #树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合
        #建议通过交叉验证（xgb.cv ) 进行调参
        #通常取值：3-10
        'max_depth': 5,
        #用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。
        #取值范围为：(0,1]
        'subsample': 0.7,
         #colsample_bytree [default=1] 
        #在建立树时对特征随机采样的比例。缺省值为1
        #取值范围：(0,1]       
        'colsample_bytree': 0.7,
 #       objective [ default=reg:linear ] 
#定义学习任务及相应的学习目标，可选的目标函数如下：
#“reg:linear” –线性回归。
#“reg:logistic” –逻辑回归。
#“binary:logistic” –二分类的逻辑回归问题，输出为概率。
#“binary:logitraw” –二分类的逻辑回归问题，输出的结果为wTx。
#“count:poisson” –计数问题的poisson回归，输出结果为poisson分布。
#在poisson回归中，max_delta_step的缺省值为0.7。(used to safeguard optimization)
#“multi:softmax” –让XGBoost采用softmax目标函数处理多分类问题，同时需要设置参数num_class（类别个数）
#“multi:softprob” –和softmax一样，但是输出的是ndata * nclass的向量，可以将该向量reshape成ndata行nclass列的矩阵。每行数据表示样本所属于每个类别的概率。

        
        'objective': 'reg:linear',
        #评价指标：“rmse”: root mean square error
        'eval_metric': 'rmse',
        #当这个参数值为1时，静默模式开启，不会输出任何信息。 
        #一般这个参数就保持默认的0，因为这样能帮我们更好地理解模型。
        'silent': 1,
    }

    x_train=df_all[:len(x_train)]
    x_test=df_all[len(x_train):]

    #迭代次数
    num_boost_rounds=345
    
    dtrain = xgb.DMatrix(x_train, np.log(y_train))
    dtest = xgb.DMatrix(x_test)
    model = xgb.train(dict(xgb_params, max_depth=5,silent=1), dtrain,num_boost_round= num_boost_rounds)
    #输出预测值
    y_predict_log=np.exp(model.predict(dtest))
    y_predict=y_predict_log
    return id_test,y_predict




def process():
    train = pd.read_csv('../input/train.csv',parse_dates=['timestamp'])
    train['price_doc']*=0.969
    test = pd.read_csv('../input/test.csv',parse_dates=['timestamp'])
    id_test = test.id

    times=pd.concat([train.timestamp,test.timestamp])
    y_train = train["price_doc"]
    num_train=len(train)

    x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
    x_test = test.drop(["id", "timestamp"], axis=1)
    df_all=pd.concat([x_train,x_test])

    for c in df_all.columns:
        if df_all[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df_all[c].values))
            df_all[c] = lbl.transform(list(df_all[c].values))
    x_train=df_all[:len(x_train)]
    x_test=df_all[len(x_train):]

    
    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1,
    }

    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test)

    num_boost_rounds=345
    model = xgb.train(dict(xgb_params, silent=1), dtrain,num_boost_round= num_boost_rounds)
    y_predict = model.predict(dtest)
  
        
    return id_test,y_predict
if __name__=='__main__':
    id_test,y_predict=process()
    id_test,y_predict_log=process_log()
    print('Mean:',y_predict.mean(), 'LB 0.3113')
    print ('LOG Mean:',y_predict_log.mean(),'LB 0.314-0.315')
    output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
    output.to_csv('xgb.csv', index=False)
    output = pd.DataFrame({'id': id_test, 'price_doc': y_predict_log})
    output.to_csv('xgb_log.csv', index=False)