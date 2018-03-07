#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:15:59 2018

@author: wangjian
"""
import cPickle
import pandas as pd
import numpy as np
from multiprocessing import Pool
import csv,cPickle


#新浪微博挑战赛，评价部分：
def _deviation(predict, real, kind):
    t = 5.0 if kind=='f' else 3.0
    return abs(predict - real) / (real + t)


def _precision_i(fp, fr, cp, cr, lp, lr):
    return 1 - 0.5 * _deviation(fp, fr, 'f') - 0.25 * _deviation(cp, cr, 'c') - 0.25 * _deviation(lp, lr, 'l')


def _sgn(x):
    return 1 if x>0 else 0


def _count_i(fr, cr, lr):
    x = fr + cr + lr
    return 100 if x>100 else (x+1)


def precision(real_and_predict):
    numerator,denominator = 0.0,0.0
    for  fr, cr, lr,fp, cp, lp in real_and_predict:
        numerator += _count_i(fr, cr, lr) * _sgn(_precision_i(fp, fr, cp, cr, lp, lr) - 0.8)
        denominator += _count_i(fr, cr, lr)
    return numerator / denominator


#数据预处理部分：



def loadData():
    traindata = pd.read_csv("/Users/wangjian/Desktop/Weibo Data／weibo_train_data.txt",header=None,sep='\t')
    traindata.columns = ["uid","mid","date","forward","comment","like","content"]

    testdata = pd.read_csv("/Users/wangjian/Desktop/Weibo Data／weibo_predict_data.txt",header=None,sep='\t')
    testdata.columns=["uid","mid","date","content"]

    return traindata,testdata

#for every uid , generate statistics of forward,comment,like
def genUidStat():
    traindata, _ = loadData()
    train_stat = traindata[['uid','forward','comment','like']].groupby('uid').agg(['min','max','median','mean'])
    #求出各组的四个统计量
    train_stat.columns = ['forward_min','forward_max','forward_median','forward_mean',\
                          'comment_min','comment_max','comment_median','comment_mean',\
                          'like_min','like_max','like_median','like_mean']
    #重新给行命名
    train_stat = train_stat.apply(pd.Series.round)
    #store into dictionary,for linear time search
    stat_dic = {}
    for uid,stats in train_stat.iterrows():
        stat_dic[uid] = stats   #type(stats) : pd.Series
    return stat_dic
    
#第三部分：线性搜索


def score(uid_data,pred):
	"""
	uid_data:
		pd.DataFrame
	pred:
		list, [fp,cp,lp]
	"""
	uid_real_pred = uid_data[['forward','comment','like']]
	uid_real_pred['fp'] = pred[0]
	uid_real_pred['cp'] = pred[1]
	uid_real_pred['lp'] = pred[2]
	return precision(uid_real_pred.values)
	



#search and return the best target value for uid
def search(uid_data,target,args):
	"""
	target:
		'forward','comment','like'
	
	args:
		(f_min,f_median,f_max,c_min,c_median,c_max,l_min,l_medain,l_max)
	"""
	args = list(args)
	target_index = ['forward','comment','like'].index(target)
	target_min,target_median,target_max = args[3*target_index:3*target_index+3]
	del args[3*target_index:3*target_index+3]
	pred = (args[1],args[4])
	
	best_num = [target_median]
	best_pred = list(pred)
	best_pred.insert(target_index,target_median)
	best_score = score(uid_data,best_pred)
	for num in range(target_min,target_max+1):
		this_pred = list(pred)
		this_pred.insert(target_index,num)
		this_score = score(uid_data,this_pred)
		if this_score >= best_score:                  
			if this_score > best_score:
				best_num = [num]
				best_score = this_score
			else:
				best_num.append(num)                       
			
	return best_num[np.array([abs(i - target_median) for i in best_num]).argmin()]


##search best target value for all uid,return a dictionary that store {uid:[f,c,l]}
def search_all_uid():
	"""
	traindata,testdata = loadData()
	stat_dic = genUidStat()
	
	#for each uid,search its best fp,cp,lp
	uid_best_pred = {}
	for uid in stat_dic:
		print "search uid: {}".format(uid)
		uid_data = traindata[traindata.uid == uid]
		args = stat_dic[uid][['forward_min','forward_median','forward_max','comment_min',\
					'comment_median','comment_max','like_min','like_median','like_max']]
		args = tuple([int(i) for i in args]) 
		fp = search(uid_data,'forward',args)	
		cp = search(uid_data,'comment',args)	
		lp = search(uid_data,'like',args)	
		uid_best_pred[uid] = [fp,cp,lp]
	"""
	#multiprocessing version for geting uid_best_pred
	traindata,testdata = loadData()
	stat_dic = genUidStat()
	uid_best_pred = {}
	pool = Pool()
	uids,f,c,l = [],[],[],[]
	for uid in stat_dic:
		print "search uid:{}".format(uid)
		uid_data = traindata[traindata.uid == uid]
		arguments = stat_dic[uid][['forward_min','forward_median','forward_max','comment_min',\
					'comment_median','comment_max','like_min','like_median','like_max']]
		arguments = tuple([int(i) for i in arguments]) 
		f.append(pool.apply_async(search,args=(uid_data,'forward',arguments)))
		c.append(pool.apply_async(search,args=(uid_data,'comment',arguments)))
		l.append(pool.apply_async(search,args=(uid_data,'like',arguments)))
		uids.append(uid)
	pool.close()
	pool.join()
	f = [i.get() for i in f]
	c = [i.get() for i in c]
	l = [i.get() for i in l]
	
	for i in range(len(uids)):
		uid_best_pred[uids[i]] = [f[i],c[i],l[i]]
	
	try:
		cPickle.dump(uid_best_pred,open('uid_best_pred.pkl','w'))
	except Exception:
		pass
		
	return uid_best_pred

	

def predict_by_search(submission=True):
	traindata,testdata = loadData()
	uid_best_pred = search_all_uid()
	print "search done,now predict on traindata and testdata..."

	#predict traindata with uid's best fp,cp,lp
	forward,comment,like = [],[],[]
	for uid in traindata['uid']:
		if uid_best_pred.has_key(uid):
			forward.append(int(uid_best_pred[uid][0]))
			comment.append(int(uid_best_pred[uid][1]))
			like.append(int(uid_best_pred[uid][2]))
		else:
			forward.append(0)
			comment.append(0)
			like.append(0)
	
	#score on the traindata
	train_real_pred = traindata[['forward','comment','like']]
	train_real_pred['fp'],train_real_pred['cp'],train_real_pred['lp'] = forward,comment,like
	print "Score on the training set:{0:.2f}%".format(precision(train_real_pred.values)*100)	
	
	
	if submission:
		test_pred = testdata[['uid','mid']]
		forward,comment,like = [],[],[]
		for uid in testdata['uid']:
			if uid_best_pred.has_key(uid):
				forward.append(int(uid_best_pred[uid][0]))
				comment.append(int(uid_best_pred[uid][1]))
				like.append(int(uid_best_pred[uid][2]))
			else:
				forward.append(0)
				comment.append(0)
				like.append(0)
		test_pred['fp'],test_pred['cp'],test_pred['lp'] = forward,comment,like
		
		#generate submission file
		result = []
		filename = "weibo_predict_search.txt"
		for _,row in test_pred.iterrows():
			result.append("{0}\t{1}\t{2},{3},{4}\n".format(row[0],row[1],row[2],row[3],row[4]))
		f = open(filename,'w')
		f.writelines(result)
		f.close()
		print 'generate submission file "{}"'.format(filename)
		
if __name__ == "__main__":
		predict_by_search()




#第四部分，拟合值预测

def predict_with_fixed_value(forward,comment,like,submission=True):
	# type check
	if isinstance(forward,int) and isinstance(forward,int) and isinstance(forward,int):
		pass
	else:
		raise TypeError("forward,comment,like should be type 'int' ")
	
	traindata,testdata = loadData()
	
	#score on the training set
	train_real_pred = traindata[['forward','comment','like']]
	train_real_pred['fp'],train_real_pred['cp'],train_real_pred['lp'] = forward,comment,like
	print "Score on the training set:{0:.2f}%".format(precision(train_real_pred.values)*100)
	
	#predict on the test data with fixed value, generate submission file
	if submission:
		test_pred = testdata[['uid','mid']]
		test_pred['fp'],test_pred['cp'],test_pred['lp'] = forward,comment,like
		
		result = []
		filename = "weibo_predict_{}_{}_{}.txt".format(forward,comment,like)
		for _,row in test_pred.iterrows():
			result.append("{0}\t{1}\t{2},{3},{4}\n".format(row[0],row[1],row[2],row[3],row[4]))
		f = open(filename,'w')
		f.writelines(result)
		f.close()
		print 'generate submission file "{}"'.format(filename)



def predict_with_stat(stat="median",submission=True):
	"""
	stat:
		string
		min,max,mean,median
	"""
	stat_dic = genUidStat()
	traindata,testdata = loadData()
	
	#get stat for each uid
	forward,comment,like = [],[],[]
	for uid in traindata['uid']:
		if stat_dic.has_key(uid):
			forward.append(int(stat_dic[uid]["forward_"+stat]))
			comment.append(int(stat_dic[uid]["comment_"+stat]))
			like.append(int(stat_dic[uid]["like_"+stat]))
		else:
			forward.append(0)
			comment.append(0)
			like.append(0)
	#score on the training set
	train_real_pred = traindata[['forward','comment','like']]
	train_real_pred['fp'],train_real_pred['cp'],train_real_pred['lp'] = forward,comment,like
	print "Score on the training set:{0:.2f}%".format(precision(train_real_pred.values)*100)
	
	#predict on the test data with fixed value, generate submission file
	if submission:
		test_pred = testdata[['uid','mid']]
		forward,comment,like = [],[],[]
		for uid in testdata['uid']:
			if stat_dic.has_key(uid):
				forward.append(int(stat_dic[uid]["forward_"+stat]))
				comment.append(int(stat_dic[uid]["comment_"+stat]))
				like.append(int(stat_dic[uid]["like_"+stat]))
			else:
				forward.append(0)
				comment.append(0)
				like.append(0)
				
				
		test_pred['fp'],test_pred['cp'],test_pred['lp'] = forward,comment,like
		
		result = []
		filename = "wjhust weibo_predict_{}.txt".format(stat)
		for _,row in test_pred.iterrows():
			result.append("{0}\t{1}\t{2},{3},{4}\n".format(row[0],row[1],row[2],row[3],row[4]))
		f = open(filename,'w')
		f.writelines(result)
		f.close()
		print 'generate submission file "{}"'.format(filename)
	



if __name__ == "__main__":
	 #predict_with_fixed_value(0,1,1,submission=False)
	 predict_with_stat(stat="median",submission=True)
































