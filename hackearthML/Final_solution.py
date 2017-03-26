# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 14:50:03 2017

@author: AshwinthPS - Hackerearth-ML Challenge.
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew,boxcox
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder


mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb

os.chdir("C:\Users\HP\Desktop\Hackathon\hackerearth")

## Reading Data and combining for preprocessing.

train=pd.read_csv("train_indessa.csv")
test=pd.read_csv("test_indessa.csv")

train_test=pd.concat(([train.iloc[:,:train.shape[1]-1],test.iloc[:,:test.shape[1]]]))

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

##

'''
corrplot1=train.select_dtypes(include=numerics).corr(method="spearman")
sns.heatmap(corrplot1,square=True)
plt.show()
'''

##

train_test=train_test.drop(["funded_amnt","funded_amnt_inv","collection_recovery_fee",'member_id','desc'],axis=1) #also dropping member id and desc

train_test['emp_length']=train_test['emp_length'].apply(lambda x:x.split(" ")[0]).replace('<',1).replace('10+','10')
#train_test['emp_length']=train_test['emp_length'].replace('n/a',np.NaN)

train_test['last_week_pay']=train_test['last_week_pay'].apply(lambda x:x.split("th week")[0])
#train_test['last_week_pay']=train_test['last_week_pay'].replace('n/a',np.NaN)

train_test['zip_code']=train_test['zip_code'].apply(lambda x:x.split("xx")[0])
#train_test['zip_code']=train_test['zip_code'].replace('n/a',np.NaN)

train_test.info()


#train['initial_list_status']=pd.factorize(train['initial_list_status'])[0]
#test['initial_list_status']=pd.factorize(test['initial_list_status'])[0]

'''
sns.distplot(train_test['annual_inc'][-pd.isnull(train_test['annual_inc'])],hist=False)
sns.distplot(np.log10(train_test['annual_inc'])[-pd.isnull(train_test['annual_inc'])],hist=False)
sns.distplot(np.log10(train_test['annual_inc']+10)[-pd.isnull(train_test['annual_inc'])],hist=False)
sns.distplot(np.log10(train_test['annual_inc']+20)[-pd.isnull(train_test['annual_inc'])],hist=False)

sns.kdeplot(train_test['annual_inc'][-pd.isnull(train_test['annual_inc'])])
sns.kdeplot(np.log(train_test['annual_inc'])[-pd.isnull(train_test['annual_inc'])])
'''

'''

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

'''


'''
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
sk=train_test.select_dtypes(include=numerics).skew()
sk
sk > 2
colnames=train_test.select_dtypes(include=numerics).columns[sk>2]
for i in colnames:
    train_test[i]=np.log10(train_test[i]+10)

'''
##

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeric_feats=train_test.select_dtypes(include=numerics).columns

skewed_feats = train_test[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.2]
skewed_feats = skewed_feats.index

for feats in skewed_feats:
    train_test[feats] = train_test[feats] + 1
    train_test[feats], lam = boxcox(train_test[feats])

##

train_test=train_test.drop(['pymnt_plan','verification_status_joint','application_type','title','batch_enrolled'],axis=1)

print('Label Encoding')

cat=['object']
train_test.select_dtypes(include=cat)
cat_col=train_test.select_dtypes(include=cat).columns

''' 
rm=['emp_length','zip_code','last_week_pay']
cat_col=[c for c in cat_col if c not in rm]
cat_col
'''



## Label encoder cannot handle NA.

for f in cat_col:
    lbl = LabelEncoder()
    lbl.fit(train_test[f].values)
    train_test[f] = lbl.transform(train_test[f].values)

train_x=train_test.iloc[:train.shape[0],:]
test_x=train_test.iloc[train.shape[0]:,:]
train_y=train.loan_status

##############################################  Modelling ############################################

folds = 5
cv_sum = 0
early_stopping = 25
fpred = []
xgb_rounds = []

d_test = xgb.DMatrix(test_x,missing=np.nan)

kf = KFold(train_x.shape[0], n_folds=folds)

for i, (train_index, test_index) in enumerate(kf):
    print('\n Fold %d\n' % (i + 1))
    print i,(train_index, test_index)
    X_train, X_val = train_x.iloc[train_index], train_x.iloc[test_index]
    y_train, y_val = train_y.iloc[train_index], train_y.iloc[test_index]
    
    params = {"objective": "binary:logistic","booster": "gbtree", "nthread": 4, "silent": 1,'eval_metric':'auc',
                "eta": 0.08, "max_depth": 13, "subsample": 0.9, "colsample_bytree": 0.7, 
                "min_child_weight": 1,"seed": 2016, "tree_method": "exact"}
    

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(d_train, 'train'), (d_valid, 'eval')]
    
    
    clf = xgb.train(params,d_train,800,watchlist,early_stopping_rounds=early_stopping)
    
    xgb_rounds.append(clf.best_iteration)
    scores_val = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)
    #cv_score = mean_absolute_error(np.exp(y_val), np.exp(scores_val))
    #print(' eval-MAE: %.6f' % cv_score)
    y_pred = clf.predict(d_test, ntree_limit=clf.best_ntree_limit)

    if i > 0:
        fpred = pred + y_pred
    else:
        fpred = y_pred
    
    pred = fpred
    #cv_sum = cv_sum + cv_score

mpred = pred / folds

test1=pd.read_csv("test_indessa.csv")

op1=pd.DataFrame({'member_id':test1.member_id,'loan_status':mpred})


op1.to_csv("final_subm.csv",index=False,columns=['member_id','loan_status'])
