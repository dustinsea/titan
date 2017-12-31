# -*- coding:utf-8 -*-
'''
tf-titanic
'''
import os
import sys
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
import tflearn
from tflearn.data_utils import to_categorical
import pdb

file_train = 'train.csv'
file_test = 'test.csv'
file_rslt = 'submit.csv'

def build_net_work(input_len):
    # Build neural network
    net = tflearn.input_data(shape=[None, input_len])
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net)
    # Define model
    model = tflearn.DNN(net)
    # Start training (apply gradient descent algorithm)
    return model


### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    #print(known_age)
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    #print(unknown_age)

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1:])
    #print(predictedAges)
    
    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = 1.
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = 0.
    return df

def set_Sex(df):
    df.loc[df.Sex.isnull(), 'Sex'] = 2.
    df.loc[df.Sex == 'female', 'Sex'] = 0.
    df.loc[df.Sex == 'male', 'Sex'] = 1.
    return df

def set_Embarked(df):
    df.loc[df.Embarked.isnull(), 'Embarked'] = 0
    df.loc[df.Embarked == 'S', 'Embarked'] = 1
    df.loc[df.Embarked == 'C', 'Embarked'] = 2
    df.loc[df.Embarked == 'Q', 'Embarked'] = 3
    return df

def series_2_array(s):
    a = np.zeros(len(s))
    for idx in range(len(s)):
        a[idx] = s[idx][0]
    return a

def preprocess(df, list_fea):
    #df = df.drop(['Name', 'Ticket', 'PassengerId', 'Sex', 'Embarked'], axis=1)
    print(df.describe())
    print(df.columns)

    df = set_Cabin_type(df)
    df.loc[ (df.Age.isnull()), 'Age' ] = 150.0
    df = set_Sex(df)
    df = set_Embarked(df)
    df = set_Title(df)
    dm = df[list_fea].as_matrix()
    return dm

def evaluate(pred, YA):
    Y_ = [0. if n > 0.5 else 1. for n,p in pred]
    right = 0.
    for idx in range(len(YA)):
        if Y_[idx] > 0.5 and YA[idx] > 0.5 or Y_[idx] < 0.5 and YA[idx] < 0.5:
            right += 1.
    precision = right/len(YA)
    return precision

def add_Title(df):
    df['Title'] = df['Name'].map(lambda x:x.split(' ')[1].replace(',','').replace('.',''))
    return df

def set_Title(df):
    df = add_Title(df)
    print(df['Title'].loc[:5])
    dict_Title = {'Capt':1.0,'Col':1.0,'Major':1.0,'Dr':1.0,'Rev':1.0
                  ,'Mme':2.0,'Ms':2.0,'Mrs':2.0,'Miss':2.0
                  ,'Mr':3.0
                  ,'Master':4.0}
    df['Title'] = df['Title'].map(dict_Title)
    df.loc[(df.Title.isnull()), 'Title'] = 5.0
    return df

def cv(data_train, data_test):
    pass
    将代码上传git


def main_start(argv):
    data_train = pd.read_csv(file_train)
    data_test = pd.read_csv(file_test)
    cv(data_train, data_test)

    Ydf_train = data_train[['Survived']]
    print(data_train[['Survived', 'Name', 'Embarked']].loc[0:10,:])
    data_train.drop('Survived', axis = 1)
    YA_train = series_2_array(Ydf_train.values)
    Y_train = to_categorical(YA_train, nb_classes = 2)

    list_fea = ['Embarked', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Sex', 'Title']
    X_train = preprocess(data_train, list_fea)
    X_test = preprocess(data_test, list_fea)

    #build network model
    #pdb.set_trace()
    model = build_net_work(len(list_fea))

    model.fit(X_train, Y_train, n_epoch=500, batch_size=16, show_metric=True)

    # prediction on train set

    pred_train = model.predict(X_train)
  
    precision = evaluate(pred_train, YA_train)
    print('train precision:%f' % (precision))

    # predictin on test set
    #pred_test = model.predict(X_test)

if __name__ == '__main__':
    main_start(sys.argv)

