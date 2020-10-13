import numpy as np
import pandas as pd
from os import path
from urllib import request
from sklearn.preprocessing import MinMaxScaler
from utils import save_generated_data,get_data_info
import zipfile
import matplotlib.pyplot as plt
import math
import torch
'''
#old class code:
# combine with shadow syntheiszer:

df_health = pd.read_csv('./dataset/health_without_year.csv')
        
df_health = process_health_per_year(df_health)

df_health = discretization(df_health)

con_vars = ['LabCount_total','LabCount_months','DrugCount_total','DrugCount_months'
          ,'PayDelay_total','PayDelay_max','PayDelay_min']

input df_health,con_vars to shadow synthesizer
'''

class HealthDataset():
    
    def __init__(self):

        print("load Health...")
        df_health = pd.read_csv('./dataset/health_without_year.csv')
        
        #basic preprocess
        df_health = process_health_per_year(df_health)
        #discretization, 
        df_health = discretization(df_health)

        self.con_vars = ['LabCount_total','LabCount_months','DrugCount_total','DrugCount_months'
          ,'PayDelay_total','PayDelay_max','PayDelay_min']
        
        #self.cat_vars = [col for col in df_health.columns if col not in self.con_vars]
        self.cat_vars = [col for col in df_health.columns if '=' in col]
        self.cat_vars.extend(['AgeAtFirstClaim','Sex','max_CharlsonIndex'])
        

        self.columns_name = self.con_vars + self.cat_vars
        df_health = df_health[self.columns_name]
        
        #get data info
        self.data_info = []
        for column in df_health.columns:
            if column in self.cat_vars:
                a = (df_health[column].unique().shape[0],'softmax',column)
                self.data_info.append(a)
            elif column in self.con_vars:
                self.data_info.append((1,'tanh', column)) 
            else:
                assert 0
        print("Data info:", self.data_info)    

        self.data = df_health  
        self.con_loc =  [self.data.columns.get_loc(var) for var in self.con_vars]    

    def transform(self):
        #self.scaler = MinMaxScaler(feature_range=(-1,1))
        self.scaler = MinMaxScaler()
        self.data[self.con_vars] = self.scaler.fit_transform(self.data[self.con_vars]) 
        self.data = pd.get_dummies(self.data, columns=self.cat_vars, prefix_sep='=')

        data_np = self.data.values
        #data_np = (data_np - 0.5)*2
        return data_np

    def inverse_transform(self,data):
        #data:numpy array
        data_i = []
        st = 0
        #data  = data/2 + 0.5
        
        data_c = self.scaler.inverse_transform(data[:,self.con_loc]).astype('int64')
        
        for item in self.data_info:
            if item[1] == 'softmax':
                ed = st +item[0]
                data_a = np.argmax(data[:, st:ed], axis =1)
                data_a.resize((len(data),1))
                data_i.append(data_a)
                st = ed
            elif item[1]=='tanh':
                ed = st +item[0] 
                st = ed
            else:
                assert 0
        data_soft = np.concatenate(data_i, axis=1)
        data_inverse = np.concatenate((data_c, data_soft), axis=1).astype('int64')
        #save_generated_data(data=data_i,name=save_name,column_name=self.columns_name)
        print("Inverse transform completed!")
        return data_inverse



def process_health_per_year(health_year):

    health_year['max_CharlsonIndex'] = health_year['max_CharlsonIndex'].replace([2,4,6],1)
    health_year['max_CharlsonIndex'] = 1 - health_year['max_CharlsonIndex']
    map_x = {'0-9': 0, '10-19': 1, '20-29':2, '30-39':3,
         '40-49':4, '50-59':5, '60-69':6,'70-79':7, '80+':8, '?':9}
    health_year['AgeAtFirstClaim'] = health_year['AgeAtFirstClaim'].map(map_x)
    map_y = {'?':0,'F':1,'M':2}
    health_year['Sex'] = health_year['Sex'].map(map_y)
    health_year.drop([ 'MemberID'], axis=1, inplace=True)
    #health_year.drop(['Year', 'MemberID'], axis=1, inplace=True)
    return health_year


def discretization(df_health):
    cat_vars = [col for col in df_health.columns if '=' in col]
    for column in cat_vars:
        max_item = df_health[column].max()+1
        min_item = df_health[column].min()-1
        q1 = df_health[column].quantile(.6) + 0.5
        q2 = df_health[column].quantile(.85) + 0.5
        if q2 ==q1:
            q2 +=1
        bins = pd.IntervalIndex.from_tuples([(min_item, q1,), (q1, q2), (q2, max_item)])
        df_health[column] = pd.cut(df_health[column], bins)
    return df_health

