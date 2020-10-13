import numpy as np
import pandas as pd
from os import path
from urllib import request
from sklearn.preprocessing import MinMaxScaler,Normalizer,RobustScaler,OneHotEncoder

#from utils import save_generated_data
import zipfile
import matplotlib.pyplot as plt
import math
import torch
import operator
from sklearn.model_selection import train_test_split
from utils import get_data_info
'''#-----------------------------
#Compas: orginal :(7214, 53)
# After preproces: (5278, 11)
# predict two_year_recid, 
# Task: binary classification

'''
class SyntheticCompasDataset():
    def __init__(self,i=1):
        print("load synthetic compas")
        df_compas = pd.read_csv("./GenerateData/Compas/Compas_syn600_bs100_seed1_times_10.csv")
        df_compas = df_compas[3694*i:3694*(i+1)]
        df_compas.loc[(df_compas["diff_jail"] < 0 ,"diff_jail")]= 0
        self.con_vars = ['age','diff_custody','diff_jail','priors_count']
        self.cat_vars = [col for col in df_compas.columns if col not in self.con_vars]
        self.columns_name = self.con_vars + self.cat_vars
        self.data = df_compas[self.columns_name]
        self.con_loc =  [self.data.columns.get_loc(var) for var in self.con_vars] 
        #get data info
        self.data_info = get_data_info(self.data ,self.cat_vars)
        print("Data info:", self.data_info)  
    
    def transform(self):
        self.scaler = MinMaxScaler() 
        self.enc = OneHotEncoder()
        con_columns = self.scaler.fit_transform(self.data[self.con_vars])
        cat_columns = self.enc.fit_transform(self.data[self.cat_vars]).toarray()
        data_np = np.column_stack((con_columns,cat_columns))
        return data_np

    def inverse_transform(self,data):
        #data:numpy array

        data_cat = self.scaler.inverse_transform(data[:,self.con_loc])
        data_cat = np.round(data_cat)
        data_con = self.enc.inverse_transform(data[:,len(self.con_loc):])       
        data_inverse = np.column_stack((data_cat,data_con))
        
        print("Inverse transform completed!")
        return data_inverse

def compas_preprocess(df):

    df = df[df['days_b_screening_arrest'] >= -30]
    df = df[df['days_b_screening_arrest'] <= 30]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != '0']
    df = df[df['score_text'] != 'N/A']

    df['in_custody'] = pd.to_datetime(df['in_custody'])
    df['out_custody'] = pd.to_datetime(df['out_custody'])
    df['diff_custody'] = (df['out_custody'] - df['in_custody']).dt.days
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['diff_jail'] = (df['c_jail_out'] - df['c_jail_in']).dt.days

    df.drop(
        [
            'id', 'name', 'first', 'last', 'v_screening_date', 'compas_screening_date', 'dob', 'c_case_number',
            'screening_date', 'in_custody', 'out_custody', 'c_jail_in', 'c_jail_out'
        ], axis=1, inplace=True
    )
    df = df[df['race'].isin(['African-American', 'Caucasian'])]

    features = df.drop(['is_recid', 'is_violent_recid', 'violent_recid', 'two_year_recid'], axis=1)
    labels = 1 - df['two_year_recid']

    features = features[[
        'age', 'sex', 'race', 'diff_custody', 'diff_jail', 'priors_count', 'juv_fel_count', 'c_charge_degree',
        'v_score_text'
    ]]

    data = pd.concat([features,labels],axis = 1)
    data[['juv_fel_count','two_year_recid']] = data[['juv_fel_count','two_year_recid']].astype('object')
    con_vars = [i for i in data.columns if data[i].dtype=='int64'or data[i].dtype=='float64']
    return data, con_vars

class CompasDataset():
    def __init__(self):

        print("load compas")
        data_dir = 'dataset'
        data_file = path.join(data_dir,'compas-scores-two-years.csv')

        df = pd.read_csv(data_file)
        print(df.shape)
        
        df = df[df['days_b_screening_arrest'] >= -30]
        df = df[df['days_b_screening_arrest'] <= 30]
        df = df[df['is_recid'] != -1]
        df = df[df['c_charge_degree'] != '0']
        df = df[df['score_text'] != 'N/A']

        df['in_custody'] = pd.to_datetime(df['in_custody'])
        df['out_custody'] = pd.to_datetime(df['out_custody'])
        df['diff_custody'] = (df['out_custody'] - df['in_custody']).dt.days
        df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
        df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
        df['diff_jail'] = (df['c_jail_out'] - df['c_jail_in']).dt.days

        df.drop(
            [
                'id', 'name', 'first', 'last', 'v_screening_date', 'compas_screening_date', 'dob', 'c_case_number',
                'screening_date', 'in_custody', 'out_custody', 'c_jail_in', 'c_jail_out'
            ], axis=1, inplace=True
        )
        df = df[df['race'].isin(['African-American', 'Caucasian'])]

        features = df.drop(['is_recid', 'is_violent_recid', 'violent_recid', 'two_year_recid'], axis=1)
        labels = 1 - df['two_year_recid']

        features = features[[
            'age', 'sex', 'race', 'diff_custody', 'diff_jail', 'priors_count', 'juv_fel_count', 'c_charge_degree',
            'v_score_text'
        ]]
        self.data = pd.concat([features,labels],axis = 1)
        self.data[['juv_fel_count','two_year_recid']] = self.data[['juv_fel_count','two_year_recid']].astype('object')
        
        #self.data = self.data.drop(['diff_jail'],axis=1)
        # discretize diff_custody
        #diff_custody(self.data)

        self.con_vars = [i for i in self.data.columns if self.data[i].dtype=='int64'or self.data[i].dtype=='float64']

        self.cat_vars = [i for i in self.data.columns if i not in self.con_vars]
        
        self.columns_name = self.con_vars + self.cat_vars
        self.data = self.data[self.columns_name]
        self.con_loc =  [self.data.columns.get_loc(var) for var in self.con_vars]  

        #get data info
        self.data_info = get_data_info(self.data ,self.cat_vars)
        print("Data info:", self.data_info)  

    
    def transform(self):
        self.scaler = MinMaxScaler() 
        self.enc = OneHotEncoder()
        con_columns = self.scaler.fit_transform(self.data[self.con_vars])
        cat_columns = self.enc.fit_transform(self.data[self.cat_vars]).toarray()
        data_np = np.column_stack((con_columns,cat_columns))
        return data_np

    def inverse_transform(self,data):
        #data:numpy array
        data_cat = self.scaler.inverse_transform(data[:,self.con_loc])
        data_cat = np.round(data_cat)
        data_con = self.enc.inverse_transform(data[:,len(self.con_loc):])       
        data_inverse = np.column_stack((data_cat,data_con))

        #save_generated_data(data=data_i,name="Adult_syn",column_name=self.columns_name)
        print("Inverse transform completed!")
        return data_inverse

def discretization(data):
    cat_vars = ['diff_custody','diff_jail']
    for column in cat_vars:
        max_item = data[column].max()+1
        min_item = data[column].min()-1
        q1 = data[column].quantile(.6) + 0.5
        q2 = data[column].quantile(.85) + 0.5
        if q2 ==q1:
            q2 +=1
        bins = pd.IntervalIndex.from_tuples([(min_item, q1,), (q1, q2), (q2, max_item)])
        data[column] = pd.cut(data[column], bins)

def diff_custody(data):
    data.loc[(data["diff_custody"] > 1000),"diff_custody"] = 'high'
    data.loc[(data["diff_custody"] == 0 ,"diff_custody")]= 'zero'
    data.loc[operator.and_(data["diff_custody"]!='zero', data["diff_custody"]!='high' ),"diff_custody"] = 'low'

def load_Compas():
    tf = Compas_Discretization_transform()
    combined = tf.data
    label = combined['two_year_recid']
    features = combined.drop(['two_year_recid'], axis=1)
    scaler = MinMaxScaler()

    con_vars = [i for i in features.columns if features[i].dtype=='int64'or features[i].dtype=='float64']
    cat_vars = [i for i in features.columns if i not in con_vars]

    features[con_vars] = scaler.fit_transform(features[con_vars])
    features = pd.get_dummies(features, columns=cat_vars, prefix_sep='=')

    X_train, X_test, y_train, y_test = train_test_split(features, label,test_size=0.3, random_state=2)
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
    return X_train,y_train,X_test,y_test
    