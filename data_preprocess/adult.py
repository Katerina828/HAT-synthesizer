import numpy as np
import pandas as pd
from os import path
from urllib import request
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,OrdinalEncoder

from utils import save_generated_data,get_data_info
import zipfile
import matplotlib.pyplot as plt
import math
import torch
import operator

'''
Adult dataset: training data(31655,)
'''

'''
class AdultDataset():
    
    def __init__(self):

        print("load Real Adult...")
        df_adult = pd.read_csv('./dataset/combined_set.csv')       
        #basic preprocess
        adult_preprocess(df_adult)

        #self.cat_vars = [col for col in df_health.columns if col not in self.con_vars]
        self.con_vars = ['Age']
        self.cat_vars = [col for col in df_adult.columns if col not in self.con_vars]

        self.columns_name = self.con_vars + self.cat_vars
        self.data = df_adult[self.columns_name]
 
        #get data info
        self.data_info = get_data_info(self.data ,self.cat_vars)
        print("Data info:", self.data_info) 

        self.con_loc =  [self.data.columns.get_loc(var) for var in self.con_vars]    

    def transform(self):

        self.scaler = MinMaxScaler() 
        self.enc = OneHotEncoder()
        con_columns = self.scaler.fit_transform(self.data[self.con_vars])
        cat_columns = self.enc.fit_transform(self.data[self.cat_vars]).toarray()
        data_np = np.column_stack((con_columns,cat_columns))

        return data_np

    def inverse_transform(self,data):

        data_cat = self.scaler.inverse_transform(data[:,self.con_loc])
        data_cat = np.round(data_cat)
        data_con = self.enc.inverse_transform(data[:,len(self.con_loc):])       
        data_inverse = np.column_stack((data_cat,data_con))
        print("Inverse transform completed!")
        return data_inverse

class SyntheicAdultDataset():
    
    def __init__(self,df_adult):
        
        print("load Synthetic Adult...")
        #self.cat_vars = [col for col in df_health.columns if col not in self.con_vars]
        self.con_vars = ['Age']
        self.cat_vars = [col for col in df_adult.columns if col not in self.con_vars]

        self.columns_name = self.con_vars + self.cat_vars
        self.data = df_adult[self.columns_name]
        
        #get data info
        self.data_info = get_data_info(self.data ,self.cat_vars)
        print("Data info:", self.data_info)    

        self.con_loc =  [self.data.columns.get_loc(var) for var in self.con_vars]    

    def transform(self):

        self.scaler = MinMaxScaler()
        self.enc = OneHotEncoder()
        con_columns = self.scaler.fit_transform(self.data[self.con_vars])
        cat_columns = self.enc.fit_transform(self.data[self.cat_vars]).toarray()
        data_np = np.column_stack((con_columns,cat_columns))

        return data_np

    def inverse_transform(self,data):

        data_cat = self.scaler.inverse_transform(data[:,self.con_loc])
        data_cat = np.round(data_cat)
        data_con = self.enc.inverse_transform(data[:,len(self.con_loc):])       
        data_inverse = np.column_stack((data_cat,data_con))
        print("Inverse transform completed!")
        return data_inverse
'''
#CapitalGain 121 --->3  CapitalLoss  97--->3
def CapitalGainLoss(data):

    data.loc[(data["CapitalGain"] > 7647.23),"CapitalGain"] = 'high'
    data.loc[(data["CapitalGain"] == 0 ,"CapitalGain")]= 'zero'
    data.loc[operator.and_(data["CapitalGain"]!='zero', data["CapitalGain"]!='high' ),"CapitalGain"] = 'low'

    data.loc[(data["CapitalLoss"] > 1874.19),"CapitalLoss"] = 'high'
    data.loc[(data["CapitalLoss"] == 0 ,"CapitalLoss")]= 'zero'
    data.loc[operator.and_(data["CapitalLoss"]!='zero', data["CapitalLoss"]!='high'),"CapitalLoss"] = 'low'


    #NativeCountry 41---> 2
def NativeCountry(data):
    
    datai = [data]

    for dataset in datai:
        dataset.loc[dataset["NativeCountry"] != ' United-States', "NativeCountry"] = 'Non-US'
        dataset.loc[dataset["NativeCountry"] == ' United-States', "NativeCountry"] = 'US'


# MaritalStatus  7 --->2
def MaritalStatus(data):
    
    data["MaritalStatus"] = data["MaritalStatus"].replace([' Divorced',' Married-spouse-absent',' Never-married',' Separated',' Widowed'],'Single')
    data["MaritalStatus"] = data["MaritalStatus"].replace([' Married-AF-spouse',' Married-civ-spouse'],'Couple')

# Age 74 dimention   
# HoursPerWeek 96 dimention
def Discretization(data):
    data['Age']= pd.cut(data['Age'],bins=35)
    data['HoursPerWeek'] = pd.cut(data['HoursPerWeek'],bins=45)


##apply this function in adult preprocess

def adult_preprocess(data):
    CapitalGainLoss(data)
    NativeCountry(data)
    MaritalStatus(data)
    data['HoursPerWeek'] = pd.cut(data['HoursPerWeek'],bins=45,labels=False)
    con_vars = ['Age']
    return data, con_vars
    
