import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,OrdinalEncoder
from utils import get_data_info
import matplotlib.pyplot as plt
import math
import torch
import operator
'''#-----------------------------
# BaseTransformer class
# input： train dataset, pandas,
            continiours columns name: ['','','']
    Usage:
    tf = BaseTransformer(train,con_vars)
    ganinput= tf.transform()
    data_inverse = tf.inverse_transform(sample)  #convert gan samples to its original form

'''


class BaseTransformer():
    
    def __init__(self,train,con_vars):
        
        print("load Synthetic Adult...")
        #self.cat_vars = [col for col in df_health.columns if col not in self.con_vars]
        self.con_vars = con_vars
        self.cat_vars = [col for col in train.columns if col not in self.con_vars]

        self.columns_name = self.con_vars + self.cat_vars
        self.train = train[self.columns_name]
        #get data info
        self.data_info = get_data_info(self.train ,self.cat_vars)
        print("Data info:", self.data_info)    

        self.con_loc =  [self.train.columns.get_loc(var) for var in self.con_vars]    

    def transform(self):

        self.scaler = MinMaxScaler()
        self.enc = OneHotEncoder()

        con_columns = self.scaler.fit_transform(self.train[self.con_vars])
        cat_columns = self.enc.fit_transform(self.train[self.cat_vars]).toarray()
        data_train = np.column_stack((con_columns,cat_columns))


        return data_train

    def inverse_transform(self,data):

        data_con = self.scaler.inverse_transform(data[:,self.con_loc])
        #data_con = np.round(data_con)
        data_cat = self.enc.inverse_transform(data[:,len(self.con_loc):])       
        data_inverse = np.column_stack((data_con,data_cat))
        print("Inverse transform completed!")
        return data_inverse