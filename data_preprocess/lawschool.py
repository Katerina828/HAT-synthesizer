import numpy as np
import pandas as pd
from os import path
from urllib import request
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from utils import get_data_info
from utils import save_generated_data

# only leveraging lawsch_preprocess function

def lawsch_preprocess(dataset):
    dataset.drop(['enroll', 'asian', 'black', 'hispanic', 'white', 'missingrace', 'urm'], axis=1, inplace=True)
    dataset.dropna(axis=0, inplace=True, subset=['admit'])
    dataset.replace(to_replace='', value=np.nan, inplace=True)
    dataset.dropna(axis=0, inplace=True)
    dataset = dataset[dataset['race'] != 'Asian']

    for col in dataset.columns:
        if dataset[col].isnull().sum() > 0:
            dataset.drop(col, axis=1, inplace=True)

    con_vars = ['lsat','gpa']
    return dataset,con_vars



class SyntheticLawschsDataset():
    def __init__(self):

        print("load Synthetic law school")
        df_lawsch = pd.read_csv("./GenerateData/lawschool/lawschool_syn_300_bs500_seed0_times_10.csv")
        df_lawsch = df_lawsch[:43011]
        df_lawsch['lsat'] = df_lawsch['lsat'].astype('int')
        df_lawsch['gpa'] = df_lawsch['gpa'].round(decimals=2)
        self.con_vars = ['lsat','gpa']
        self.cat_vars = [col for col in df_lawsch.columns if col not in self.con_vars]
        self.columns_name = self.con_vars + self.cat_vars
        self.data = df_lawsch[self.columns_name]
        
        self.data_info = get_data_info(self.data ,self.cat_vars)
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


class LawschsDataset():
    def __init__(self):

        print("load law school")
        data_dir = 'dataset'
        data_file = path.join(data_dir, 'lawschs1_1.dta')

        if not path.exists(data_file):
            request.urlretrieve(
                'http://www.seaphe.org/databases/FOIA/lawschs1_1.dta', data_file
            )
        dataset = pd.read_stata(data_file)
        dataset.drop(['enroll', 'asian', 'black', 'hispanic', 'white', 'missingrace', 'urm'], axis=1, inplace=True)
        dataset.dropna(axis=0, inplace=True, subset=['admit'])
        dataset.replace(to_replace='', value=np.nan, inplace=True)
        dataset.dropna(axis=0, inplace=True)
        dataset = dataset[dataset['race'] != 'Asian']

        for col in dataset.columns:
            if dataset[col].isnull().sum() > 0:
                dataset.drop(col, axis=1, inplace=True)

        self.con_vars = ['lsat','gpa']
        self.cat_vars = [col for col in dataset.columns if col not in self.con_vars]
        self.columns_name = self.con_vars + self.cat_vars
        self.data = dataset[self.columns_name]

        self.data_info = get_data_info(self.data ,self.cat_vars)
        self.con_loc = [dataset.columns.get_loc(var) for var in self.con_vars]
        
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