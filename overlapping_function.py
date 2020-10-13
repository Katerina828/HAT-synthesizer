import pandas as pd
import numpy as np
import statsmodels as sm
import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import math
import sklearn.metrics as metrics
import sklearn.tree as tree
import seaborn as sns
import operator
from collections import Counter

#locate every row in mat
def findByRow(mat, row):
    return np.where((mat == row).all(1))[0]


#compute the uniqueness of database
def unique_exam(data):
    uniques,index,counts = np.unique(data.values,axis=0,return_index = True,return_counts=True)
    return Counter(counts).most_common(80), uniques


#compute the overlapping of two datasets, return intersection data duplicate
def compute_intersection(data1,data2):
    x = np.concatenate((data1, data2), axis=0)
    uniques,index,counts = np.unique(x,axis=0,return_index = True,return_counts=True)
    idx = []
    for i,count in enumerate(counts):
        if count==2:
            idx.append(index[i])
    duplicate = x[idx]
    return duplicate


#add label "freq" and "real_data", return a labeled synthetic database
def add_label(real,fake):
    fake_new = fake 
    unique, inverse_index,count = np.unique(fake_new,axis=0, return_inverse=True, return_counts=True)
    fake_label = pd.DataFrame(unique[inverse_index],columns = real.columns)
    fake_label['freq'] = count[inverse_index].reshape(-1,1)

    _,uniques_real = unique_exam(real)
        #add real label
    real_part = compute_intersection(unique,uniques_real)
    test_data = np.concatenate((unique,real_part), axis=0) 
    _,count2 = np.unique(test_data,axis=0, return_counts=True)
    d_count = count2-1
    real_label = (d_count[inverse_index] >0)
    fake_label['real_data'] = real_label
    fake_label['real_data'] = fake_label['real_data'].astype('int')
    return fake_label
    

#find the best frequency threshold
def find_best_freq_thre(fake_label):
    max_ = 0
    freq_ =0

    for i in range(1,fake_label['freq'].max()-1):
        fake_label_high = fake_label[(fake_label['freq']> i )] 
        fake_label_low = fake_label[(fake_label['freq']<= i)]
        per = fake_label_high.shape[0]/(fake_label_high.shape[0]+ fake_label_low.shape[0])
        x_high = fake_label_high['ture_label'].value_counts(normalize=True)[1]
        x_low = fake_label_low['ture_label'].value_counts(normalize=True)[0]
        item = x_high*per+x_low*(1-per)
        if (item>max_):
            max_ = item
            freq_ = i
    return max_,freq_ 


#compute overlapping of 5 database
def compute_overlapping(real, fake_times10):
    data = fake_times10
    len_ = real.shape[0]
    fake1= data[0:len_]
    fake2 = data[len_:len_*2]
    fake3 = data[len_*2:len_*3]
    fake4 = data[len_*3:len_*4]
    fake5 = data[len_*4:len_*5]
    
    info_real,uniques_real = unique_exam(real)
    
    info_fake1,uniques_fake1 = unique_exam(fake1)
    info_fake2,uniques_fake2 = unique_exam(fake2)
    info_fake3,uniques_fake3 = unique_exam(fake3)
    info_fake4,uniques_fake4 = unique_exam(fake4)
    info_fake5,uniques_fake5 = unique_exam(fake5)
    
    du_fake12 = compute_intersection(uniques_fake1,uniques_fake2)
    real_of_du_fake12 = compute_intersection(uniques_real,du_fake12)
    print("Number of intersection:{}, real:{}({})"
      .format(du_fake12.shape,real_of_du_fake12.shape[0],round(real_of_du_fake12.shape[0]/du_fake12.shape[0]*100,2)))

    du_fake123 = compute_intersection(du_fake12,uniques_fake3)
    real_of_du_fake123 = compute_intersection(uniques_real,du_fake123)
    print("Number of intersection:{}, real:{}({})"
      .format(du_fake123.shape,real_of_du_fake123.shape[0],round(real_of_du_fake123.shape[0]/du_fake123.shape[0]*100,2)))
    
    du_fake1234 = compute_intersection(du_fake123,uniques_fake4)
    real_of_du_fake1234 = compute_intersection(uniques_real,du_fake1234)
    print("Number of intersection:{}, real:{}({})"
      .format(du_fake1234.shape,real_of_du_fake1234.shape[0],round(real_of_du_fake1234.shape[0]/du_fake1234.shape[0]*100,2)))

    du_fake12345 = compute_intersection(du_fake1234,uniques_fake5)
    real_of_du_fake12345 = compute_intersection(uniques_real,du_fake12345)
    print("Number of intersection:{}, real:{}({})"
      .format(du_fake12345.shape[0],real_of_du_fake12345.shape[0],round(real_of_du_fake12345.shape[0]/du_fake12345.shape[0]*100,2)))
    
    return real_of_du_fake12345,round(real_of_du_fake12345.shape[0]/du_fake12345.shape[0]*100,2)