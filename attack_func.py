from sklearn import svm, datasets
from sklearn.neural_network import MLPClassifier
from Attack_experiment import unique_exam,compute_intersection,add_true_label
from sklearn import metrics
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
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

def train_classifier(train):
    train = train.drop(columns=['counts_in_real'])
    y_train = train.pop('label')
    X_train = train.values
    #classifier = svm.LinearSVC(random_state=random_state)
    #classifier.fit(X_train, y_train)
    classifier = MLPClassifier(hidden_layer_sizes=(100, ),max_iter = 200)
    classifier.fit(X_train, y_train)
    
    return classifier

def predict_prob(model,test):
    test  = test.drop(columns=['counts_in_real'])
    y_test = test.pop('label')
    X_test = test.values
    y_score = model.predict_proba(X_test)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    AUC = metrics.auc(recall,precision)
    return precision, recall, thresholds,AUC
    
def ordinal_encode(enc,data):
    colomns = [col for col in data.columns if data[col].dtype=="object"]
    data[colomns] = enc.fit_transform(data[colomns])
    return data

def cut_dataset(dataset,n,perlen):
    np.random.seed(3000)
    sub_fake = []
    rand_ind = np.random.permutation(perlen*n)
    for i in range(n):
        indices = rand_ind[perlen*i : perlen*i + perlen]
        sub_fake.append(dataset.iloc[indices]) 
    return sub_fake
    
def sample_n(precision, recall, n=200):
    samp_list = np.linspace(0,precision.shape[0]-1,n).astype('int')
    x = np.column_stack((precision,recall))
    return x[samp_list]

    


    