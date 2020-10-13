import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import MinMaxScaler

#from health import process_health_per_year, discretization

def model_eval(y_test,y_test_pred,y_train,y_train_pred):
    out={}
    out['train_acc'] =  metrics.accuracy_score(y_train, y_train_pred)
    out['test_acc'] =  metrics.accuracy_score(y_test, y_test_pred)
    out['precision'] = metrics.precision_score(y_test, y_test_pred)
    out['recall'] = metrics.recall_score(y_test, y_test_pred)
    out['f1_score'] = metrics.f1_score(y_test, y_test_pred)
    
    return out

def train(x,y):
    print("Ada training...")
    #adaboost
    ada = AdaBoostClassifier(n_estimators=100) 
    ada.fit(x, y)

    print("decisiontree training...")
    #decisiontree
    #DT = DecisionTreeClassifier(criterion = 'entropy')
    DT = DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 0.05, min_samples_leaf = 0.001)
    #DT = DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 0.05, min_samples_leaf = 0.001)
    DT.fit(x,y)

    print("randomforeast training...")
    #randomforeast
    rf = RandomForestClassifier(n_estimators =100, min_samples_split = 0.05, min_samples_leaf = 0.001, n_jobs=4)
    rf.fit(x, y)

    print("MLPClassifier...")
    #MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(50, ),max_iter = 500)
    mlp.fit(x, y)

    
    print("logistic regression training ... ")
    #logistic regression
    #log_reg = LogisticRegression(solver = 'lbfgs',n_jobs=2,class_weight = 'balanced', max_iter = 200)
    log_reg = LogisticRegression(penalty = 'l2', dual = False, tol = 1e-4, fit_intercept = True, 
                            solver = 'liblinear')
    log_reg.fit(x,y)
 
    
    
    return DT,rf,mlp,ada,log_reg

    
def evaluation(X_train, y_train,x_test,y_test,DT,rf,mlp,ada,log_reg):
    #adaboost
    ada_train_pred = ada.predict(X_train)
    ada_test_pred = ada.predict(x_test)
    ada_result= model_eval(y_test, ada_test_pred,y_train,ada_train_pred)
    ada_table = round(pd.DataFrame([ada_result], index = ['AdaBoost']),4)


    #decisiontree
    dt_train_pred = DT.predict(X_train)
    dt_test_pred = DT.predict(x_test)
    dt_result = model_eval(y_test, dt_test_pred,y_train,dt_train_pred)
    dt_table = round(pd.DataFrame([dt_result], index = ['DecisionTree']),4)
    
    #randomforeast
    rf_train_pred = rf.predict(X_train)
    rf_test_pred  = rf.predict(x_test)
    rf_result= model_eval(y_test, rf_test_pred,y_train, rf_train_pred)
    rf_table = round(pd.DataFrame([rf_result], index = ['RandomForest']),4)
    
    #MLPClassifier
    mlp_train_pred = mlp.predict(X_train)
    mlp_test_pred  = mlp.predict(x_test)
    mlp_result= model_eval(y_test, mlp_test_pred,y_train,mlp_train_pred)
    mlp_table = round(pd.DataFrame([mlp_result], index = ['MLPC']),4)
    
    #logisticregression
    log_train_pred = log_reg.predict(X_train)
    log_reg_test_pred = log_reg.predict(x_test)
    log_reg_result= model_eval(y_test, log_reg_test_pred,y_train,log_train_pred)
    log_reg_table = round(pd.DataFrame([log_reg_result], index = ['LogisticRegression']),4)

    return dt_table,rf_table,mlp_table,ada_table,log_reg_table
    

def main(X_train,y_train,X_test,y_test):


    DT,rf,mlp,ada,log_reg = train(X_train,y_train)
    dt_table,rf_table,mlp_table,ada_table,log_reg_table = evaluation(X_train,y_train,X_test,y_test,DT,rf,mlp,ada,log_reg)
    overall_eval = pd.concat([ dt_table,rf_table,mlp_table,ada_table,log_reg_table], axis = 0)
    return overall_eval

    #overall_eval.sort_values(by = ['f1_score', 'accuracy'], ascending = False, inplace = True)

def Adult_ML_eval(base=1):
    if base:
        train = pd.read_csv("./GenerateData/Adult/Adult_tain.csv")
    else:
        train = pd.read_csv("./GenerateData/Adult/Adult_syn_gan.csv")
    test = pd.read_csv("./GenerateData/Adult/Adult_test.csv")
    y_train = train['Income']
    X_train = train.drop('Income',axis=1)
    y_test = test['Income']
    X_test = test.drop('Income',axis=1)
    overall_eval = main(X_train,y_train,X_test,y_test)
    print(overall_eval)

def load_lawschool():
    train = pd.read_csv("./GenerateData/lawschool/lawschool_tain.csv")
    test = pd.read_csv("./GenerateData/lawschool/lawschool_test.csv")
    y_train = train['admit']
    X_train = train.drop('admit',axis=1)
    y_test = test['admit']
    X_test = test.drop('admit',axis=1)

    scaler = MinMaxScaler()

    X_train[['lsat','gpa']] = scaler.fit_transform(X_train[['lsat','gpa']])
    X_train = pd.get_dummies(X_train, columns=['race','college','year', 'gender','resident'], prefix_sep='=')
    X_test[['lsat','gpa']] = scaler.fit_transform(X_test[['lsat','gpa']])
    X_test = pd.get_dummies(X_test, columns=['race','college','year', 'gender','resident'], prefix_sep='=')
    return X_train,y_train,X_test,y_test

def load_health():
    df_health = pd.read_csv('./dataset/health_without_year.csv')
    #df_health = pd.read_csv('./dataset/health_full.csv')
    
    '''
    health_Y1 = df_health.loc[(df_health['Year']=='Y1')]
    health_Y2 = df_health.loc[(df_health['Year']=='Y2')]
    health_Y3 = df_health.loc[(df_health['Year']=='Y3')]
    '''
    #choose one year data
    health_new = process_health_per_year(df_health)

    label = health_new['max_CharlsonIndex']
    features = health_new.drop(['max_CharlsonIndex'], axis=1)

    #discretize features with "=" in columns
    discretization(features)
        
    #continous columns
    c_vars = ['LabCount_total','LabCount_months','DrugCount_total','DrugCount_months'
          ,'PayDelay_total','PayDelay_max','PayDelay_min']

    scaler = MinMaxScaler()
    features[c_vars] = scaler.fit_transform(features[c_vars])

    cat_vars = [col for col in features.columns if '=' in col]
    #c_vars.extend(ex_vars)

    var_ex = ['AgeAtFirstClaim','Sex']
    cat_vars.extend(var_ex)
    
    features = pd.get_dummies(features, columns=cat_vars, prefix_sep='=')

    X_train, X_test, y_train, y_test = train_test_split(features ,label, test_size=0.3, random_state=2)
    return X_train, X_test, y_train, y_test 


np.concatenate