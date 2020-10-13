import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from collections import Counter
import math
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from MLclassifier import model_eval,train,evaluation,main
from sklearn.preprocessing import OrdinalEncoder
from GanSythesizer import GanSythesizer

#dataset need to be segmented into n sub-dataset, each with len=perlen
def cut_dataset(dataset,n,perlen):
    np.random.seed(0)
    sub_fake = {}
    rand_ind = np.random.permutation(perlen*n)
    for i in range(n):
        indices = rand_ind[n*i : n*i + perlen]
        sub_fake[i] = dataset.iloc[indices]
    return sub_fake
    

def unique_exam(data):
    uniques,counts = np.unique(data.values,axis=0,return_counts=True)
    return Counter(counts).most_common(80), uniques

def load_real_fake():
    real = pd.read_csv('./GenerateData/Adult/adult_syn_seed1.csv') 
    real = real[:31655]
    fake = pd.read_csv("./GenerateData/Adult/Adult_syn_syn_300_bs500_seed0_times_10.csv")
    return real,fake

def compute_intersection(data1,data2):
        x = np.concatenate((data1, data2), axis=0)
        _,index,counts = np.unique(x,axis=0,return_index = True,return_counts=True)
        #print('重复次数:',Counter(counts))
        idx = []
        for i,count in enumerate(counts):
            if count==2:
                idx.append(index[i])
        duplicate = x[idx]
        return duplicate

def add_true_label(real,fake):
    unique, inverse_index,count = np.unique(fake,axis=0, return_inverse=True, return_counts=True)
    print(unique.shape)
    fake_new = pd.DataFrame(unique[inverse_index],columns = real.columns)
    fake_new['freq'] = count[inverse_index].reshape(-1,1)

    real_con_fake  = np.concatenate((fake,real),axis=0)
    _, inverse_index2,count2 = np.unique(real_con_fake,axis=0, return_inverse=True, return_counts=True)
    fake_new['counts_in_real'] = count2[inverse_index2].reshape(-1,1)[:fake_new.shape[0]] - count[inverse_index].reshape(-1,1)

    _,uniques_real = unique_exam(real)
    #add real label
    real_part = compute_intersection(unique,uniques_real)
    test_data = np.concatenate((unique,real_part), axis=0) 
    _,count2 = np.unique(test_data,axis=0, return_counts=True)
    d_count = count2-1
    real_label = (d_count[inverse_index] >0)
    fake_new['label'] = real_label
    fake_new['label'] = fake_new['label'].astype('int')
    return fake_new

def train_classifier(train):
    train = train.drop(columns=['counts_in_real'])
    y_train = train.pop('label')
    X_train = train.values
    mlp = MLPClassifier(hidden_layer_sizes=(100, ),max_iter = 200)
    mlp.fit(X_train, y_train)
    return mlp


def multi_attack_model(data,n=10):
    if data=="adult":
        real = pd.read_csv('./GenerateData/Adult/Adult_train.csv')
        syn = pd.read_csv("./GenerateData/Adult/adult_syn_seed1.csv")
        syn = syn[:len(real)*n]
        data = pd.concat([real, syn])
    elif data == "lawsch":
        real = pd.read_csv('./GenerateData/lawschool/lawschool_train.csv')
        syn = pd.read_csv("./GenerateData/lawschool/lawschool_syn_300_bs500_seed0_times_10.csv")
        syn = syn[:len(real)*n]
        data = pd.concat([real, syn])
        data['lsat'] = data['lsat'].astype('int')
        data['gpa'] = data['gpa'].round(decimals=2)
        discretizer = KBinsDiscretizer(n_bins=60, encode='ordinal', strategy='uniform')
        data['gpa'] = discretizer.fit_transform(data.values[:,1:2])
    elif data =='compas':
        dis_vars = ['diff_custody','diff_jail']
        cat_vars= ['sex','race','juv_fel_count','c_charge_degree','v_score_text', 'two_year_recid']
        real = pd.read_csv("./GenerateData/Compas/Compas_train.csv")
        fake = pd.read_csv("./GenerateData/Compas/Compas_syn600_bs100_seed1_times_10.csv")
        fake.loc[(fake["diff_jail"] < 0 ,"diff_jail")]= 0
        fake = fake[:len(real)*n]
        data = pd.concat([real, fake])
        discretizer = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
        data[dis_vars] = discretizer.fit_transform(data[dis_vars])
        enc = OrdinalEncoder()
        data[cat_vars] = enc.fit_transform(data[cat_vars])
    Real = data[:len(real)]
    Syn = data[len(real):] 
    sub_fake = {}
    rand_ind = np.random.permutation(len(Real)*n)
    for i in range(n):
        indices = rand_ind[n*i : n*i + len(real)]
        sub_fake[i] = Syn.iloc[indices]
    
    for i in range(n):
        gan = GanSythesizer(epochs = 600,seed=1,DP = False,batch_size = 100)
        myD = gan.fit(data ='synthetic',del_=False)     



class attack_experiment():
    def __init__(self,data,size=5):
        #load data
        if data =='lawschool':
            print("load lawschool dataset...")
            real = pd.read_csv("./GenerateData/lawschool/lawschool_train.csv")
            fake1 = pd.read_csv("./GenerateData/lawschool/lawschool_syn_300_bs500_seed0_times_10.csv")
            fake1['lsat'] = fake1['lsat'].astype('int')
            fake1['gpa'] = fake1['gpa'].round(decimals=2)
            data = pd.concat([real, fake1])
            #discretize data
            discretizer = KBinsDiscretizer(n_bins=60, encode='ordinal', strategy='uniform')
            data.values[:,1:2] = discretizer.fit_transform(data.values[:,1:2])
            data_size = len(real)
            self.real_new  = data[:data_size]
            self.fake_new= data[data_size:]
            self.fake1_new = self.fake_new[0: data_size*size]
            print("fake size:",self.fake1_new.shape[0])

            #self.fake1_new = self.fake_new[1: math.ceil(self.fake_new.shape[0]/5)*1]
            counter, _=self.unique_exam(self.real_new)
            print("Real: unique per:%.4f \t less than three:%.4f" % 
                (counter[0][1]/self.real_new.shape[0],(counter[0][1]+counter[1][1]+counter[2][1])/self.real_new.shape[0]))
            self.counter, _=self.unique_exam(self.fake1_new)
            print("Syn: unique per:%.4f \t less than three:%.4f" % 
                (self.counter[0][1]/self.fake1_new.shape[0],(self.counter[0][1]+self.counter[1][1]+self.counter[2][1])/self.fake1_new.shape[0]))
            

        elif data =='Adult':
            print("load Adult dataset...")
            real = pd.read_csv("./GenerateData/Adult/Adult_train.csv").astype('int')
            fake = pd.read_csv("./GenerateData/Adult/Adult_syn_300_bs500_seed1_DP2_times_10.csv")
            data = pd.concat([real, fake])
            #discretizer = KBinsDiscretizer(n_bins=40, encode='ordinal', strategy='uniform')
            #data.values[:,0:1] = discretizer.fit_transform(data.values[:,0:1])
            data_size = len(real)
            self.real_new  = data[:data_size]
            self.fake_new= data[data_size:]
            self.fake1_new = self.fake_new[1: math.ceil(self.fake_new.shape[0]/10)*size]

            
            counter, _=self.unique_exam(self.real_new)
            print("Real: unique per:%.4f \t less than three:%.4f" % 
                (counter[0][1]/self.real_new.shape[0],(counter[0][1]+counter[1][1]+counter[2][1])/self.real_new.shape[0]))
            self.counter, _=self.unique_exam(self.fake1_new)
            print("Syn: unique per:%.4f \t less than three:%.4f" % 
                (self.counter[0][1]/self.fake1_new.shape[0],(self.counter[0][1]+self.counter[1][1]+self.counter[2][1])/self.fake1_new.shape[0]))
            
        elif data =='health':
            real = pd.read_csv("./GenerateData/health/health_tain.csv")
            fake = pd.read_csv("./GenerateData/health/health_fake_sigmoid.csv")
            data = pd.concat([real, fake])
            discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
            data.values[:,0:7] = discretizer.fit_transform(data.values[:,0:7])

            self.real_new  = data[:real.shape[0]]
            self.fake1_new = data[real.shape[0]:]

            unique, _,_ = np.unique(self.real_new,axis=0, return_inverse=True, return_counts=True)
            print("Unique per in real data:", unique.shape[0]/self.real_new.shape[0])
            unique, _,_ = np.unique(self.fake1_new,axis=0, return_inverse=True, return_counts=True)
            print("Unique per in synthetic data:", unique.shape[0]/self.fake1_new.shape[0])

        elif data == "compas":
            real = pd.read_csv("./GenerateData/Compas/Compas_train.csv")
            fake = pd.read_csv("./GenerateData/Compas/compas_syn_600_bs100_seed1_times_10.csv")
            fake.loc[(fake["diff_jail"] < 0 ,"diff_jail")]= 0
            data = pd.concat([real, fake])

            #discretizer = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='uniform')
            #data.values[:,0:1] = discretizer.fit_transform(data.values[:,0:1])

            discretizer = KBinsDiscretizer(n_bins=60, encode='ordinal', strategy='uniform')
            data.values[:,1:3] = discretizer.fit_transform(data.values[:,1:3])
            data_size = len(real)
            self.real_new  = data[:data_size]
            self.fake_new= data[data_size:]
            self.fake1_new = self.fake_new[0: data_size*size]
            print("fake size:",self.fake1_new.shape[0])

            counter, _=self.unique_exam(self.real_new)
            print("Real: unique per:%.4f \t less than three:%.4f" % 
                (counter[0][1]/self.real_new.shape[0],(counter[0][1]+counter[1][1]+counter[2][1])/self.real_new.shape[0]))
            self.counter, _=self.unique_exam(self.fake1_new)
            print("Syn: unique per:%.4f \t less than three:%.4f" % 
                (self.counter[0][1]/self.fake1_new.shape[0],(self.counter[0][1]+self.counter[1][1]+self.counter[2][1])/self.fake1_new.shape[0]))
            




    def add_true_label(self):
        fake_new = self.fake1_new 
        unique, inverse_index,count = np.unique(fake_new,axis=0, return_inverse=True, return_counts=True)
        print(unique.shape)
        self.fake = pd.DataFrame(unique[inverse_index],columns = self.real_new.columns)
        self.fake['freq'] = count[inverse_index].reshape(-1,1)

        real_con_fake  = np.concatenate((fake_new,self.real_new),axis=0)
        _, inverse_index2,count2 = np.unique(real_con_fake,axis=0, return_inverse=True, return_counts=True)
        self.fake['counts_in_real'] = count2[inverse_index2].reshape(-1,1)[:self.fake.shape[0]] - count[inverse_index].reshape(-1,1)

        _,uniques_real = self.unique_exam(self.real_new)
        #add real label
        real_part = self.compute_intersection(unique,uniques_real)
        test_data = np.concatenate((unique,real_part), axis=0) 
        _,count2 = np.unique(test_data,axis=0, return_counts=True)
        d_count = count2-1
        real_label = (d_count[inverse_index] >0)
        self.fake['true_label'] = real_label
        self.fake['true_label'] = self.fake['true_label'].astype('int')
        #print(self.fake['true_label'].value_counts())

        #add guess label
        #freq_thre = self.fake['freq'].quantile(0.79)

    def add_guess_label(self,threshold):
        freq_thre = threshold
        table = pd.DataFrame([freq_thre],columns=['Thre'])
        #
        #self.fake['freq'].quantile(0.79)
        self.fake['guess_label'] = (self.fake['freq'] > freq_thre).astype('int')
        """
        _,uniques_fake2 = self.unique_exam(self.fake2_new)
        guess_real_part = self.compute_intersection(unique,uniques_fake2)
        guess_test_data = np.concatenate((unique,guess_real_part), axis=0)
        _,guess_count2 = np.unique(guess_test_data,axis=0, return_counts=True)
        guess_d_count = guess_count2-1
        guess_label = (guess_d_count[inverse_index]>0)
        """
        #sns.heatmap(self.fake[['freq','guess_label','true_label']].corr(), square=True,cmap="YlGnBu",annot=True)
        #print("true label distribution:", self.fake['ture_label'].value_counts()/self.fake['ture_label'].shape[0]*100)

        true = self.fake[self.fake['guess_label']==1]
        true['compare']= true[["true_label","guess_label"]].apply(lambda x: x["true_label"] == x["guess_label"],axis=1)
        result = true['compare'].value_counts()


        table['uniques']  = round(true['counts_in_real'].value_counts()[1]/true.shape[0]*100,4)
        table['LessThan3'] = round((true['counts_in_real'].value_counts()[1]+true['counts_in_real'].value_counts()[2]+
                                    true['counts_in_real'].value_counts()[3])/true.shape[0]*100,4)
        
        table['num'] = true.shape[0]
        table['adv']= round(result[1]/true.shape[0]*100,4)
        return true,table


    
    def classifier(self, drop_count=0,adv_flag=0,seed=0 ):
        self.fake = self.fake.drop(columns=['counts_in_real'])
        if drop_count ==1:
            self.fake = self.fake.drop(columns=['freq'])
        r_label = self.fake.pop('true_label')
        g_label = self.fake.pop('guess_label')
        features = self.fake.values
        if adv_flag:
            
            X_train, _, y_train, _ = train_test_split(features ,g_label, test_size=0.3, random_state=seed)
            _, X_test, _, y_test = train_test_split(features ,r_label, test_size=0.3, random_state=seed)
        else:

            X_train, X_test, y_train, y_test = train_test_split(features ,r_label, test_size=0.3, random_state=seed)
        
        print(X_train.shape,X_test.shape)
        print(pd.Series(y_train).value_counts(),pd.Series(y_test).value_counts())
        print(pd.Series(y_train).value_counts()[1]/X_train.shape[0],pd.Series(y_test).value_counts()[1]/X_test.shape[0])
        #overall_eval = main(X_train,y_train,X_test,y_test)
        
        result = []
        for i in range(10):
            mlp = MLPClassifier(hidden_layer_sizes=(100, ),max_iter = 200)
            mlp.fit(X_train, y_train)
            y_predict = mlp.predict(X_test)
            result.append(self.model_eval(y_test, y_predict))
        
        avg = self.compute_avg(result)
        
        
        return avg
        


    @staticmethod
    def unique_exam(data):
        uniques,counts = np.unique(data.values,axis=0,return_counts=True)
        return Counter(counts).most_common(80), uniques

    @staticmethod
    def compute_intersection(data1,data2):
        x = np.concatenate((data1, data2), axis=0)
        _,index,counts = np.unique(x,axis=0,return_index = True,return_counts=True)
        #print('重复次数:',Counter(counts))
        idx = []
        for i,count in enumerate(counts):
            if count==2:
                idx.append(index[i])
        duplicate = x[idx]
        return duplicate

    @staticmethod
    def model_eval(actual,pred):
        out={}
        out['accuracy']  =  round(metrics.accuracy_score(actual, pred)*100,2)
        out['balanced_acc']  =  round(metrics.balanced_accuracy_score(actual, pred)*100,2)
        out['precision'] =  round(metrics.precision_score(actual, pred)*100,2)
        out['recall']    =  round(metrics.recall_score(actual, pred)*100,2)
        out['f1_score']  =  round(metrics.f1_score(actual, pred)*100,2)
    
        return out


    @staticmethod
    def compute_avg(result):
        avg={}
        avg['accuracy']=0
        avg['precision'] = 0
        avg['recall'] =0
        avg['f1_score'] =0
        avg["balanced_acc"]=0
        for i in range(len(result)):
            avg['accuracy'] += result[i]['accuracy']
            avg['balanced_acc'] += result[i]['balanced_acc']
            avg['precision'] += result[i]['precision']
            avg['recall'] += result[i]['recall']
            avg['f1_score'] += result[i]['f1_score']
        avg['accuracy'] = round(avg['accuracy'] /len(result),2)
        avg['balanced_acc'] =round(avg['balanced_acc'] /len(result),2)
        avg['precision'] = round(avg['precision'] /len(result),2)
        avg['recall'] = round(avg['recall'] /len(result),2)
        avg['f1_score'] = round(avg['f1_score'] /len(result),2)
        return avg

    @staticmethod
    def plot_roc(labels, predict_prob):
        false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, predict_prob)
        roc_auc=auc(false_positive_rate, true_positive_rate)
        plt.title('ROC')
        plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.ylabel('TPR')
        plt.xlabel('FPR')

def findByRow(mat, row):
    return np.where((mat == row).all(1))[0]

def compute_counts_in_real(fake, real):
    count = []
    i=0
    for item in fake:
        x = findByRow(real.values,item)
        c = x.shape[0]
        if c is not None:
            i = i+1
        count.append(c)
    c_ = np.array(count).reshape(-1,1)
    fake_new = np.concatenate((fake,c_),axis=1)
    fake_new_pd = pd.DataFrame(fake_new,index=None,columns=['lsat', 'gpa', 'race', 'college', 'year', 'gender', 'resident',
       'admit','count'])
    return fake_new,fake_new_pd, i/fake.shape[0]


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
