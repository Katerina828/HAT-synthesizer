import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import KBinsDiscretizer
from os import path
from urllib import request
import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_compas_CDF(real,syn):
    sns.set(style="ticks", color_codes=True,font_scale=1.5)
    columns=['age','diff_custody','diff_jail','priors_count']
    #real['label']=1
    #syn['label']= 0
    #df = pd.concat([real,syn],axis=0)
    fig = plt.figure(figsize=(30,6))
    cols = 4
    rows = math.ceil(float(len(columns)) / cols)
    for i,info in enumerate(columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        #fig, ax = plt.subplots(figsize=(8, 6))
        x= np.sort(real[info])
        x2= np.sort(syn[info])
        y = np.arange(1,len(x)+1)/len(x)
        _ = ax.plot(x,y,marker='.',markeredgewidth='5', linestyle ='none',label='real')
        _ = ax.plot(x2,y,marker='.', markeredgewidth='5',linestyle ='none',label='fake')
        _ = ax.set_xlabel(info, fontsize=30)
        _ = ax.set_ylabel('percentage',fontsize=30)
        ax.grid(True)
        ax.legend(loc='lower right',fontsize=30)
        ax.margins(0.02)
    #plt.subplots_adjust(hspace=0.7, wspace=0.2)
    
    plt.tight_layout()
    plt.savefig("./figure/"+"Compas/"+ "compas.eps",dpi=600)
    plt.show() 

def plot_ECDF(real,syn,data_info):
        sns.set(style="ticks", color_codes=True,font_scale=1.5)
        
        #real  = pd.DataFrame(self.train_inverse,index=None,columns = self.tf.columns_name)
        #sample = self.sample(n=real.shape[0])
        #syn = pd.DataFrame(sample,index=None,columns = self.tf.columns_name)

        #real['label']=1
        #syn['label']= 0
        #df = pd.concat([real,syn],axis=0)

        for info in data_info:
            fig, ax = plt.subplots(figsize=(8, 5))
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            x= np.sort(real[info[2]])
            x2= np.sort(syn[info[2]])
            y = np.arange(1,len(x)+1)/len(x)
            _ = ax.plot(x,y,marker='.',markeredgewidth='4', linestyle ='none',label='real')
            _ = ax.plot(x2,y,marker='.',markeredgewidth='5', linestyle ='none',label='fake')
            _ = ax.set_xlabel(info[2], fontsize=30)
            _ = ax.set_ylabel('percentage',fontsize=30)
            ax.grid(True)
            ax.legend(loc='lower right',fontsize=30)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            plt.margins(0.02)
            #plt.savefig("./figure/"+ str(info[2]) +".png",dpi=600)
            plt.show()  


def plot_distribution(data,data_info):

    fig = plt.figure(figsize=(20,15))   #设定图像尺寸
    cols = 5
    rows = math.ceil(float(data.shape[1]) / cols)   #train_df.shape[1]=15, math.ceil将数字四舍五入到大的整数，所以，rows= 3
    for i, info in enumerate(data_info):
        ax = fig.add_subplot(rows, cols, i + 1)
        #ax.set_title(info[2])
        if info[1] == 'tanh':
            sns.distplot(data[info[2]],kde = False)
            #data[info[2]].hist(axes=ax,bins=100)
            #plt.xticks(rotation="vertical")
        elif info[1] == 'softmax':
            sns.countplot(data[info[2]])
           
    plt.subplots_adjust(hspace=0.7, wspace=0.2)
    plt.show()
    #fig.savefig('./MySythesizer20200301/figure/health_distribution.png',dpi=600)


#plot catagorical values
def plot_distribution_real_vs_fake(real,fake,cat_vars,file_name):
    
    real['data']=1
    fake['data']=0
    df = pd.concat([real,fake],axis =0)
    sns.set(style="ticks", color_codes=True,font_scale=1.5)
    fig = plt.figure(figsize=(20,10))
    cols = 5
    rows = math.ceil(float(df.shape[1]) / cols)
    for i,column in enumerate(cat_vars): 
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_yticks([])
        sns.countplot(x=column,data = df, hue="data",ax=ax)

        #plt.grid(True)

    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.savefig("./figure/"+ file_name +".png",dpi=600)

def plot_adult_distribution(real,fake):
    sns.set(style="ticks", color_codes=True,font_scale=1.5)
    con_vars = ['Age']
    cat_vars = [columns for columns in fake.columns if columns not in con_vars]

    real['label']=1
    fake['label']= 0
    df = pd.concat([real,fake],axis=0)

    for column in  fake.columns:
        if column in con_vars:
            #sns.distplot(fake[column],bins=100,kde = False,label='fake')
            #sns.distplot(real[column],bins=100,kde = False,label='real')
            #plt.legend(loc='upper left')
            g= sns.FacetGrid(df,hue="real",col_wrap=3,
                height=3.5, aspect=1.2)
            g.map(sns.distplot, column, hist=False, rug=True)
            plt.show()
        else:
            sns.countplot(x=column,data = df, hue="label")
            plt.show()

def plot_health_distribution(real,fake):
    sns.set(style="ticks", color_codes=True,font_scale=1.5)
    con_vars = ['LabCount_total','LabCount_months','DrugCount_total','DrugCount_months'
          ,'PayDelay_total','PayDelay_max','PayDelay_min']
    cat_vars = [columns for columns in fake.columns if columns not in con_vars]

    real['label']=1
    fake['label']= 0
    df = pd.concat([real,fake],axis=0)

    for column in  fake.columns:
        if column in con_vars:
            sns.distplot(fake[column],bins=100,kde = False,label='fake')
            sns.distplot(real[column],bins=100,kde = False,label='real')
            plt.legend(loc='upper right')
            
            plt.show()
        else:
            sns.countplot(x=column,data = df, hue="label")
            plt.show()

def plot_compas_distribution(real,fake):
    sns.set(style="ticks", color_codes=True,font_scale=1.5)
    con_vars = ['age','diff_custody','diff_jail','priors_count']
    #cat_vars = [columns for columns in fake.columns if columns not in con_vars]

    real['label']=1
    fake['label']= 0
    df = pd.concat([real,fake],axis=0)

    for column in  fake.columns:
        if column in con_vars:
            sns.distplot(fake[column],bins=100,kde = False,label='fake')
            sns.distplot(real[column],bins=100,kde = False,label='real')
            plt.legend(loc='upper right')
            
            plt.show()
        else:
            sns.countplot(x=column,data = df, hue="label")
            plt.show()

def plot_lawshcool_distribution(real,fake):
    sns.set(style="ticks", color_codes=True,font_scale=1.5)
    con_vars = ['lsat','gpa']
    cat_vars = [columns for columns in fake.columns if columns not in con_vars]

    real['label']=1
    fake['label']= 0
    df = pd.concat([real,fake],axis=0)

    for column in  fake.columns:
        if column in con_vars:
            sns.distplot(fake[column],bins=100,kde = False,label='fake')
            sns.distplot(real[column],bins=100,kde = False,label='real')
            plt.legend(loc='upper left')
            #plt.savefig('./figure/lawschool/'+column + '.png', dpi=600,bbox_inches = 'tight')
            plt.show()
        else:
            sns.countplot(x=column,data = df, hue="label")
            plt.show()


def plot_vae_loss(kl_loss_vector,recon_loss_vector,loss_vector):
    sns.set(style="ticks", color_codes=True,font_scale=1.5)
    plt.plot(kl_loss_vector,'b',label='KL')
    plt.plot(recon_loss_vector,'g',label='recon')
    plt.plot(loss_vector,'r',label ='sum')
    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title('loss')
    plt.show()


def plot_gan_loss(G_loss,C_loss):
    sns.set(style="ticks", color_codes=True,font_scale=1.5)
    plt.plot(G_loss,'b',label='G_loss')
    plt.plot(C_loss,'r',label ='C_loss')
    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title('loss')
    plt.show()



 