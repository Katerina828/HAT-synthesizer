import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import math
import random
import time
from base import BaseSynthesizer
from GanSythesizer import Critic,Residual,Generator,gradient_penalty_compute,apply_activate
from data_preprocess.adult import adult_preprocess
from data_preprocess.transformer import BaseTransformer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from utils import set_seed,get_data_info

'''
main class for synthesizing
Usage:
synthesizer = Synthesizer()
synthesizer.fit(data,data_info)
samples = synthesizer.sample(n,seed=0)
'''

class Synthesizer(BaseSynthesizer):
    def __init__(self, 
                 embedding_dim = 128,
                 gen_dim =(256, 256),
                 dis_dim = (256,256),
                 l2scale = 1e-6,
                 batch_size = 100,
                 epochs = 150,
                 sigma = 5.0,
                 DP = False,
                 seed = 0
                 ):
        self.embedding_dim = embedding_dim
        self.dis_dim = dis_dim
        self.gen_dim = gen_dim
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.DP = DP
        self.seed = seed
        if self.DP == True:
            self.cg = 1.0
            self.sigma = sigma
            self.noise_multiplier = self.sigma/self.cg
            print("noise_multiplier:", self.noise_multiplier) 
        else: 
            print("No DP")
    
    def fit(self,data,data_info):
        set_seed(self.seed)
        #data need to be transformed
        self.data_info = data_info
        data_dim = data.shape[1]
        print("data dim:", data_dim)
        self.ganinput = torch.from_numpy(data.astype('float32')).to(self.device)
        dataset = TensorDataset(self.ganinput)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,num_workers=0)

        #initialize models and optimizers
        self.myG = Generator(self.embedding_dim, self.gen_dim, data_dim).to(self.device)
        myD = Critic(data_dim, self.dis_dim).to(self.device)
        optimG = optim.Adam(self.myG.parameters(),lr = 1e-4, betas = (0.5,0.9), weight_decay=self.l2scale)
        optimD = optim.Adam(myD.parameters(), lr=1e-4, betas = (0.5,0.9) )

        mean = torch.zeros((self.batch_size+300),self.embedding_dim,device = self.device)
        std = mean +1


        self.tanh_list = []
        self.soft_list =[]
        st=0
        for item in self.data_info:
            if item[1]=='tanh':
                self.tanh_list.append(st)
                st = st + item[0]
            elif item[1]=='softmax':
                self.soft_list.append(st)
                st = st + item[0]
            else:
                assert 0

        Loss_D = []
        Loss_G = []

        c,f,c2,f2 = construct_coefficients(samplesize=800,
                                            dim =data_dim ,
                                            trainingsize =data.shape[0], 
                                            device=self.device)

        print("Begin training...")
        for i in range(self.epochs):
            torch.cuda.synchronize()
            start = time.time()
            for _, data in enumerate(loader):

                for _ in range(2):
                    
                    
                    real = data[0]
                    noise = torch.normal(mean = mean, std = std) #(500,128)
                    fake = self.myG(noise)  #(500,124)  
                    #apply activation

                    f_samples = apply_activate(fake, self.tanh_list,self.soft_list) 

                    #choose fakeact that disjoint with the training set
                    choose_index = choose_disjoint(f_samples,self.ganinput, self.soft_list,self.tanh_list,self.device,c,f,c2,f2)
                    assert len(choose_index)>=500, "Insufficient samples"
                    fakeact = f_samples[choose_index][:self.batch_size]
                    
                    y_real = myD(real)
                    y_fake = myD(fakeact.detach())
                   
                    pen = gradient_penalty_compute(myD, real, fakeact, self.device)
                   
                    loss_d = -torch.mean(y_real) + torch.mean(y_fake) + pen

                    optimD.zero_grad()
                    loss_d.backward() 

                    if self.DP == True:
                        bs = len(real) + len(fake)
                        #clip gradient
                        torch.nn.utils.clip_grad_norm_(myD.parameters(), self.cg)
                        #add noise
                        for name, params in myD.named_parameters():
                            noise = torch.randn(params.grad.shape).to(self.device)* (self.sigma**2) / bs
                            params.grad = params.grad + noise.detach()

                    optimD.step()

                    torch.cuda.synchronize()
                    
                    
                noise_2 = torch.normal(mean = mean, std = std)
                fake_2 = self.myG(noise_2)
                f_samples2 = apply_activate(fake_2,  self.tanh_list,self.soft_list)

                choose_index2 = choose_disjoint(f_samples2,self.ganinput, self.soft_list,self.tanh_list,self.device,c,f,c2,f2)
                assert len(choose_index2)>=500, "Insufficient samples"
                fakeact_2 = f_samples2[choose_index2][:self.batch_size]
        
                y_fake_2 = myD(fakeact_2)
                loss_g = -torch.mean(y_fake_2)

                optimG.zero_grad()
                loss_g.backward()
                optimG.step()
            
            Loss_D.append(loss_d.item())
            Loss_G.append(loss_g.item())

            torch.cuda.synchronize()   
            end = time.time()
            diff = end-start
            
            print('[%d/%d]\t  Loss_D: %.4f\tLoss_G: %.4f\t runtime: %.4f s\t' % (i, self.epochs,
                    loss_d.item(), loss_g.item(),diff))


        return Loss_D,Loss_G

    '''  
            if i<50:
                if ((i+1) % 5 ==0):
                    print('epoch:',i,"begin sample")
                    fake = self.sample(n=self.ganinput.shape[0])
                    syn_list.append(fake)
            else:
                if ((i+1) % 20 ==0):
                    print('epoch:',i,"begin sample")
                    fake = self.sample(n=self.ganinput.shape[0])
                    syn_list.append(fake)
                #self.plot_distribution()
                #self.evaluation(eval_syn=1)  
                #        
        
        return myD
    '''
    
    def sample(self,n,seed=0):
        print("Begin sample，seed=",seed)
        set_seed(seed)
        steps = n // self.batch_size +1
        data = []
        for _ in range(steps):

            noise = torch.randn(self.batch_size , self.embedding_dim).to(self.device)
            fake = self.myG(noise)
            fakeact = apply_activate(fake,  self.tanh_list,self.soft_list)
            data.append(fakeact.detach().cpu().numpy())  
        data = np.concatenate(data, axis=0)  #输出一个[batch_size*step]的np array
        self.data = data[:n]  #多生成点，取前n个
        return self.data     
    
'''
if __name__ == "__main__":
    df_adult = pd.read_csv('./dataset/combined_set.csv')
    data, con_vars = adult_preprocess(df_adult)

    train_data, test_data = train_test_split(data, test_size=0.3, random_state=2)

    tf = BaseTransformer(train_data,con_vars)

    ganinput= tf.transform()
    print("Gan input shape:", ganinput.shape)
    gan = ShadowSynthesizer(epochs = 300,seed=5,DP = False,batch_size = 500)
    gan.fit(ganinput,tf.data_info)
'''


def choose_disjoint(fake_sample,training,soft_list,tanh_list,device,c,f,c2,f2):
    temp = fake_sample[:,tanh_list[0]:(tanh_list[-1]+1)]
    for i in range(len(soft_list)-1):
        tem_data = fake_sample[:,soft_list[i]:soft_list[i+1]]
        x = torch.zeros_like(tem_data,device = device)
        index = torch.argmax(tem_data,dim=1)
        x[np.arange(0,fake_sample.shape[0],1),index]=1
        temp = torch.cat([temp,x],1)
    tem_data = fake_sample[:,soft_list[-1]:]
    x = torch.zeros_like(tem_data,device = device)
    tem_soft = torch.argmax(tem_data,dim=1)
    x[np.arange(0,fake_sample.shape[0],1),tem_soft]=1
    temp = torch.cat([temp,x],1)
    
    #transform age   
    temp = torch.round( temp* c + f)
    
    training = torch.round(training*c2 +f2)
    
    concat = torch.cat((temp,training), axis=0)
    unique,inverse_index,count = torch.unique(concat,dim=0, return_inverse=True,return_counts=True)
    x = ~(count-1)[inverse_index][:temp.shape[0]].bool()
    choose_index = torch.nonzero(x).squeeze()
    
    return choose_index

def construct_coefficients(samplesize,dim,trainingsize,device):
    a = torch.tensor([73.])
    a = a.repeat(samplesize,1)
    b= torch.ones([samplesize,dim-1])
    c = torch.cat([a,b],1).to(device)
    
    d = torch.tensor([19.])
    d = d.repeat(samplesize,1)
    e= torch.zeros_like(b)
    f = torch.cat([d,e],1).to(device)

    a2 = torch.tensor([73.])
    a2 = a2.repeat(trainingsize,1)
    b2= torch.ones([trainingsize,dim-1])
    c2 = torch.cat([a2,b2],1).to(device)
    
    d2 = torch.tensor([19.])
    d2 = d2.repeat(trainingsize,1)
    e2 = torch.zeros_like(b2)
    f2 = torch.cat([d2,e2],1).to(device)
    return c,f,c2,f2