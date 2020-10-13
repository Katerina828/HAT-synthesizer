import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import math
import random
from base import BaseSynthesizer
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from utils import set_seed,save_generated_data,data_generator,get_data_info

#from data_preprocess.lawschool import LawschsDataset,SyntheticLawschsDataset
#from data_preprocess.health import HealthDataset
#from data_preprocess.adult import AdultDataset,SyntheicAdultDataset
#from data_preprocess.compas import CompasDataset, SyntheticCompasDataset
#from data_preprocess.crime import CrimeDataset
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()
from MLclassifier import main

import matplotlib.pyplot as plt
import seaborn as sns
import time

'''
We only use  Critic,Residual,Generator,gradient_penalty_compute,apply_activate function

'''


class Critic(nn.Module):
    def __init__(self,input_dim,dis_dims, pac=1):
        super(Critic, self).__init__()
        dim = input_dim*pac
        self.pac = pac
        self.pacdim = dim
        seq =[]
        for item in list(dis_dims):
            seq +=[
                nn.Linear(dim,item),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5)
            ]
            dim = item
        seq +=[nn.Linear(dim,1)]
        self.seq = nn.Sequential(*seq)    
       
        
        
    def forward(self,input):
        assert input.size()[0]% self.pac == 0
        return self.seq(input.view(-1, self.pacdim))

class Residual(nn.Module):
    def __init__(self, i, o):
        super(Residual,self).__init__()
        self.fc = nn.Linear(i,o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def forward(self,input):
        out = self.relu(self.bn(self.fc(input)))
        return torch.cat([out,input],dim =1)


class Generator(nn.Module):
    def __init__(self, embedding_dim,gen_dim, data_dim):
        super(Generator, self).__init__()

        dim = embedding_dim
        seq = []
        for item in list(gen_dim):
            seq +=[
                Residual(dim,item)
            ]
            dim +=item
        seq.append(nn.Linear(dim,data_dim))
        self.seq = nn.Sequential(*seq)   
        
    def forward(self,input):
        data = self.seq(input)
        return data


def apply_activate(data,tanh_list,soft_list):
    temp = torch.sigmoid(data[:,tanh_list[0]:(tanh_list[-1]+1)])
    for i in range(len(soft_list)-1):
        tem_soft = F.gumbel_softmax(data[:,soft_list[i]:soft_list[i+1]], tau=0.2)
        temp = torch.cat([temp,tem_soft],1)
    tem_soft = F.gumbel_softmax(data[:,soft_list[-1]:], tau=0.2)
    temp = torch.cat([temp,tem_soft],1)
    return temp
'''
def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1]=='tanh':
            ed = st + item[0]
            data_t.append(torch.sigmoid(data[:, st:ed]))  #此处修改了tanh成sigmoid
            st = ed 
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:,st:ed], tau=0.2))
            st = ed
        else:
            assert 0
    return torch.cat(data_t, dim=1)
'''    
class GanSythesizer(BaseSynthesizer):
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

    def fit(self,data='real',dataset='Adult',del_=12519):

        set_seed(self.seed)
        if data=='real':
            if dataset =='Adult':
                self.tf = AdultDataset()
                self.label = "Income"
            elif dataset == "Lawschool":
                self.tf = LawschsDataset()
                self.label = "admit"
            elif dataset == "Health":
                self.tf = HealthDataset()
                self.label = "max_CharlsonIndex"
            elif dataset == "Compas":
                self.tf = CompasDataset()
                self.label = "two_year_recid"

            combined_data = self.tf.transform()   #return numpy array
            train_data, test_data = train_test_split(combined_data, test_size=0.3, random_state=2)
            self.ganinput = train_data  #self.ganinput: numpy array
            self.testinput = test_data
            self.combined_inverse = self.tf.inverse_transform(combined_data)  # return pandas dataframe
            self.train_inverse, self.test_inverse = train_test_split(self.combined_inverse, test_size=0.3, random_state=2)
            self.data_info = self.tf.data_info
            save_generated_data(data=self.train_inverse,name= dataset + '_train',column_name=self.tf.columns_name)
            save_generated_data(data=self.test_inverse,name= dataset + '_test',column_name=self.tf.columns_name)
            if (del_):
                self.ganinput = np.delete(self.ganinput,del_,0)
            #assert (self.ganinput[5198] == self.ganinput[13218]).all()

            
        
        #synthetic data
        elif data == 'synthetic':
            '''
            self.ganinput = data_generator()
            print("training data shape:", self.ganinput.shape)
            self.data_info = [(4,'softmax'),(1,'tanh')]
            '''
            self.tf = SyntheicAdultDataset(i=1)
            self.ganinput = self.tf.transform()
            self.data_info = self.tf.data_info
            self.train_inverse = self.tf.inverse_transform(self.ganinput)
        else:
            assert 0
        print("Gan input shape: ", self.ganinput.shape)
        data_dim = self.ganinput.shape[1]
        

        dataset = TensorDataset(torch.from_numpy(self.ganinput.astype('float32')).to(self.device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        #initialize models and optimizers
        self.myG = Generator(self.embedding_dim, self.gen_dim, data_dim).to(self.device)
        myD = Critic(data_dim, self.dis_dim).to(self.device)

        optimG = optim.Adam(self.myG.parameters(),lr = 1e-4, betas = (0.5,0.9), weight_decay=self.l2scale)
        optimD = optim.Adam(myD.parameters(), lr=1e-4, betas = (0.5,0.9) )

        assert self.batch_size %2 == 0
        mean = torch.zeros(self.batch_size,self.embedding_dim,device = self.device)
        std = mean +1
        
        print("Begin training...")
        for i in range(self.epochs):
            torch.cuda.synchronize()
            start = time.time()
            #批采样的indices
            for batch_idx, data in enumerate(loader):
  
                for _ in range(2):

                    real = data[0]

                    noise = torch.normal(mean = mean, std = std) #(500,128)
                    fake = self.myG(noise)  #(500,124)
                    
                    fakeact = apply_activate(fake, self.data_info)   #不打包只用这一句

                    

                    y_real = myD(real)
                    y_fake = myD(fakeact.detach())

                    #gradient penalty
                    pen = gradient_penalty_compute(myD, real, fakeact, self.device)
                    loss_d = -torch.mean(y_real) + torch.mean(y_fake) + pen
                    #C_loss.append(loss_d.item())
                   # writer.add_scalar('Loss_d', loss_d, i*steps_per_epoch+id_)

                    optimD.zero_grad()
                    loss_d.backward()
 
                    #add differential privacy
                    '''
                    bs = len(real) + len(fake)
                    #clip gradient
                    torch.nn.utils.clip_grad_norm_(myD.parameters(), self.cg)
                    #add noise
                    for name, params in myD.named_parameters():
                        noise = torch.randn(params.grad.shape).to(self.device)* (self.sigma**2) / bs
                        params.grad = params.grad + noise.detach()
                    '''    
                    optimD.step()
                    
                noise_2 = torch.normal(mean = mean, std = std)
                fake_2 = self.myG(noise_2)
                fakeact_2 = apply_activate(fake_2, self.data_info)
        
                y_fake_2 = myD(fakeact_2)
                loss_g = -torch.mean(y_fake_2)
                #G_loss.append(loss_g.item())
                #writer.add_scalar('Loss_g', loss_g, i*steps_per_epoch+id_)
                optimG.zero_grad()
                loss_g.backward()
                optimG.step()
            #writer.add_scalar('train/Loss_d', loss_d, i)
            torch.cuda.synchronize()
            end = time.time()
            diff = end-start

            print('[%d/%d]\t  Loss_D: %.4f\tLoss_G: %.4f\t runtime: %.4f s\t' % (i, self.epochs,
                     loss_d.item(), loss_g.item(),diff))
            
            if ((i+1) % 50 ==0):
                #fake = self.sample(n=self.ganinput.shape[0])
                #name = str(dataset) +'_syn_epoch' + str(i) + '_seed'+ str(self.seed) + "_del"+ str(del_)
                #save_generated_data(data=fake,name=name,column_name=self.tf.columns_name)
                self.plot_distribution()
                #self.evaluation(eval_syn=1)
                                  
        return myD
        

    def sample(self, n,save_name='synthetic_data',seed=0):
        print("Begin sample，seed=",seed)
        set_seed(seed)
        steps = n // self.batch_size +1
        data = []
        for _ in range(steps):

            noise = torch.randn(self.batch_size , self.embedding_dim ).to(self.device)
            fake = self.myG(noise)
            fakeact = apply_activate(fake, self.data_info)
            data.append(fakeact.detach().cpu().numpy())  
        data = np.concatenate(data, axis=0)  #输出一个[batch_size*step]的np array
        self.data = data[:n]  #多生成点，取前n个
        return self.tf.inverse_transform(self.data)

    def evaluation(self,eval_syn=1):

        print("Eval base")
        real  = pd.DataFrame(self.train_inverse,index=None,columns = self.tf.columns_name)
        test = pd.DataFrame(self.test_inverse,index=None,columns = self.tf.columns_name)
        y_train = real[self.label]
        X_train = real.drop(self.label,axis=1)
        y_test = test[self.label]
        X_test = test.drop(self.label,axis=1)
        overall_eval =main(X_train,y_train,X_test,y_test)

        if eval_syn:
            print("Eval syn")
            sample = self.sample(n=real.shape[0])
            syn = pd.DataFrame(sample,index=None,columns = self.tf.columns_name)
            y_train = syn[self.label]
            X_train = syn.drop(self.label,axis=1)
            overall_eval =main(X_train,y_train,X_test,y_test)
        
        print(overall_eval)

    def count_target(self,target):
        count = 0
        for i in range(10):
            fake = self.sample(n=self.ganinput.shape[0]*10)
            c = findByRow(fake, target).shape[0]
            count = count + c
        return count

    def plot_distribution(self):
        sns.set(style="ticks", color_codes=True,font_scale=1.5)
        real  = pd.DataFrame(self.train_inverse,index=None,columns = self.tf.columns_name)
        sample = self.sample(n=real.shape[0])
        syn = pd.DataFrame(sample,index=None,columns = self.tf.columns_name)

        #real['label']=1
        #syn['label']= 0
        #df = pd.concat([real,syn],axis=0)

        for info in self.data_info:
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
        



def findByRow(mat, row):
    return np.where((mat == row).all(1))[0]




def gradient_penalty_compute(netD,real_data,fake_data,device='cpu',pac=1,lambda_ = 10):

    alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
    alpha = alpha.repeat(1, pac, real_data.size(1))
    alpha = alpha.view(-1, real_data.size(1))

    interpolates = (alpha*real_data + (1-alpha)*fake_data).requires_grad_(True)

    d_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size(),device=device),
        create_graph=True,
        retain_graph=True, 
        only_inputs=True)[0]

    gradient_penalty = (
        (gradients.view(-1, pac*real_data.size(1)).norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty

 

    
