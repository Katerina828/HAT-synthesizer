import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import math
import random
from synthesizer.model import Critic,Residual,Generator,apply_activate,gradient_penalty_compute
from torch.utils.data import DataLoader, TensorDataset
import time
from utils import set_seed, get_data_info


class ShadowSynthesizer():
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
        self.ganinput = data
        self.data_info = data_info
        data_dim = self.ganinput.shape[1]
        print("data dim:", data_dim)

        dataset = TensorDataset(torch.from_numpy(self.ganinput.astype('float32')).to(self.device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        #initialize models and optimizers
        self.myG = Generator(self.embedding_dim, self.gen_dim, data_dim).to(self.device)
        myD = Critic(data_dim, self.dis_dim).to(self.device)
        optimG = optim.Adam(self.myG.parameters(),lr = 1e-4, betas = (0.5,0.9), weight_decay=self.l2scale)
        optimD = optim.Adam(myD.parameters(), lr=1e-4, betas = (0.5,0.9) )

        mean = torch.zeros(self.batch_size,self.embedding_dim,device = self.device)
        std = mean +1

        print("Begin training...")
        for i in range(self.epochs):
            torch.cuda.synchronize()
            start = time.time()
            for _, data in enumerate(loader):
  
                for _ in range(2):

                    real = data[0]

                    noise = torch.normal(mean = mean, std = std) #(500,128)
                    fake = self.myG(noise)  #(500,124)
                    
                    fakeact = apply_activate(fake, self.data_info)   #不打包只用这一句

                    y_real = myD(real)
                    y_fake = myD(fakeact.detach())

                    pen = gradient_penalty_compute(myD, real, fakeact, self.device)
                    loss_d = -torch.mean(y_real) + torch.mean(y_fake) + pen

                    optimD.zero_grad()
                    loss_d.backward() 
                    optimD.step()
                    
                noise_2 = torch.normal(mean = mean, std = std)
                fake_2 = self.myG(noise_2)
                fakeact_2 = apply_activate(fake_2, self.data_info)
        
                y_fake_2 = myD(fakeact_2)
                loss_g = -torch.mean(y_fake_2)

                optimG.zero_grad()
                loss_g.backward()
                optimG.step()

            torch.cuda.synchronize()
            end = time.time()
            diff = end-start

            print('[%d/%d]\t  Loss_D: %.4f\tLoss_G: %.4f\t runtime: %.4f s\t' % (i, self.epochs,
                     loss_d.item(), loss_g.item(),diff))

        return myD

    
    def sample(self,n,seed=0):
        print("Begin sample，seed=",seed)
        set_seed(seed)
        steps = n // self.batch_size +1
        data = []
        for _ in range(steps):

            noise = torch.randn(self.batch_size , self.embedding_dim).to(self.device)
            fake = self.myG(noise)
            fakeact = apply_activate(fake, self.data_info)
            data.append(fakeact.detach().cpu().numpy())  
        data = np.concatenate(data, axis=0)  #输出一个[batch_size*step]的np array
        self.data = data[:n]  #多生成点，取前n个
        return self.data     
    


