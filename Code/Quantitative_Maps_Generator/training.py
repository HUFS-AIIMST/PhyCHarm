import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import warnings

from monai.networks import nets

import os
from tqdm.notebook import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import enum
from IPython import display
from tqdm.notebook import tqdm

from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.patches as mpatches
import natsort
from scipy import io
import glob
import nibabel as nib
from utils import *
import socket
import natsort



def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_root = '...' #Define data root
    save_root = '...' #Define save root

    lambda_1 = 1
    lambda_2 = 1
    lambda_3 = 1e-6

    save_state_dir=f'{save_root}/...'
    save_fig_dir=f'{save_root}/...'
    os.makedirs(save_state_dir, exist_ok=True)
    os.makedirs(save_fig_dir, exist_ok=True)
    
    subjects_list = get_subjects_list(f'{data_root}')
    k=1

    generator_T1map=nets.BasicUnet(
        in_channels=1,
        out_channels=1,
        spatial_dims=2,
        features=(16,32,64,128,256,32),
    ).to(device)
    
    generator_M0map=nets.BasicUnet(
        in_channels=1,
        out_channels=1,
        spatial_dims=2,
        features=(16,32,64,128,256,32),
    ).to(device)

    optimizer_G_T1map=torch.optim.Adam(generator_T1map.parameters(), lr=1e-03)
    optimizer_G_M0map=torch.optim.Adam(generator_M0map.parameters(), lr=1e-03)
    
    z=224
    batch_size = 2
    subjects_list = subjects_list[:25]
    train_dataset = CustomDataset(
        data_root=data_root, k_list=subjects_list, num_k=subjects_list[5*(k-1):5*(k-1)+5], num_z_slice=z,train=True,
    )

    valid_dataset = CustomDataset(
        data_root=data_root, k_list=subjects_list, num_k=subjects_list[5*(k-1):5*(k-1)+5], num_z_slice=z, train=False,
    )

    train_dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
        
    
    
    MSE_loss=nn.MSELoss().to(device)
    
    val_interval = 1


    num_epochs=500
    
    train_losses_g_T1map=[]
    train_losses_g_M0map=[]
    train_losses_g_syn=[]

    valid_losses_g_T1map=[]
    valid_losses_g_M0map=[]
    valid_losses_g_syn=[]

    lowest_loss_T1map = 1e9
    lowest_loss_M0map = 1e9

    for epoch_idx in range(0, num_epochs):
        
        print('Starting epoch', epoch_idx)
        generator_T1map.train()
        generator_M0map.train()
        
        epoch_G_loss_T1map=[]
        epoch_G_loss_M0map=[]
        epoch_G_loss_cons=[]
        
        for batch in (tqdm(train_dl)):

            slide_input_T1w = batch[0].to(device)                                                                                                   
            slide_T1map =  batch[1].to(device)
            slide_M0map = batch[2].to(device)
            mask = batch[3].to(device)

            slide_input_T1w = slide_input_T1w/4095
            slide_T1map = slide_T1map/4095
            slide_M0map = slide_M0map/500
            
            """generative network (mapping)"""
            # T1w to T1map

            optimizer_G_T1map.zero_grad()
            gen_T1map=generator_T1map(slide_input_T1w) # T1w to T1map
            gen_T1map = mask*(gen_T1map - gen_T1map.min())*4095
                    
            loss_gen_T1map = MSE_loss(gen_T1map, slide_T1map*4095)
            
            
            optimizer_G_M0map.zero_grad()
            gen_M0map=generator_M0map(slide_input_T1w) # T1w to M0map
            gen_M0map = mask*(gen_M0map - gen_M0map.min())*500
            loss_gen_M0map = MSE_loss(gen_M0map, slide_M0map*500)
            
            
            TI = 2830
            TR = 5000
            
            print(gen_T1map.max(), gen_M0map.max())
            
            eps=1e-1
            scale = 14.87
            syn_T1w = mask*scale*(gen_M0map)*(1-2*torch.exp(-TI/(gen_T1map+1e-1))+torch.exp(-TR/(gen_T1map+1e-1))) 
            slide_input_T1w = slide_input_T1w*4095
            syn_T1w[gen_T1map.max()==0]=0
            syn_T1w[gen_M0map.max()==0]=0
            syn_T1w = torch.nan_to_num(syn_T1w, nan=0.0)
            syn_T1w = (syn_T1w - syn_T1w.min())


            nominal_T2term = mask*slide_input_T1w/(syn_T1w+eps)
            nominal_T2term = torch.nan_to_num(nominal_T2term)

            syn_T1w = mask*scale*(gen_M0map)*(1-2*torch.exp(-TI/(gen_T1map+1e-1))+torch.exp(-TR/(gen_T1map+1e-1)))*nominal_T2term
            syn_T1w[gen_T1map.max()==0]=0
            syn_T1w[gen_M0map.max()==0]=0
            syn_T1w = torch.nan_to_num(syn_T1w, nan=0.0)
            syn_T1w = (syn_T1w - syn_T1w.min())
            
            loss_cons = MSE_loss(syn_T1w,slide_input_T1w) 

            total_loss = lambda_1*loss_gen_T1map+lambda_2*loss_gen_M0map+lambda_3*loss_cons

            epoch_G_loss_T1map.append(loss_gen_T1map.item())
            epoch_G_loss_M0map.append(loss_gen_M0map.item())
            epoch_G_loss_total.append(total_loss.item())

            total_loss.backward()

            optimizer_G_T1map.step()
            optimizer_G_M0map.step()

            print(f'T1map : {gen_T1map.mean()} M0map : {gen_M0map.mean()}\n', end='\r')
            print(f'T1 loss:{loss_gen_T1map} M0 loss: {loss_gen_M0map} Cons loss : {loss_cons}\n',end='\r')
            
            
                
        train_loss_g_T1map=np.array(epoch_G_loss_T1map).mean()
        train_loss_g_M0map=np.array(epoch_G_loss_M0map).mean()

        train_GT_T1map=slide_T1map.detach().cpu().numpy()*4095
        train_pred_T1map=gen_T1map.detach().cpu().numpy()
        train_losses_g_T1map.append(train_loss_g_T1map)

        
        train_GT_M0map=slide_M0map.detach().cpu().numpy()*500
        train_pred_M0map=gen_M0map.detach().cpu().numpy()
        show_train_results(save_fig_dir, k, train_GT_T1map, train_pred_T1map, train_GT_M0map, train_pred_M0map)

        train_losses_g_M0map.append(train_loss_g_M0map)
                        
        
        if (epoch_idx +1) % val_interval ==0:
            
            epoch_G_loss_T1map=[]
            epoch_G_loss_M0map=[]
            epoch_G_loss_cons=[]
            
            generator_T1map.eval()
            generator_M0map.eval()
            
            with torch.no_grad():
                
                for batch in valid_dl:
                    
                    slide_input_T1w = batch[0].to(device)                                                                                                   
                    slide_T1map =  batch[1].to(device)
                    slide_M0map = batch[2].to(device)
                    mask = batch[3].to(device)

                    slide_input_T1w = slide_input_T1w/4095
                    slide_T1map = slide_T1map/4095
                    slide_M0map = slide_M0map/500

                    gen_T1map=generator_T1map(slide_input_T1w) # T1w to T1map
                    gen_M0map=generator_M0map(slide_input_T1w)# T1w to T1map
                    
                    gen_T1map = mask*(gen_T1map - gen_T1map.min())*4095
                    gen_M0map = mask*(gen_M0map - gen_M0map.min())*500
                    
                    
                    loss_gen_T1map = MSE_loss(gen_T1map, slide_T1map*4095)
                    loss_gen_M0map = MSE_loss(gen_M0map, slide_M0map*500)

                    TI = 2830
                    TR = 5000
                    
                    # print(gen_T1map.max(), gen_M0map.max())
                    
                    eps=1e-1
                    syn_T1w = mask*scale*(gen_M0map)*(1-2*torch.exp(-TI/(gen_T1map+1e-1))+torch.exp(-TR/(gen_T1map+1e-1)))
                    slide_input_T1w = slide_input_T1w*4095
                    syn_T1w[gen_T1map.max()==0]=0
                    syn_T1w[gen_M0map.max()==0]=0
                    syn_T1w = torch.nan_to_num(syn_T1w, nan=0.0)
                    syn_T1w = (syn_T1w - syn_T1w.min())

                    nominal_T2term = mask*slide_input_T1w/(syn_T1w+eps)
                    # nominal_T2term[syn_T1w==0]=0
                    nominal_T2term = torch.nan_to_num(nominal_T2term)

                    syn_T1w = mask*scale*(gen_M0map)*(1-2*torch.exp(-TI/(gen_T1map+1e-1))+torch.exp(-TR/(gen_T1map+1e-1)))*nominal_T2term
                    syn_T1w[gen_T1map.max()==0]=0
                    syn_T1w[gen_M0map.max()==0]=0
                    syn_T1w = torch.nan_to_num(syn_T1w, nan=0.0)
                    syn_T1w = (syn_T1w - syn_T1w.min())
                    
                    loss_cons = MSE_loss(syn_T1w,slide_input_T1w) 

                    total_loss = lambda_1*loss_gen_T1map+lambda_2*loss_gen_M0map+lambda_3*loss_cons
                    epoch_G_loss_T1map.append(total_loss_T1.item())
                    epoch_G_loss_M0map.append(total_loss_M0.item())
                    epoch_G_loss_total.append(total_loss.item())

                    print(f'T1 mean : {gen_T1map.mean()} T1 loss:{total_loss_T1} M0 mean :{gen_M0map.mean()} M0 loss: {total_loss_M0} Cons loss : {loss_cons}',end='\r')
                        
            valid_loss_g_T1map=np.array(epoch_G_loss_T1map).mean()
            valid_loss_g_M0map=np.array(epoch_G_loss_M0map).mean()

            val_GT_T1map=slide_T1map.detach().cpu().numpy()*4095
            val_pred_T1map=gen_T1map.detach().cpu().numpy()
            
            valid_losses_g_T1map.append(valid_loss_g_T1map)

            val_GT_M0map=slide_M0map.detach().cpu().numpy()*500
            val_pred_M0map=gen_M0map.detach().cpu().numpy()
            
            valid_losses_g_M0map.append(valid_loss_g_M0map)


            if (epoch_idx%2==0) or (epoch_idx==num_epochs-1):
                show_val_results(save_fig_dir, k, val_GT_T1map, val_pred_T1map,val_GT_M0map, val_pred_M0map)                    
            
            if lowest_loss_T1map > valid_loss_g_T1map:
                lowest_loss_T1map=valid_loss_g_T1map
                save_pth_dir = f'{save_state_dir}/T1map'
                save_model_state_T1(save_pth_dir, k, epoch_idx, generator_T1map, best=True)

            if lowest_loss_M0map > valid_loss_g_M0map:
                lowest_loss_M0map=valid_loss_g_M0map
                save_pht_dir_M0 = f'{save_state_dir}/M0map'
                os.makedirs(save_pht_dir_M0, exist_ok=True)                    
                save_model_state_M0(save_pht_dir_M0, k, epoch_idx, generator_M0map, best=True)
                    
            save_fig_subdir = f'{save_fig_dir}/loss'
            os.makedirs(save_fig_subdir, exist_ok=True)  
            save_loss_graph(save_fig_subdir,k,train_losses_g_T1map, valid_losses_g_T1map, train_losses_g_M0map, valid_losses_g_M0map)
        

if __name__ =="__main__" :
    main()

            
        
        

