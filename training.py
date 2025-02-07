from os import listdir
from os.path import join
import random
import matplotlib.pyplot as plt
import natsort

import os
import time
import torch
from torch import nn as nn
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm
import numpy as np
from torch import optim

from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import *
from math import pi
from monai.networks import nets

setting_dict = {
    'source': 'GE', # source scanner // 'Siemens', 'GE', 'Philips'
    'target': 'Philips',  # target scanner // 'Siemens', 'GE', 'Philips'
}


def make_colorbar_with_padding(ax):
    """
    Create colorbar axis that fits the size of a plot - detailed here: http://chris35wills.github.io/matplotlib_axis/
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    return(cax)


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_root = '...' #trainig data root 
    save_root = '...' #save root for training/validation results
    load_state_dir = '...' # load dir for Quantitative Maps Generator's model parameter

    k_list = get_subjects_list(setting_dict, data_root) # for k-fold 
    print(k_list)

    z = 224 # the number of slices

    lr = 1e-3 #
    beta1 = 0.5
    beta2 = 0.999
    num_epochs = 200
    batch_size = 2


    for k in k_list:
        # dir for figure and pth save
        save_fig_dir = f"{save_root}/fig/{setting_dict['source']}_{setting_dict['target']}/fold{k}"
        save_pth_dir = f"{save_root}/pth/{setting_dict['source']}_{setting_dict['target']}/fold{k}"

        os.makedirs(save_fig_dir, exist_ok=True)

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


        denoising_net_st=nets.BasicUnet(
            in_channels=1,
            out_channels=1,
            spatial_dims=2,
            features=(16,32,64,128,256,32),
            dropout=0.1,
        ).to(device)

        denoising_net_ts=nets.BasicUnet(
            in_channels=1,
            out_channels=1,
            spatial_dims=2,
            features=(16,32,64,128,256,32),
            dropout=0.1,
        ).to(device)

        
        recon_loss = nn.MSELoss().to(device)
        constraint_loss = nn.MSELoss().to(device)

        optimizer_target = torch.optim.Adam(denoising_net_st.parameters(),lr=lr,betas=(beta1,beta2))
        optimizer_source = torch.optim.Adam(denoising_net_ts.parameters(),lr=lr,betas=(beta1,beta2))

        checkpoint = torch.load(f'{load_state_dir}/MapG_T1.pth')
        generator_T1map.load_state_dict(checkpoint['T1map'])

        checkpoint = torch.load(f'{load_state_dir}/MapG_M0.pth')
        generator_M0map.load_state_dict(checkpoint['M0map'])

        source = setting_dict['source']
        target = setting_dict['target']

        start_time = time.time()
        train_dataset = CustomDataset(
            data_root=data_root, setting_dict=setting_dict, k_list=k_list, num_k=k, num_z_slice=z,
        )
        
        valid_dataset = CustomDataset(
            data_root=data_root, setting_dict=setting_dict, k_list=k_list, num_k=k, num_z_slice=z, train=False,
        )

        train_dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
        valid_dl = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
                

        loss_hist = {'target':[],
                    'source':[],
                    }
        mean_hist = {'target':[],
                     'source':[],
                    }

        lowest_val_loss = 1e9
        for epoch in range(0, num_epochs):

            denoising_net_st.train()
            denoising_net_ts.train()
            generator_T1map.eval()
            generator_M0map.eval()

            for batch_idx, batch in enumerate(tqdm(train_dl)):

                slide_source = batch[0].to(device)
                slide_target = batch[1].to(device)
                slide_mask = batch[2].to(device)

                slide_source = slide_source / 4095
                slide_target = slide_target / 4095

                with torch.no_grad(): # Generate T1map, M0map from the pre-trained Quantitative Maps Generator


                    pred_T1map_s = generator_T1map(slide_source)
                    pred_M0map_s = generator_M0map(slide_source)

                    pred_T1map_t = generator_T1map(slide_target)
                    pred_M0map_t = generator_M0map(slide_target)

                    pred_T1map_s = slide_mask*(pred_T1map_s-pred_T1map_s.min())*4095
                    pred_M0map_s = slide_mask*(pred_M0map_s-pred_M0map_s.min())*500

                    pred_T1map_t = slide_mask*(pred_T1map_t-pred_T1map_t.min())*4095
                    pred_M0map_t = slide_mask*(pred_M0map_t-pred_M0map_t.min())*500

                    cons_source, cons_target = cal_consT1w(slide_source, pred_T1map_s, pred_M0map_s, pred_T1map_t, pred_M0map_t, setting_dict) 
                    
                    cons_target = cons_target*slide_mask
                    cons_source = cons_source*slide_mask
                    
                    cons_target = torch.nan_to_num(cons_target,nan=0.0)
                    cons_source = torch.nan_to_num(cons_source,nan=0.0)

                    
                    print(f'Constrained mean value : {cons_target.mean()}, {cons_source.mean()}' ,end="\r")

                # Train the Harmoniztaion Network

                slide_source = slide_source * 4095
                slide_target = slide_target * 4095
                denoising_net_st.zero_grad()

                pred_target = denoising_net_st(cons_target)

                recon_loss_target = recon_loss(pred_target, slide_target)

                total_loss_target = recon_loss_target

                total_loss_target.backward()
                optimizer_target.step()

                denoising_net_ts.zero_grad()

                pred_source = denoising_net_ts(cons_source)
                recon_loss_source = recon_loss(pred_source, slide_source)

                total_loss_source = recon_loss_source
                total_loss_source.backward()
                optimizer_source.step()

                loss_hist['target'].append(total_loss_target.item())
                loss_hist['source'].append(total_loss_source.item())
                    
            mean_hist['target'].append(sum(loss_hist['target']) / len(loss_hist['target']))
            mean_hist['source'].append(sum(loss_hist['source']) / len(loss_hist['source']))

            print('Epoch: %.0f, ST Loss: %.6f, TS Loss: %.6f, time: %.2f min' %(epoch, mean_hist['target'][-1], mean_hist['source'][-1], (time.time()-start_time)/60))

            # validation
            denoising_net_st.eval()
            denoising_net_ts.eval()
            generator_T1map.eval()
            generator_M0map.eval()

            val_loss_hist = {'target':[], 'source':[]}
            val_mean_hist = {'target':[], 'source':[]}

            real_a = []
            real_b = []
            pred_a = []
            pred_b = []
            cons_a = []
            cons_b = []

            pred_T1 = []
            pred_M0 = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(valid_dl)):
                    slide_source = batch[0].to(device)
                    slide_target = batch[1].to(device)
                    slide_mask = batch[2].to(device)

                    slide_source = slide_source*slide_mask
                    slide_target = slide_target*slide_mask

                    slide_source = slide_source / 4095
                    slide_target = slide_target / 4095

                    pred_T1map_s = generator_T1map(slide_source)
                    pred_M0map_s = generator_M0map(slide_source)

                    pred_T1map_t = generator_T1map(slide_target)
                    pred_M0map_t = generator_M0map(slide_target)

                    pred_T1map_s = slide_mask*(pred_T1map_s-pred_T1map_s.min())*4095
                    pred_M0map_s = slide_mask*(pred_M0map_s-pred_M0map_s.min())*500

                    pred_T1map_t = slide_mask*(pred_T1map_t-pred_T1map_t.min())*4095
                    pred_M0map_t = slide_mask*(pred_M0map_t-pred_M0map_t.min())*500

                    cons_source, cons_target = cal_consT1w(slide_source, pred_T1map_s, pred_M0map_s, pred_T1map_t, pred_M0map_t, setting_dict) 

                    
                    cons_target = cons_target*slide_mask
                    cons_source = cons_source*slide_mask
                    
                    cons_target = torch.nan_to_num(cons_target,nan=0.0)
                    cons_source = torch.nan_to_num(cons_source,nan=0.0)

                    slide_source = slide_source * 4095
                    slide_target = slide_target * 4095
                    denoising_net_st.zero_grad()
                    pred_target = denoising_net_st(cons_target)

                    recon_loss_target = recon_loss(pred_target, slide_target)
                    lambda_3 = 1e-6

                    total_loss_target = recon_loss_target

                    denoising_net_ts.zero_grad()

                    pred_source = denoising_net_ts(cons_source)
                    recon_loss_source = recon_loss(pred_source, slide_source)
                    total_loss_source = recon_loss_source

                    val_loss_hist['source'].append(total_loss_source.item())
                    val_loss_hist['target'].append(total_loss_target.item())

                    real_a.append(slide_source.detach().cpu().numpy())
                    real_b.append(slide_target.detach().cpu().numpy())

                    pred_a.append(pred_source.detach().cpu().numpy())
                    pred_b.append(pred_target.detach().cpu().numpy())

                    cons_a.append(cons_source.detach().cpu().numpy())
                    cons_b.append(cons_target.detach().cpu().numpy())
                    
                    pred_T1.append(pred_T1map_s.detach().cpu().numpy())
                    pred_M0.append(pred_M0map_s.detach().cpu().numpy())

            val_mean_hist['source'].append(sum(val_loss_hist['source']) / len(val_loss_hist['source']))
            val_mean_hist['target'].append(sum(val_loss_hist['target']) / len(val_loss_hist['target']))

            real_a = np.concatenate(real_a, axis=0).squeeze(1).transpose(1, 2, 0)
            real_b = np.concatenate(real_b, axis=0).squeeze(1).transpose(1, 2, 0)
            pred_a = np.concatenate(pred_a, axis=0).squeeze(1).transpose(1, 2, 0)
            pred_b = np.concatenate(pred_b, axis=0).squeeze(1).transpose(1, 2, 0)
            cons_a = np.concatenate(cons_a, axis=0).squeeze(1).transpose(1, 2, 0)
            cons_b = np.concatenate(cons_b, axis=0).squeeze(1).transpose(1, 2, 0)
            pred_T1 = np.concatenate(pred_T1, axis=0).squeeze(1).transpose(1, 2, 0)
            pred_M0 = np.concatenate(pred_M0, axis=0).squeeze(1).transpose(1, 2, 0)

            print('Val Target Epoch: %.0f, Loss: %.6f, time: %.2f min' %(epoch, val_mean_hist['target'][-1], (time.time()-start_time)/60))  
            print('Val Source Epoch: %.0f, Loss: %.6f, time: %.2f min' %(epoch, val_mean_hist['source'][-1], (time.time()-start_time)/60))         

            save_loss_graph(save_fig_dir, k, mean_hist, val_mean_hist)

            source = setting_dict['source']
            target = setting_dict['target']

            save_target_path = f'{save_pth_dir}/{source}_to_{target}'
            save_source_path = f'{save_pth_dir}/{target}_to_{source}'


            if lowest_val_loss > mean_hist['target'][-1]:
                save_figure(setting_dict, save_fig_dir, k, real_a, real_b, pred_a, pred_b,pred_T1,pred_M0, cons_a, cons_b)
                save_model_state(save_target_path, k, epoch, denoising_net_st, best=True)
            
            if lowest_val_loss > mean_hist['source'][-1]:
                save_figure(setting_dict, save_fig_dir, k, real_a, real_b, pred_a, pred_b,pred_T1,pred_M0, cons_a, cons_b)
                save_model_state(save_source_path, k, epoch, denoising_net_ts, best=True)

            if (epoch%25==0) or (epoch==num_epochs-1):
                save_model_state(save_source_path, k, epoch, denoising_net_ts, best=False)
                save_model_state(save_target_path, k, epoch, denoising_net_st, best=False)
    
if __name__ =="__main__" :
    main()