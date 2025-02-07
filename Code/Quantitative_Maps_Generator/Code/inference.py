from os import listdir
from os.path import join
import random
import matplotlib.pyplot as plt
import natsort

import os
import time
from PIL import Image
import torch
from torch import nn as nn
# from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm
import numpy as np

from torch import optim
import socket


from mpl_toolkits.axes_grid1 import make_axes_locatable


# from models import *
from utils import *
from math import pi
from monai.networks import nets
import glob

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


def Bloch_eq(t1map,m0map,opt):
    """
    This calculates bloch equation for T1w
    """
    b,c,h,w=t1map.shape
    # opt 1 : Siemens 
    if opt==1 :
        #Siemens parameter
        TR=2400
        TI=1200        
        TE = 2.12
        eps=1e-1
        
        cons_T1w=m0map*(1-2*torch.exp(-TI/(t1map+eps))+torch.exp(-TR/(t1map+eps)))

        return cons_T1w, TE
    
    elif opt==2 : #Philips
        TR=1800
        TI=1000 
        TE = 2.12
        eps=1e-1
        
        cons_T1w=m0map*(1-2*torch.exp(-TI/(t1map+eps))+torch.exp(-TR/(t1map+eps)))
        
        return cons_T1w, TE

    #opt3: GE 
    elif opt==3 :
        #GE parameter
        eps=1e-1

        TR = 8.232
        TE = 3.192
        TI = 450

        cons_T1w=m0map*(1-2*torch.exp(-TI/(t1map+eps))+torch.exp(-TR/(t1map+eps)))
        cons_T1w[t1map==0]=0

        return cons_T1w, TE


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = '...' #inference data root 
    save_root = '...' #save root for inference results
    load_state_dir_map = '.../CheckPoint/Quantitative_Maps_Generator' # load dir for Quantitative Maps Generator's model parameter 
    load_state_dir_phycharm = '.../CheckPoint/Harmonization_Network' # load dir for the Harmonization Network's model parameter

    sample = nib.load(glob.glob(f'{data_root}/*.nii.gz')[0]).get_fdata()
    [x,y,z] = sample.shape

    lr = 2e-4 #
    beta1 = 0.5
    beta2 = 0.999
    batch_size = 1
    
    save_data_dir = f"{save_root}/{setting_dict['source']}_{setting_dict['target']}"
    os.makedirs(save_data_dir, exist_ok=True)

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
        features=(16,32,64,128,256,32),
        spatial_dims=2,
    ).to(device)

        
    # load model parameters of the Quantitative Maps Generator    

    checkpoint = torch.load(f'{load_state_dir_map}/MapG_T1.pth')
    generator_T1map.load_state_dict(checkpoint['T1map'])

    checkpoint = torch.load(f'{load_state_dir_map}/MapG_M0.pth')
    generator_M0map.load_state_dict(checkpoint['M0map'])

    source_domain = setting_dict['source']
    target_domain = setting_dict['target'] 

    #load model parameters of the Harmonization Network
    checkpoint=torch.load(f'{load_state_dir_phycharm}/{source_domain}_to_{target_domain}/Harm_{source_domain}_{target_domain}.pth')
    denoising_net_st.load_state_dict(checkpoint['model'])

    for para in generator_T1map.parameters():
        para.requires_grad = False
    for para in generator_M0map.parameters():
        para.required_grad = False

    start_time = time.time()
        
    inference_dataset = CustomDataset_inference(data_root=data_root, setting_dict=setting_dict, num_z_slice=z)

    inference_dl = DataLoader(inference_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    generator_T1map.eval()
    generator_M0map.eval()
    denoising_net_st.eval()
    

    with torch.no_grad():
        
        volume_pred_T1 = torch.zeros((x,y,z)).to(device)
        volume_pred_M0 = torch.zeros((x,y,z)).to(device)
        volume_cons_target = torch.zeros((x,y,z)).to(device)
        volume_harmonized_target = torch.zeros((x,y,z)).to(device)

        for batch_idx, batch in enumerate(tqdm(inference_dl)):
            
            slide_source = batch[0].to(device)
            slide_mask = batch[1].to(device)

            slide_source = slide_source / 4095

            if batch_idx==0:
                print('---1. Generate T1map and M0map---')
            pred_T1map = generator_T1map(slide_source)
            pred_M0map = generator_M0map(slide_source)

            pred_T1map = slide_mask*(pred_T1map-pred_T1map.min())*4095
            pred_M0map = slide_mask*(pred_M0map-pred_M0map.min())*500
            
            if batch_idx==0:
                print('---2. Calculate the constrained T1w---')

            if setting_dict['target'] =='Siemens':
                if setting_dict['source']=='GE':
                    opt_target = 1
                    opt_source = 3

                    cons_source, TE = Bloch_eq(pred_T1map, pred_M0map, opt_source) # Eq.7 in the article.
                    T2term_source = slide_source/cons_source # Eq.8 in the article.

                    T2map = -TE/torch.log(T2term_source) # Eq.9 in the article.
                    T2map[T2term_source==0]=0

                    cons_target, TE = Bloch_eq(pred_T1map, pred_M0map_s, opt_target)  # Eq.10 in the article.
                    cons_target = cons_target*torch.exp(-TE/(T2map+1e-1)) # Eq.10 in the article.


                elif setting_dict['source']=='Philips':
                    opt_target=1
                    opt_source=2

                    cons_source, TE = Bloch_eq(pred_T1map, pred_M0map, opt_source) # Eq.7 in the article.
                    T2term_source = slide_source/cons_source  # Eq.8 in the article.

                    T2map = -TE/torch.log(T2term_source) # Eq.9 in the article.
                    T2map[T2term_source==0]=0

                    cons_target, TE = Bloch_eq(pred_T1map, pred_M0map, opt_target)  # Eq.10 in the article.
                    cons_target = cons_target*torch.exp(-TE/(T2map+1e-1)) # Eq.10 in the article.


            
            elif setting_dict['target']=='Philips':
                if setting_dict['source']=='GE':
                    opt_target = 2
                    opt_source = 3

                    cons_source, TE = Bloch_eq(pred_T1map, pred_M0map, opt_source) # Eq.7 in the article.
                    T2term_source = slide_source/cons_source  # Eq.8 in the article.

                    T2map = -TE/torch.log(T2term_source) # Eq.9 in the article.
                    T2map[T2term_source==0]=0

                    cons_target, TE = Bloch_eq(pred_T1map, pred_M0map, opt_target)  # Eq.10 in the article.
                    cons_target = cons_target*torch.exp(-TE/(T2map+1e-1)) # Eq.10 in the article.


                elif setting_dict['source']=='Siemens':
                    opt_target=2
                    opt_source=1

                    cons_source, TE = Bloch_eq(pred_T1map, pred_T1map, opt_source) # Eq.7 in the article.
                    T2term_source = slide_source/cons_source # Eq.8 in the article.

                    T2map = -TE/torch.log(T2term_source) # Eq.9 in the article.
                    T2map[T2term_source==0]=0

                    cons_target, TE = Bloch_eq(pred_T1map, pred_M0map, opt_target)  # Eq.10 in the article.
                    cons_target = cons_target*torch.exp(-TE/(T2map+1e-1)) # Eq.10 in the article.


            elif setting_dict['target']=='GE':
                if setting_dict['source']=='Siemens':
                    opt_target = 3
                    opt_source = 1

                    cons_source, TE = Bloch_eq(pred_T1map, pred_M0map, opt_source) # Eq.7 in the article.
                    T2term_source = slide_source/cons_source # Eq.8 in the article.

                    T2map = -TE/torch.log(T2term_source) # Eq.9 in the article.
                    T2map[T2term_source==0]=0

                    cons_target, TE = Bloch_eq(pred_T1map, pred_M0map, opt_target)  # Eq.10 in the article.
                    cons_target = cons_target*torch.exp(-TE/(T2map+1e-1)) # Eq.10 in the article.


                elif setting_dict['source']=='Philips':
                    opt_target=3
                    opt_source=2

                    cons_source, TE = Bloch_eq(pred_T1map, pred_M0map, opt_source) # Eq.7 in the article.
                    T2term_source = slide_source/cons_source # Eq.8 in the article.

                    T2map = -TE/torch.log(T2term_source) # Eq.9 in the article.
                    T2map[T2term_source==0]=0

                    cons_target, TE = Bloch_eq(pred_T1map, pred_M0map, opt_target)  # Eq.10 in the article.
                    cons_target = cons_target*torch.exp(-TE/(T2map+1e-1)) # Eq.10 in the article.
            
            
            cons_target = torch.nan_to_num(cons_target,nan=0.0)
            cons_target = cons_target*slide_mask

            if batch_idx==0:
                print('---3. Harmonization---')
            pred_target = denoising_net_st(cons_target)
            
            volume_pred_T1[:,:,batch_idx] = pred_T1map[0,0,:,:]
            volume_pred_M0[:,:,batch_idx] = pred_M0map[0,0,:,:]
            volume_cons_target[:,:,batch_idx] = cons_target[0,0,:,:]
            volume_harmonized_target[:,:,batch_idx] = pred_target[0,0,:,:]
            print(f"\r---slice {batch_idx}/{z}", end="")
            

        volume_pred_T1 = volume_pred_T1.detach().cpu().numpy()
        volume_pred_M0 = volume_pred_M0.detach().cpu().numpy()
        volume_cons_target = volume_cons_target.detach().cpu().numpy()
        volume_harmonized_target=volume_harmonized_target.detach().cpu().numpy()
    
    print('---4. End! Save data.---')

    save_inference_data(save_data_dir, data_root, setting_dict, volume_pred_T1, volume_pred_M0, volume_cons_target, volume_harmonized_target)

    
if __name__ =="__main__" :
    main()
