import os
from glob import glob
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import torch
import natsort
from torch.utils.data import Dataset



from mpl_toolkits.axes_grid1 import make_axes_locatable
def make_colorbar_with_padding(ax):
    """
    Create colorbar axis that fits the size of a plot - detailed here: http://chris35wills.github.io/matplotlib_axis/
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    return(cax) 


class CustomDataset(Dataset):
    def __init__(self, data_root, k_list, num_k, num_z_slice, train):
        self.data_root = data_root

        self.dataset = []
        for k in k_list:
            if train:
                if k in num_k:
                    continue
            else:
                if k not in num_k:
                    continue
           
            for z in range(num_z_slice):
                self.dataset.append([k, z])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        k, z = self.dataset[idx]
        
        T1w_path = f'{self.data_root}/T1w*.nii.gz'
        T1map_path = f'{self.data_root}/T1map*.nii.gz'
        M0map_path = f'{self.data_root}/M0map*.nii.gz'
        mask_path = f'{self.data_root}/T1w*.nii.gz'
        
        T1w = nib.load(T1w_path).get_fdata()[None, ..., z].astype(np.float32)
        T1map = nib.load(T1map_path).get_fdata()[None, ..., z].astype(np.float32)
        M0map = nib.load(M0map_path).get_fdata()[None, ..., z].astype(np.float32)
        mask = nib.load(mask_path).get_fdata()[None, ..., z].astype(np.uint8)

        T1w = torch.from_numpy(T1w)
        T1map = torch.from_numpy(T1map)
        M0map = torch.from_numpy(M0map)
        mask = torch.from_numpy(mask)
        
        return T1w, T1map, M0map,mask
    
    
def show_train_results(save_fig_dir, k, train_GT_T1map, train_pred_T1map, train_GT_M0map,train_pred_M0map ):
        
        train_diff_gen_T1map=100*np.abs(train_GT_T1map-train_pred_T1map)/(train_GT_T1map.max()-train_GT_T1map.min())
        train_diff_gen_T1map[train_GT_T1map==0]=0

        train_diff_gen_M0map=100*np.abs(train_GT_M0map-train_pred_M0map)/(train_GT_M0map.max()-train_GT_M0map.min())
        train_diff_gen_M0map[train_GT_M0map==0]=0

        plt.figure(dpi=200)
        ax1=plt.subplot(2,3,1)
        plt.title('train_GT_T1map', fontsize=5)
        plt.imshow(train_GT_T1map[0,0,:,:],cmap='gray',vmin=0,vmax=0.8*train_GT_T1map.max())        
        plt.axis('off')
        cax1=make_colorbar_with_padding(ax1)
        plt.colorbar(cax=cax1)

        ax1=plt.subplot(2,3,2)
        plt.imshow(train_pred_T1map[0,0,:,:],cmap='gray',vmin=0,vmax=0.8*train_GT_T1map.max())
        plt.axis('off')
        cax1=make_colorbar_with_padding(ax1)
        plt.colorbar(cax=cax1)

        ax1=plt.subplot(2,3,3)
        plt.imshow(train_diff_gen_T1map[0,0,:,:],cmap='jet',vmin=0,vmax=100)
        plt.axis('off')
        cax1=make_colorbar_with_padding(ax1)
        plt.colorbar(cax=cax1)
        
        ax1=plt.subplot(2,3,4)
        plt.title('train_GT_M0map', fontsize=5)
        plt.imshow(train_GT_M0map[0,0,:,:],cmap='gray',vmin=0, vmax=train_GT_M0map.max())        
        plt.axis('off')
        cax1=make_colorbar_with_padding(ax1)
        plt.colorbar(cax=cax1)

        ax1=plt.subplot(2,3,5)
        plt.imshow(train_pred_M0map[0,0,:,:],cmap='gray',vmin=0, vmax=train_GT_M0map.max())
        plt.axis('off')
        cax1=make_colorbar_with_padding(ax1)
        plt.colorbar(cax=cax1)

        ax1=plt.subplot(2,3,6)
        plt.imshow(train_diff_gen_M0map[0,0,:,:],cmap='jet',vmin=0, vmax=100)
        plt.axis('off')
        cax1=make_colorbar_with_padding(ax1)
        plt.colorbar(cax=cax1)
        plt.tight_layout()
        plt.savefig(f'{save_fig_dir}/results_train_{k}.png', dpi=300, bbox_inches='tight')
        plt.show()

def show_val_results(save_fig_dir,k,val_GT_T1map, val_pred_T1map, val_GT_M0map, val_pred_M0map):
        
        val_diff_gen_T1map=100*np.abs(val_GT_T1map-val_pred_T1map)/(val_GT_T1map.max()-val_GT_T1map.min())
        val_diff_gen_T1map[val_GT_T1map==0]=0
        
        val_diff_gen_M0map=100*np.abs(val_GT_M0map-val_pred_M0map)/(val_GT_M0map.max()-val_GT_M0map.min())
        val_diff_gen_M0map[val_GT_M0map==0]=0

        plt.figure(dpi=200)
        ax1=plt.subplot(2,3,1)
        plt.title(f'valid_GT_T1map', fontsize=5)
        plt.imshow(val_GT_T1map[0,0,:,:],cmap='gray',vmin=0,vmax=val_GT_T1map.max())        
        plt.axis('off')
        cax1=make_colorbar_with_padding(ax1)
        plt.colorbar(cax=cax1)

        ax1=plt.subplot(2,3,2)
        plt.imshow(val_pred_T1map[0,0,:,:],cmap='gray',vmin=0,vmax=val_GT_T1map.max()) 
        plt.axis('off')
        cax1=make_colorbar_with_padding(ax1)
        plt.colorbar(cax=cax1)

        ax1=plt.subplot(2,3,3)
        plt.imshow(val_diff_gen_T1map[0,0,:,:],cmap='jet',vmin=0,vmax=100)
        plt.axis('off')
        cax1=make_colorbar_with_padding(ax1)
        plt.colorbar(cax=cax1)

        ax1=plt.subplot(2,3,4)
        plt.title('valid_GT_M0map', fontsize=5)
        plt.imshow(val_GT_M0map[0,0,:,:],cmap='gray',vmin=0,vmax=val_GT_M0map.max())        
        plt.axis('off')
        cax1=make_colorbar_with_padding(ax1)
        plt.colorbar(cax=cax1)

        ax1=plt.subplot(2,3,5)
        plt.imshow(val_pred_M0map[0,0,:,:],cmap='gray',vmin=0,vmax=val_GT_M0map.max()) 
        plt.axis('off')
        cax1=make_colorbar_with_padding(ax1)
        plt.colorbar(cax=cax1)

        ax1=plt.subplot(2,3,6)
        plt.imshow(val_diff_gen_M0map[0,0,:,:],cmap='jet',vmin=0,vmax=100) 
        plt.axis('off')
        cax1=make_colorbar_with_padding(ax1)
        plt.colorbar(cax=cax1)
        plt.tight_layout()
        plt.savefig(os.path.join(save_fig_dir,'val_results.png'),dpi=300)
        plt.show()

def save_loss_graph(save_fig_dir, k, train_T1_loss, val_T1_loss, train_M0_loss, val_M0_loss, train_total_loss, val_total_loss):
    plt.figure(figsize=(7,5))
    plt.title('T1map Loss Progress')
    plt.plot(train_T1_loss, label='train Loss')
    plt.plot(val_T1_loss, label='val Loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_fig_dir}/T1_loss_valid_{k}.png', dpi=300)
    
    plt.figure(figsize=(7,5))
    plt.title('M0map Loss Progress')
    plt.plot(train_M0_loss, label='train Loss')
    plt.plot(val_M0_loss, label='val Loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_fig_dir}/M0_loss_valid_{k}.png', dpi=300)

    plt.figure(figsize=(7,5))
    plt.title('Total Loss Progress')
    plt.plot(train_total_loss, label='train Loss')
    plt.plot(val_total_loss, label='val Loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_fig_dir}/Total_loss_valid_{k}.png', dpi=300)

def save_loss_graph_no_total(save_fig_dir, k, train_T1_loss, val_T1_loss, train_M0_loss, val_M0_loss):
    plt.figure(figsize=(7,5))
    plt.title('T1map Loss Progress')
    plt.plot(train_T1_loss, label='train Loss')
    plt.plot(val_T1_loss, label='val Loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_fig_dir}/T1_loss_valid_{k}.png', dpi=300)
    
    plt.figure(figsize=(7,5))
    plt.title('M0map Loss Progress')
    plt.plot(train_M0_loss, label='train Loss')
    plt.plot(val_M0_loss, label='val Loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_fig_dir}/M0_loss_valid_{k}.png', dpi=300)



def save_model_state_T1(save_pth_dir, k, epoch, generator_T1map, best=True):
    save_pth_path = f'{save_pth_dir}/valid_{k}_best.pth' if best else f'{save_pth_dir}/valid_{k}/epoch_{epoch}.pth'
    os.makedirs(save_pth_dir, exist_ok=True)
    os.makedirs(f'{save_pth_dir}/valid_{k}', exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'T1map': generator_T1map.state_dict(),
        },
        save_pth_path
        )
    
def save_model_state_M0(save_pth_dir, k, epoch, generator_M0map, best=True):
    save_pth_path = f'{save_pth_dir}/valid_{k}_best.pth' if best else f'{save_pth_dir}/valid_{k}/epoch_{epoch}.pth'
    os.makedirs(save_pth_dir, exist_ok=True)
    os.makedirs(f'{save_pth_dir}/valid_{k}', exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'M0map': generator_M0map.state_dict(),
        },
        save_pth_path
        )