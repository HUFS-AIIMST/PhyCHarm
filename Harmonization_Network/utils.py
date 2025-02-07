import os
from glob import glob
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import Dataset


def get_subjects_list(setting_dict, data_root): #get small number of subjects
    source_vender = setting_dict['source']
    target_vender = setting_dict['target']

    source_subjects_list = glob(f'{data_root}/T1w/Sub*/{source_vender}/*.nii.gz') 
    target_subjects_list = glob(f'{data_root}/T1w/Sub*/{target_vender}/*.nii.gz')

    source_subjects_list = [x.split('/')[-3] for x in source_subjects_list]
    target_subjects_list = [x.split('/')[-3] for x in target_subjects_list]

    subjects_list = sorted(list(set(source_subjects_list).intersection(target_subjects_list)))
    k_list = [int(x.split('Sub')[1]) for x in subjects_list]

    return k_list

class CustomDataset(Dataset):
    def __init__(self, data_root, setting_dict, k_list, num_k, num_z_slice, train=True):
        self.data_root = data_root
        self.source_vender = setting_dict['source']
        self.target_vender = setting_dict['target']

        self.dataset = []
        for k in k_list:
            if train:
                if k == num_k:
                    continue
            else:
                if k != num_k:
                    continue
           
            for z in range(num_z_slice):
                self.dataset.append([k, z])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        k, z = self.dataset[idx]
        
        source_path = f'{self.data_root}/T1w/Sub{k}/{self.source_vender}/Sub{k}_{self.source_vender[0]}.nii.gz'
        target_path = f'{self.data_root}/T1w/Sub{k}/{self.target_vender}/Sub{k}_{self.target_vender[0]}.nii.gz'

        mask_path = f'{self.data_root}/mask/Sub{k}'
        mask_file = glob(f'{mask_path}/*.nii.gz')
        mask_path=f'{mask_file[0]}'
        
        data_A = nib.load(source_path).get_fdata()[None, ..., z].astype(np.float32)
        data_B = nib.load(target_path).get_fdata()[None, ..., z].astype(np.float32)
        mask = nib.load(mask_path).get_fdata()[None, ..., z].astype(np.uint8)

        data_A = torch.from_numpy(data_A)
        data_B = torch.from_numpy(data_B)
        mask = torch.from_numpy(mask)
        
        return data_A, data_B, mask


class CustomDataset_inference(Dataset):
    def __init__(self, data_root, setting_dict, num_z_slice):
        self.data_root = data_root
        self.source_vender = setting_dict['source']

        self.dataset = []
        for z in range(num_z_slice):
            self.dataset.append(z)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        z = self.dataset[idx]
        
        source_path = glob(f'{self.data_root}/*.nii.gz')[0]
        mask_path = glob(f'{self.data_root}/*_mask.nii.gz')[0]

        data_A = nib.load(source_path).get_fdata().astype(np.float32)
        mask = nib.load(mask_path).get_fdata().astype(np.uint8)

        data_A = data_A*mask
        
        data_A = nib.load(source_path).get_fdata()[None, ..., z].astype(np.float32)
        mask = nib.load(mask_path).get_fdata()[None, ..., z].astype(np.uint8)
        
        data_A = torch.from_numpy(data_A)
        mask = torch.from_numpy(mask)
        
        return data_A, mask

def Bloch_eq(t1map,m0map,opt):
    """
    This calculates bloch equation for consT1w
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

def cal_consT1w(slide_source, pred_T1map_s, pred_M0map_s, pred_T1map_t, pred_M0map_t, setting_dict):
    # caldulation of constrained T1w
    if setting_dict['target'] =='Siemens':
        if setting_dict['source']=='GE':
            opt_target = 1
            opt_source = 3

            cons_source, TE = Bloch_eq(pred_T1map_s, pred_M0map_s, opt_source) # Eq.7 in the article.
            T2term_source = slide_source/cons_source # Eq.8 in the article.

            T2map = -TE/torch.log(T2term_source) # Eq.9 in the article.
            T2map[T2term_source==0]=0

            cons_target, TE = Bloch_eq(pred_T1map_s, pred_M0map_s, opt_target)  # Eq.10 in the article. (source to target)
            cons_target = cons_target*torch.exp(-TE/(T2map+1e-1)) # Eq.10 in the article. (source to target)

            cons_source, TE = Bloch_eq(pred_T1map_t, pred_M0map_t, opt_source) # Eq.10 in the article. (target to source)
            cons_source = cons_source*torch.exp(-TE/(T2map+1e-1)) # Eq.10 in the article. (target to source)

        elif setting_dict['source']=='Philips':
            opt_target=1
            opt_source=2

            cons_source, TE = Bloch_eq(pred_T1map_s, pred_M0map_s, opt_source) # Eq.7 in the article.
            T2term_source = slide_source/cons_source # Eq.8 in the article.

            T2map = -TE/torch.log(T2term_source) # Eq.9 in the article.
            T2map[T2term_source==0]=0

            cons_target, TE = Bloch_eq(pred_T1map_s, pred_M0map_s, opt_target)  # Eq.10 in the article. (source to target)
            cons_target = cons_target*torch.exp(-TE/(T2map+1e-1)) # Eq.10 in the article. (source to target)

            cons_source, TE = Bloch_eq(pred_T1map_t, pred_M0map_t, opt_source) # Eq.10 in the article. (target to source)
            cons_source = cons_source*torch.exp(-TE/(T2map+1e-1)) # Eq.10 in the article. (target to source)

    
    elif setting_dict['target']=='Philips':
        if setting_dict['source']=='GE':
            opt_target = 2
            opt_source = 3

            cons_source, TE = Bloch_eq(pred_T1map_s, pred_M0map_s, opt_source) # Eq.7 in the article.
            T2term_source = slide_source/cons_source # Eq.8 in the article.

            T2map = -TE/torch.log(T2term_source) # Eq.9 in the article.
            T2map[T2term_source==0]=0

            cons_target, TE = Bloch_eq(pred_T1map_s, pred_M0map_s, opt_target)  # Eq.10 in the article. (source to target)
            cons_target = cons_target*torch.exp(-TE/(T2map+1e-1)) # Eq.10 in the article. (source to target)

            cons_source, TE = Bloch_eq(pred_T1map_t, pred_M0map_t, opt_source) # Eq.10 in the article. (target to source)
            cons_source = cons_source*torch.exp(-TE/(T2map+1e-1)) # Eq.10 in the article. (target to source)


        elif setting_dict['source']=='Siemens':
            opt_target=2
            opt_source=1

            cons_source, TE = Bloch_eq(pred_T1map_s, pred_M0map_s, opt_source) # Eq.7 in the article.
            T2term_source = slide_source/cons_source  # Eq.8 in the article.

            T2map = -TE/torch.log(T2term_source) # Eq.9 in the article.
            T2map[T2term_source==0]=0

            cons_target, TE = Bloch_eq(pred_T1map_s, pred_M0map_s, opt_target)   # Eq.10 in the article. (source to target)
            cons_target = cons_target*torch.exp(-TE/(T2map+1e-1)) # Eq.10 in the article. (source to target)

            cons_source, TE = Bloch_eq(pred_T1map_t, pred_M0map_t, opt_source) # Eq.10 in the article. (target to source)
            cons_source = cons_source*torch.exp(-TE/(T2map+1e-1)) # Eq.10 in the article. (target to source)

    elif setting_dict['target']=='GE':
        if setting_dict['source']=='Siemens':
            opt_target = 3
            opt_source = 1

            cons_source, TE = Bloch_eq(pred_T1map_s, pred_M0map_s, opt_source) # Eq.7 in the article.
            T2term_source = slide_source/cons_source # Eq.8 in the article.

            T2map = -TE/torch.log(T2term_source)  # Eq.9 in the article.
            T2map[T2term_source==0]=0

            cons_target, TE = Bloch_eq(pred_T1map_s, pred_M0map_s, opt_target)    # Eq.10 in the article. (source to target)
            cons_target = cons_target*torch.exp(-TE/(T2map+1e-1))   # Eq.10 in the article. (source to target)

            cons_source, TE = Bloch_eq(pred_T1map_t, pred_M0map_t, opt_source) # Eq.10 in the article. (target to source)
            cons_source = cons_source*torch.exp(-TE/(T2map+1e-1))  # Eq.10 in the article. (target to source)


        elif setting_dict['source']=='Philips':
            opt_target=3
            opt_source=2

            cons_source, TE = Bloch_eq(pred_T1map_s, pred_M0map_s, opt_source) # Eq.7 in the article.
            T2term_source = slide_source/cons_source # Eq.8 in the article.

            T2map = -TE/torch.log(T2term_source) # Eq.9 in the article.
            T2map[T2term_source==0]=0

            cons_target, TE = Bloch_eq(pred_T1map_s, pred_M0map_s, opt_target)  # Eq.10 in the article. (source to target)
            cons_target = cons_target*torch.exp(-TE/(T2map+1e-1)) # Eq.10 in the article. (source to target)

            cons_source, TE = Bloch_eq(pred_T1map_t, pred_M0map_t, opt_source)  # Eq.10 in the article. (target to source)
            cons_source = cons_source*torch.exp(-TE/(T2map+1e-1))  # Eq.10 in the article. (target to source)
    
    return cons_source, cons_target

def save_loss_graph(save_fig_dir, k, mean_hist, val_mean_hist):

    plt.figure(figsize=(7,5))
    plt.title('Loss Progress_Generator')
    plt.plot(mean_hist['target'], label='Train Loss')
    plt.plot(val_mean_hist['target'], label='Val Loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_fig_dir}/ST_loss_valid_{k}', dpi=300)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.title('Loss Progress_Generator')
    plt.plot(mean_hist['source'], label='Train Loss')
    plt.plot(val_mean_hist['source'], label='Val Loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_fig_dir}/TS_loss_valid_{k}', dpi=300)
    plt.close()
        

def save_figure(setting_dict, save_fig_dir, k, real_a, real_b, fake_a, fake_b, pred_T1, pred_M0, cons_a, cons_b):
    s=130
    source = setting_dict['source']
    target = setting_dict['target']

    diff_a = 100*np.abs(real_a - fake_a)/(real_a.max()-real_a.min())
    diff_b = 100*np.abs(real_b - fake_b)/(real_b.max()-real_b.min())

    diff_cons_a = 100*np.abs(real_a - cons_a)/(real_a.max()-real_a.min())
    diff_cons_b = 100*np.abs(real_b - cons_b)/(real_b.max()-real_b.min())
 
    fig, [axes1, axes2, axes3] = plt.subplots(3,3)
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    axes1[0].axis('off')
    
    axes1[1].imshow(np.rot90(pred_T1[...,s],-1), 'gray', vmin=0, vmax=pred_T1.max())
    axes1[1].plot([1],[1])
    axes1[1].tick_params(axis=u'both', which=u'both',length=0)
    axes1[1].set_xticks([])
    axes1[1].set_yticks([])
    axes1[1].set_title('Pred T1',rotation='vertical', x=-0.1, y=0.1)

    axes1[2].imshow(np.rot90(pred_M0[...,s],-1), 'gray', vmin=0, vmax=pred_M0.max())
    axes1[2].plot([1],[1])
    axes1[2].tick_params(axis=u'both', which=u'both',length=0)
    axes1[2].set_xticks([])
    axes1[2].set_yticks([])
    axes1[2].set_title('Pred M0',rotation='vertical',x=-0.1, y=0.1)

    axes2[0].imshow(np.rot90(real_a[...,s],-1), 'gray', vmin=0, vmax=real_a.max())
    axes2[0].plot([1],[1])
    axes2[0].tick_params(axis=u'both', which=u'both',length=0)
    axes2[0].set_xticks([])
    axes2[0].set_yticks([])
    axes2[0].set_title(f'GT {source}',rotation='vertical',x=-0.1, y=-0.2)

    axes2[1].imshow(np.rot90(cons_a[...,s],-1), 'gray', vmin=0, vmax=cons_a.max())
    axes2[1].plot([1],[1])
    axes2[1].tick_params(axis=u'both', which=u'both',length=0)
    axes2[1].set_xticks([])
    axes2[1].set_yticks([])
    axes2[1].set_title(f'Cons\n{source}', rotation='vertical',x=-0.1, y=0.2)

    axes2[2].imshow(np.rot90(fake_a[...,s],-1), 'gray', vmin=0, vmax=real_a.max())
    axes2[2].plot([1],[1])
    axes2[2].tick_params(axis=u'both', which=u'both',length=0)
    axes2[2].set_xticks([])
    axes2[2].set_yticks([])
    axes2[2].set_title(f'Denoised\n{source}', rotation='vertical',x=-0.1, y=0.2)

    axes3[0].axis('off')

    axes3[1].imshow(np.rot90(diff_cons_a[...,s],-1), 'jet', vmin=0, vmax=100)
    axes3[1].plot([1],[1])
    axes3[1].tick_params(axis=u'both', which=u'both',length=0)
    axes3[1].set_xticks([])
    axes3[1].set_yticks([])

    axes3[2].imshow(np.rot90(diff_a[...,s],-1), 'jet', vmin=0, vmax=100)
    axes3[2].plot([1],[1])
    axes3[2].tick_params(axis=u'both', which=u'both',length=0)
    axes3[2].set_xticks([])
    axes3[2].set_yticks([])

    plt.savefig(f'{save_fig_dir}/Source_results_valid_{k}.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, [axes1, axes2, axes3] = plt.subplots(3,3)
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    axes1[0].axis('off')
    
    axes1[1].imshow(np.rot90(pred_T1[...,s],-1), 'gray', vmin=0, vmax=pred_T1.max())
    axes1[1].plot([1],[1])
    axes1[1].tick_params(axis=u'both', which=u'both',length=0)
    axes1[1].set_xticks([])
    axes1[1].set_yticks([])
    axes1[1].set_title('Pred T1', rotation='vertical',x=-0.1, y=0.1)

    axes1[2].imshow(np.rot90(pred_M0[...,s],-1), 'gray', vmin=0, vmax=pred_M0.max())
    axes1[2].plot([1],[1])
    axes1[2].tick_params(axis=u'both', which=u'both',length=0)
    axes1[2].set_xticks([])
    axes1[2].set_yticks([])
    axes1[2].set_title('Pred M0', rotation='vertical',x=-0.1, y=0.1)

    axes2[0].imshow(np.rot90(real_b[...,s],-1), 'gray', vmin=0, vmax=real_b.max())
    axes2[0].plot([1],[1])
    axes2[0].tick_params(axis=u'both', which=u'both',length=0)
    axes2[0].set_xticks([])
    axes2[0].set_yticks([])
    axes2[0].set_title(f'GT {target}', rotation='vertical',x=-0.1, y=0.2)

    axes2[1].imshow(np.rot90(cons_b[...,s],-1), 'gray', vmin=0, vmax=cons_b.max())
    axes2[1].plot([1],[1])
    axes2[1].tick_params(axis=u'both', which=u'both',length=0)
    axes2[1].set_xticks([])
    axes2[1].set_yticks([])
    axes2[1].set_title(f'Cons\n{target}', rotation='vertical',x=-0.1, y=0.2)

    axes2[2].imshow(np.rot90(fake_b[...,s],-1), 'gray', vmin=0, vmax=real_b.max())
    axes2[2].plot([1],[1])
    axes2[2].tick_params(axis=u'both', which=u'both',length=0)
    axes2[2].set_xticks([])
    axes2[2].set_yticks([])
    axes2[2].set_title(f'Denoised\n{target}', rotation='vertical',x=-0.1, y=0.2)

    axes3[0].axis('off')

    axes3[1].imshow(np.rot90(diff_cons_b[...,s],-1), 'jet', vmin=0, vmax=100)
    axes3[1].plot([1],[1])
    axes3[1].tick_params(axis=u'both', which=u'both',length=0)
    axes3[1].set_xticks([])
    axes3[1].set_yticks([])

    axes3[2].imshow(np.rot90(diff_b[...,s],-1), 'jet', vmin=0, vmax=100)
    axes3[2].plot([1],[1])
    axes3[2].tick_params(axis=u'both', which=u'both',length=0)
    axes3[2].set_xticks([])
    axes3[2].set_yticks([])

    plt.savefig(f'{save_fig_dir}/Target_results_valid_{k}.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_model_state(save_pth_dir, k, epoch, model, best=False):
    save_pth_path = f'{save_pth_dir}/valid_{k}_best.pth' if best else f'{save_pth_dir}/valid_{k}/epoch_{epoch}.pth'
    os.makedirs(save_pth_dir, exist_ok=True)
    os.makedirs(f'{save_pth_dir}/valid_{k}', exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        },
        save_pth_path
        )
    
    
def save_data(save_data_dir, data_root, setting_dict, k, fake_a, fake_b):
     target_vender = setting_dict['target']

     target_data = f'{data_root}/T1w/Sub{k}/{target_vender}/Sub{k}_{target_vender[0]}.nii.gz'
     target_data = nib.load(target_data)

     save_data_path = f'{save_data_dir}/Sub{k}_{target_vender[0]}.nii.gz'
     os.makedirs(save_data_dir, exist_ok=True)
     os.makedirs(f'{save_data_dir}/valid_{k}', exist_ok=True)

     save_output = nib.Nifti1Image(fake_b, target_data.affine, target_data.header)
     nib.save(save_output, f'{save_data_path}')

     source_vender = setting_dict['source']
     source_data = f'{data_root}/T1w/Sub{k}/{source_vender}/Sub{k}_{source_vender[0]}.nii.gz'

     source_data = nib.load(source_data)

     save_data_path = f'{save_data_dir}/Sub{k}_{source_vender[0]}.nii.gz'
     os.makedirs(save_data_dir, exist_ok=True)
     os.makedirs(f'{save_data_dir}/valid_{k}', exist_ok=True)

     save_output = nib.Nifti1Image(fake_a, source_data.affine, source_data.header)
     nib.save(save_output, f'{save_data_path}')


def save_inference_data(save_inference_dir, data_root, setting_dict, volume_pred_T1, volume_pred_M0, volume_cons_target, volume_harmonized_target):
    
     os.makedirs(save_inference_dir, exist_ok=True)

     source_vender = setting_dict['source']
     target_vender = setting_dict['target']
     
     header_data = glob(f'{data_root}/*.nii.gz')[0]
     file_name = header_data.split('/')[-1]
     file_name = file_name.split('.')[0]
     header_data = nib.load(header_data)

     save_T1_path = f'{save_inference_dir}/{file_name}_T1.nii.gz'
     save_M0_path = f'{save_inference_dir}/{file_name}_M0.nii.gz'
     save_harmonized_data_path = f'{save_inference_dir}/{file_name}_harmonized_{target_vender}.nii.gz'
     
     save_output = nib.Nifti1Image(volume_pred_T1, header_data.affine, header_data.header)
     nib.save(save_output, f'{save_T1_path}')

     save_output = nib.Nifti1Image(volume_pred_M0, header_data.affine, header_data.header)
     nib.save(save_output, f'{save_M0_path}')

     #save_cons_path = f'{save_inference_dir}/{file_name}_consT1w.nii.gz' 
     #save_output = nib.Nifti1Image(volume_cons_target, header_data.affine, header_data.header)
     #nib.save(save_output, f'{save_cons_path}')

     save_output = nib.Nifti1Image(volume_harmonized_target, header_data.affine, header_data.header)
     nib.save(save_output, f'{save_harmonized_data_path}')
