from data_utils import verify_preprocessed_data_aug, PreLoadDataset,  DataTransform, ACPCLandMarkDataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import os, re, torch, time
import argparse
from pathlib import Path
import sys

from loss import SigmoidFocalLoss
from model import UNet3D
import torch.optim as optim
from torch.optim import lr_scheduler

import random, shutil

from sklearn.model_selection import train_test_split, GroupKFold, StratifiedGroupKFold, KFold

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def center_patch_hu(input_pt):
    
    """Performs per channel mean subtraction (the patch mean would be more meaningful as that is what the model sees)"""
    mean_pc_voxel_patch = input_pt[:,0,:,:,:].mean(dim = (1,2,3)).reshape(len(input_pt),1,1,1).repeat(1,
                                                                                                      input_pt.shape[2],
                                                                                                      input_pt.shape[3],
                                                                                                      input_pt.shape[4])
    pc_voxel_patch_norm = input_pt[:,0,:,:,:] - mean_pc_voxel_patch
    
    mean_ac_voxel_patch = input_pt[:,1,:,:,:].mean(dim = (1,2,3)).reshape(len(input_pt),1,1,1).repeat(1,
                                                                                                      input_pt.shape[2],
                                                                                                      input_pt.shape[3],
                                                                                                      input_pt.shape[4])
    ac_voxel_patch_norm = input_pt[:,1,:,:,:] - mean_ac_voxel_patch
    
    return torch.stack([pc_voxel_patch_norm, ac_voxel_patch_norm], axis = 1), (mean_ac_voxel_patch.mean(), mean_pc_voxel_patch.mean())

def pre_load_data_transform(scan_path, heatmap_path, labels, patch_size): 
    
    """The labels, input, and groundtruth patches are read and stacked as tensors"""
    v_x, v_y, v_z = patch_size, patch_size, patch_size
    labels = torch.tensor(labels)
        
    #Read input voxel patches around predicted AC and PC
    
    input_pat = torch.tensor(np.load(scan_path).reshape(2, v_z, v_y, v_x))
    
    #Generate output heatmaps around predicted AC and PC
    gt_pat = torch.tensor(np.load(heatmap_path).reshape(4, v_z, v_y, v_x))
                                             
    
    return input_pat, gt_pat, labels



def infer_landmark_preds(preds, ac_sampling_origin, pc_sampling_origin, device):

    """
    Returns predicted AC and PC image coordinates as the mean location of active voxels within the AC-channel's AC patch and the PC-channel's PC patch,
    respectively. 
    
    Args: 
        preds: Predicted heatmaps of dimension batch_size x 6 (channel dim) x patch_size x patch_size x patch_size. 
               Predicted heatmaps will be stacked corresponding to the ground-truth heatmaps.
               Channels 0, 1, and 2 correspond to the AC, PC, and background channels in the PC-patch. 
               Channels 3, 4, and 5 correspond to the AC, PC, and background channels in the AC-patch. 
        ac_sampling_origin: Image coordinates of the origin of the AC-patch, computed in reference to the coarse localized AC and patch size
        pc_sampling_origin: Image coordinates of the origin of the PC-patch, computed in reference to the coarse localized PC and patch size
        
    """

    #Grab predictions from the AC-patch's AC channel
    ac_channel_preds = preds[:,[3],:,:,:] 
    #Grab predictions from the PC-patch's PC channel
    pc_channel_preds = preds[:,[1],:,:,:] 

    if ac_channel_preds.isnan().any():
        sys.exit('NAs seen in model AC predictions, debug your training')
    if pc_channel_preds.isnan().any():
        sys.exit('NAs seen in model PC predictions, debug your training')

    #Min-max normalize the AC-patch AC-channel's activations
    ac_channel_preds_min = ac_channel_preds.amin(axis = (1,2,3,4)).reshape(ac_channel_preds.shape[0],1,1,1,1).repeat(
                                1,1,ac_channel_preds.shape[2],ac_channel_preds.shape[3],ac_channel_preds.shape[4])
    ac_channel_preds_min_max_norm = (ac_channel_preds - ac_channel_preds_min)/(
                                     ac_channel_preds.amax(axis = (1,2,3,4)).reshape(ac_channel_preds.shape[0],1,1,1,1).repeat(
                                1,1,ac_channel_preds.shape[2],ac_channel_preds.shape[3],ac_channel_preds.shape[4])- ac_channel_preds_min)
    
    #Min-max normalize the PC-patch AC-channel's activations
    pc_channel_preds_min =   pc_channel_preds.amin(axis = (1,2,3,4)).reshape(pc_channel_preds.shape[0],1,1,1,1).repeat(
                                1,1,pc_channel_preds.shape[2],pc_channel_preds.shape[3],pc_channel_preds.shape[4])
    pc_channel_preds_min_max_norm = (pc_channel_preds- pc_channel_preds_min)/(
                                     pc_channel_preds.amax(axis = (1,2,3,4)).reshape(pc_channel_preds.shape[0],1,1,1,1).repeat(
                                1,1,pc_channel_preds.shape[2],pc_channel_preds.shape[3],pc_channel_preds.shape[4])- pc_channel_preds_min)
    

    ac_channel_preds_min_max_norm = ac_channel_preds_min_max_norm.cpu().detach().numpy()
    pc_channel_preds_min_max_norm = pc_channel_preds_min_max_norm.cpu().detach().numpy()
    
    #Determine the average location of active voxels in the AC-patch's AC channel. 
    ac_channel_ac_patch_ind = [np.array(np.where(ac_channel_preds_min_max_norm[b,0,:,:,:] > 0.5))[::-1].mean(axis = 1) for b in range(preds.shape[0])]
    #Determine the average location of active voxels in the PC-patch's PC channel. 
    pc_channel_pc_patch_ind = [np.array(np.where(pc_channel_preds_min_max_norm[b,0,:,:,:] > 0.5))[::-1].mean(axis = 1) for b in range(preds.shape[0])]
    
    #Convert the predicted AC location in the voxel patch's scope to the full image's scope by translating with the AC-patch's origin
    ac_img_ind = [x+y for x,y in zip(ac_channel_ac_patch_ind,ac_sampling_origin)] 
    
    #Convert the predicted PC location in the voxel patch's scope to the full image's scope by translating with the PC-patch's origin    
    pc_img_ind = [x+y for x,y in zip(pc_channel_pc_patch_ind,pc_sampling_origin)]
    
    return (ac_img_ind, pc_img_ind) 

def mre(ac_true, pc_true, ac, pc):
    """
    Returns the mean radial error (MRE) between the predicted and reference-standard (ground-truth) AC and PC coordinates for a batch of data
    Also returns absolute coordinate-specific errors (x, y, and z) for both landmarks
    Assumes isotropic 1mm x 1mm x 1 mm spacing, which means the resultant MREs can be interpreted directly in mm. 
    Args:
        ac_true: Image coordinates of reference-standard AC (expected as tensor/array?)
        pc_true: Image coordinates of reference-standard PC 
        ac: Image coordinates of predicted AC
        pc: Image coordinates of predicted PC
    """
        
    ac_mre = 0
    pc_mre = 0
    ac_z_err, ac_y_err, ac_x_err = 0,0,0
    pc_z_err, pc_y_err, pc_x_err = 0,0,0
    for ac_true_, pc_true_, ac_, pc_ in zip(ac_true, pc_true, ac, pc):
        ac_ = [temp for temp in ac_]
        pc_ = [temp for temp in pc_]
        ac_true_ = [temp.cpu() for temp in ac_true_]
        pc_true_ = [temp.cpu() for temp in pc_true_] 
    
        ac_mre += np.sqrt((ac_[0] - ac_true_[0])**2 + (ac_[1] - ac_true_[1])**2  + (ac_[2] - ac_true_[2])**2) 
        pc_mre += np.sqrt((pc_[0] - pc_true_[0])**2 + (pc_[1] - pc_true_[1])**2  + (pc_[2] - pc_true_[2])**2)

        ac_x_err += np.abs(ac_[0] - ac_true_[0])
        ac_y_err += np.abs(ac_[1] - ac_true_[1])
        ac_z_err += np.abs(ac_[2] - ac_true_[2])
        pc_x_err += np.abs(pc_[0] - pc_true_[0])
        pc_y_err += np.abs(pc_[1] - pc_true_[1])
        pc_z_err += np.abs(pc_[2] - pc_true_[2])

    ac_mre = ac_mre/len(ac_true)
    pc_mre = pc_mre/len(pc_true)

    ac_z_err = ac_z_err/len(ac_true)
    ac_y_err = ac_y_err/len(ac_true)
    ac_x_err = ac_x_err/len(ac_true)
    
    pc_z_err = pc_z_err/len(pc_true)
    pc_y_err = pc_y_err/len(pc_true)
    pc_x_err = pc_x_err/len(pc_true)


    return (ac_mre + pc_mre)/2, ac_mre, pc_mre, ac_x_err, ac_y_err, ac_z_err, pc_x_err, pc_y_err, pc_z_err

def load_data(input_tensors, 
              output_tensors, 
              label_tensors,
              scan_id_map,
              sigma, 
              patch_size, 
              pre_calc_scaling_factors, 
              selected_scans_cv, 
              img_dir): 

    """
    Returns the ACPCLandmarkDataset by assembling the input and ground-truth samples into training batches

    Args: 
        input_tensors: The long tensor stacked with the patched/cropped input images, which is already read into CPU memory
        output_tensors: The long tensor stacked with the patched ground-truth heatmaps, already read into CPU memory. 
        label_tensors: The long tensor stacked with the image coordinates of the reference-standard and coarse-localized AC-PC landmarks,
                       read into CPU memory. 
        scan_id_map: mapping between the accession numbers (scans) and their index in the long (input, output, label) tensors
        sigma: selected hyper-parameter for the standard deviation of the Gaussian
        patch_size: fixed patch size for training
        pre_calc_scaling_factors: dataframe containing precomputed scaling factors for the Gaussian heatmaps across different sigma values
        selected_scans_cv: list of scan identifiers selected for each cross-validation fold
        img_dir: location of the preprocessed images
        rot_id: rotational augmentation selected for training. Default is 0 (no augmentations during hyperparameter tuning)
    """

    #This modulates the heatmap patches with sigma, normalizes them
    data_transform = DataTransform(sigma, pre_calc_scaling_factors) 

    selected_indices = [scan_id_map[str((img_dir.parent.resolve() / x))] for x in selected_scans_cv]

    
    dataset = ACPCLandMarkDataset(input_tensors[selected_indices],
                                  output_tensors[selected_indices], 
                                  label_tensors[selected_indices], 
                                  [str((img_dir.parent.resolve() / x)) for x in selected_scans_cv],     
                                  patch_size, 
                                  data_transform)

    return dataset

def train_model(config, data_dict = None):
    #unpack data_dict
    input_tensors = data_dict["input_tensors"] 
    output_tensors = data_dict["output_tensors"]                                                          
    label_tensors = data_dict["label_tensors"] 
    scan_id_map = data_dict["scan_id_map"]
    
    train_df = data_dict["data_df_train"]
    test_df = data_dict["data_df_test"]    
    outer_fold = data_dict["outer_fold"]
    pre_calc_scaling_factors = data_dict["pre_calc_scaling_factors"]
    results_folder = data_dict["results_folder"]
    img_dir = data_dict["img_dir"]
  
    #from previously selected values
    num_epochs = config["num_epochs"]  
    num_batches = config["batch_size"]
    voxel_center_offset = config["patch_size"]//2
    sigma = config["sigma"]
    patch_size = config["patch_size"]
    alpha = config["alpha"]
    gamma = config["gamma"]
    min_thresh = config["min_thresh"]    
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {'num_workers': 0, 'pin_memory': True} if device=='cuda' else {}
        
    criterion = SigmoidFocalLoss(alpha, gamma, min_thresh, "mean")          

    train_dataset = load_data(input_tensors, output_tensors, label_tensors, scan_id_map, sigma, patch_size, pre_calc_scaling_factors, train_df['scan_path'].values, img_dir)
    
    train_dataloader = DataLoader(train_dataset, batch_size = num_batches, shuffle = True, **kwargs)
    train_len = len(train_dataloader)

    test_dataset = load_data(input_tensors, output_tensors, label_tensors, scan_id_map, sigma, patch_size, pre_calc_scaling_factors, test_df['scan_path'].values, img_dir)
    
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False, **kwargs)
    test_len = len(test_dataloader)

    #for each epoch - to make sure that the loss is going down, model is learning.. 
    train_loss_epochs = []
    train_mre_epochs = []
    train_ac_mre_epochs = []
    train_pc_mre_epochs = []
    train_ac_x_err_epochs = []
    train_ac_y_err_epochs = []
    train_ac_z_err_epochs = []
    train_pc_x_err_epochs = []
    train_pc_y_err_epochs = []
    train_pc_z_err_epochs = []

    track_train_set_predictions_path_outer = os.path.join(results_folder, 'tracked_train_set_predictions_outer.csv')
    track_train_set_metrics_path_outer = os.path.join(results_folder, 'tracked_train_set_metrics_outer.csv')

    torch.manual_seed(46)
    unet_model = UNet3D(in_channels = 2, out_channels = 6)
    unet_model = unet_model.to(device)
    optimizer = optim.SGD(unet_model.parameters(), lr=config["lr"], momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(num_epochs):
        
        epoch_start_time = time.time()
        train_running_loss = 0.0
        train_running_mre = 0.0
        train_running_ac_mre = 0.0
        train_running_pc_mre = 0.0
        train_running_ac_x_err = 0.0
        train_running_ac_y_err = 0.0
        train_running_ac_z_err = 0.0
        train_running_pc_x_err = 0.0
        train_running_pc_y_err = 0.0
        train_running_pc_z_err = 0.0
        train_nf = 0
        
        if epoch == 0:
            train_mean_hus_ac = []
            train_mean_hus_pc = []
        unet_model.train()  # Set model to training mode
        # Each epoch has a training and validation phase
        batch_num  = 0

        torch.manual_seed(23 + epoch)
        for data in train_dataloader:
            
            inputs_unnorm, targets, labels = data['image'].to(device), data['target'].to(device), data['label'].to(device)

            inputs, mean_hus = center_patch_hu(inputs_unnorm.float())
               
            if epoch == 0:                    
                train_mean_hus_ac = train_mean_hus_ac + [mean_hus[0].item()]
                train_mean_hus_pc = train_mean_hus_pc + [mean_hus[1].item()]
            
            inputs, targets = inputs.float(), targets.float() #norm
                                         
            #true coordinates are required to compute the radial error statistics an coarse predictions to compute
            #fine predicted locations
            ac_true, ac = labels[:,:3], labels[:,3:6] 
            pc_true, pc = labels[:,6:9], labels[:,9:]

            # zero the parameter gradients
            optimizer.zero_grad()          

            outputs = unet_model(inputs) 
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            ac_sampling_origin = torch.tensor([[int(x - voxel_center_offset), int(y - voxel_center_offset), int(z - voxel_center_offset)] for x,y,z in ac]).cpu().numpy()
            pc_sampling_origin = torch.tensor([[int(x - voxel_center_offset), int(y - voxel_center_offset), int(z - voxel_center_offset)] for x,y,z in pc]).cpu().numpy()            
  
            ac_pred, pc_pred = infer_landmark_preds(outputs, ac_sampling_origin, pc_sampling_origin, device)

            if epoch == num_epochs - 1:
                ac_predicted_coordinate_df = pd.DataFrame(ac_pred)
                ac_predicted_coordinate_df.columns = ['ac_x','ac_y','ac_z']
                ac_predicted_coordinate_df['scan_id'] = [x.split('/')[-2] for x in data['scan_path']]
                
                
                pc_predicted_coordinate_df = pd.DataFrame(pc_pred)            
                pc_predicted_coordinate_df.columns = ['pc_x','pc_y','pc_z']
                pc_predicted_coordinate_df['scan_id'] = [x.split('/')[-2] for x in data['scan_path']]

                predicted_coordinates_df = ac_predicted_coordinate_df.merge(pc_predicted_coordinate_df, how = 'inner', on = 'scan_id')
                predicted_coordinates_df['outer_fold'] = outer_fold
                predicted_coordinates_df['batch_num'] = batch_num
                predicted_coordinates_df.to_csv(track_train_set_predictions_path_outer, mode='a', header=not os.path.exists(track_train_set_predictions_path_outer))

            batch_num = batch_num + 1
            mre_, ac_mre_, pc_mre_, ac_x_err, ac_y_err, ac_z_err, pc_x_err, pc_y_err, pc_z_err = mre(ac_true, pc_true, ac_pred, pc_pred)

            train_running_loss += loss.item() * inputs.size(0)
            train_running_mre += mre_ * inputs.size(0)
            train_running_ac_mre += ac_mre_ * inputs.size(0)
            train_running_pc_mre += pc_mre_ * inputs.size(0)
            train_running_ac_x_err += ac_x_err * inputs.size(0)
            train_running_ac_y_err += ac_y_err * inputs.size(0)
            train_running_ac_z_err += ac_z_err * inputs.size(0)
            train_running_pc_x_err += pc_x_err * inputs.size(0)
            train_running_pc_y_err += pc_y_err * inputs.size(0)
            train_running_pc_z_err += pc_z_err * inputs.size(0)

            train_nf = train_nf + inputs.size(0)

    
        exp_lr_scheduler.step() 

        train_epoch_loss = train_running_loss/train_nf
        train_epoch_mre = train_running_mre/train_nf
        train_epoch_ac_mre = train_running_ac_mre/train_nf
        train_epoch_pc_mre = train_running_pc_mre/train_nf
        train_epoch_ac_x_err = train_running_ac_x_err/train_nf
        train_epoch_ac_y_err = train_running_ac_y_err/train_nf
        train_epoch_ac_z_err = train_running_ac_z_err/train_nf
        train_epoch_pc_x_err = train_running_pc_x_err/train_nf
        train_epoch_pc_y_err = train_running_pc_y_err/train_nf
        train_epoch_pc_z_err = train_running_pc_z_err/train_nf

        train_loss_epochs = train_loss_epochs + [train_epoch_loss]
        train_mre_epochs = train_mre_epochs + [train_epoch_mre]
        train_ac_mre_epochs = train_ac_mre_epochs + [train_epoch_ac_mre]
        train_pc_mre_epochs = train_pc_mre_epochs + [train_epoch_pc_mre]
        train_ac_x_err_epochs = train_ac_x_err_epochs + [train_epoch_ac_x_err]        
        train_ac_y_err_epochs = train_ac_y_err_epochs + [train_epoch_ac_y_err]
        train_ac_z_err_epochs = train_ac_z_err_epochs + [train_epoch_ac_z_err]
        train_pc_x_err_epochs = train_pc_x_err_epochs + [train_epoch_pc_x_err]
        train_pc_y_err_epochs = train_pc_y_err_epochs + [train_epoch_pc_y_err]        
        train_pc_z_err_epochs = train_pc_z_err_epochs + [train_epoch_pc_z_err]

        epoch_time = time.time() - epoch_start_time
        
        #print statements        
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)            
        print(f"""Train Loss: {train_epoch_loss:.4f} | MRE: {train_epoch_mre:.4f} | AC (MRE,X,Y,Z): {train_epoch_ac_mre:.4f},{train_epoch_ac_x_err:.4f},{train_epoch_ac_y_err:.4f},{train_epoch_ac_z_err:.4f} | PC(MRE,X,Y,Z): {train_epoch_pc_mre:.4f},{train_epoch_pc_x_err:.4f},{train_epoch_pc_y_err:.4f},{train_epoch_pc_z_err:.4f} """)        
        print(f'Epoch Run Time: {epoch_time// 60:.3f}')

    pd.DataFrame({'outer_fold':[outer_fold],
                 'train_loss_epochs':[train_loss_epochs],
                 'train_mre_epochs':[train_mre_epochs],
                 'train_ac_mre_epochs':[train_ac_mre_epochs],
                 'train_pc_mre_epochs':[train_pc_mre_epochs],
                 'train_ac_x_err_epochs':[train_ac_x_err_epochs],
                 'train_ac_y_err_epochs':[train_ac_y_err_epochs],
                 'train_ac_z_err_epochs':[train_ac_z_err_epochs],
                 'train_pc_x_err_epochs':[train_pc_x_err_epochs],
                 'train_pc_y_err_epochs':[train_pc_y_err_epochs],
                 'train_pc_z_err_epochs':[train_pc_z_err_epochs]}).to_csv(track_train_set_metrics_path_outer, mode='a', header=not   os.path.exists(track_train_set_metrics_path_outer))
        
    
    # test set predictions
    predicted_ac_coords = []
    predicted_pc_coords = []
    scan_path_list = []
    test_ac_true_coords = []
    test_pc_true_coords = []
    

    test_mre = []
    test_mre_ac = []
    test_mre_pc = []
    test_ac_x_err = []
    test_ac_y_err = []
    test_ac_z_err = []
    test_pc_x_err = []
    test_pc_y_err = []
    test_pc_z_err = []   
    
    ac_mean_norm = np.mean(train_mean_hus_ac)
    pc_mean_norm = np.mean(train_mean_hus_pc)
    
    unet_model.eval()
    
    for data in test_dataloader:

        inputs_unnorm, targets, labels = data['image'].to(device), data['target'].to(device), data['label'].to(device)
        accnum_list = [x.split('/')[-2] for x in data['scan_path']]
        scan_path_list = scan_path_list + [accnum_list]
    
        inputs = torch.stack([inputs_unnorm[:,0,:,:,:] - pc_mean_norm, inputs_unnorm[:,1,:,:,:] - ac_mean_norm], axis = 1)
        
        inputs, targets = inputs.float(), targets.float()
        with torch.no_grad():            
            outputs = unet_model(inputs)

        ac_true, ac = labels[:,:3], labels[:,3:6] 
        pc_true, pc = labels[:,6:9], labels[:,9:]
        
        ac_sampling_origin = torch.tensor([[int(x - voxel_center_offset), int(y - voxel_center_offset), int(z - voxel_center_offset)] for x,y,z in ac]).cpu().numpy()
        pc_sampling_origin = torch.tensor([[int(x - voxel_center_offset), int(y - voxel_center_offset), int(z - voxel_center_offset)] for x,y,z in pc]).cpu().numpy()

        ac_pred, pc_pred = infer_landmark_preds(outputs, ac_sampling_origin, pc_sampling_origin, device)
        
        predicted_ac_coords = predicted_ac_coords + [ac_pred]
        predicted_pc_coords = predicted_pc_coords + [pc_pred]
        test_ac_true_coords = test_ac_true_coords + [ac_true]
        test_pc_true_coords = test_pc_true_coords + [pc_true]
        
        mre_, ac_mre_, pc_mre_, ac_x_err, ac_y_err, ac_z_err, pc_x_err, pc_y_err, pc_z_err = mre(ac_true, pc_true, ac_pred, pc_pred)
        
        test_mre = test_mre + [mre_]
        test_mre_ac = test_mre_ac + [ac_mre_]
        test_mre_pc = test_mre_pc + [pc_mre_]
        test_ac_x_err = test_ac_x_err + [ac_x_err]
        test_ac_y_err = test_ac_y_err + [ac_y_err]
        test_ac_z_err = test_ac_z_err + [ac_z_err]
        test_pc_x_err = test_pc_x_err + [pc_x_err]
        test_pc_y_err = test_pc_y_err + [pc_y_err]
        test_pc_z_err = test_pc_z_err + [pc_z_err]
        
    track_test_set_metrics_path = os.path.join(results_folder, 'tracked_test_set_metrics.csv')
    pd.DataFrame({'outer_fold':[outer_fold], 'test_mre': [test_mre], 'test_mre_ac': [test_mre_ac], 
                  'test_mre_pc':[test_mre_pc], 'test_ac_x_err':[test_ac_x_err], 'test_ac_y_err': [test_ac_y_err], 
                  'test_ac_z_err':[test_ac_z_err], 'test_pc_x_err':[test_pc_x_err], 'test_pc_y_err':[test_pc_y_err], 
                  'test_pc_z_err':[test_pc_z_err], 'scan_path':[scan_path_list]
                 }).to_csv(track_test_set_metrics_path, mode='a', header=not os.path.exists(track_test_set_metrics_path))
    
    track_test_set_predictions_path = os.path.join(results_folder, 'tracked_test_set_predictions.csv')
    pd.DataFrame({'outer_fold':[outer_fold], 'predicted_ac_coords':[predicted_ac_coords], 'predicted_pc_coords': [predicted_pc_coords],
                  'true_ac_coords':[test_ac_true_coords], 'true_pc_coords':[test_pc_true_coords],
                  'scan_path':[scan_path_list]
                 }).to_csv(track_test_set_predictions_path, mode='a', header=not os.path.exists(track_test_set_predictions_path))


    
def main(data_dir, num_epochs, patch_size, min_thresh, batch_size, hp_trial_results_path, augmentation_scales, num_rotations):
    #validate data preprocessing and required data paths
    scan_info_df, pre_calc_scaling_factors_df, img_dir, optim_hp = verify_preprocessed_data_aug(
        data_dir, 
        scan_info_name = "scan_info.csv", 
        pre_calc_scaling_factors_name = "scaling_factors_info.csv", 
        hp_trial_results_path = hp_trial_results_path
    )
    print(f'Optimal HP config chosen: {optim_hp}')

    #unpacking the optimal hps
  
    config = {"lr":optim_hp["lr"], 
             "sigma":optim_hp["sigma"], 
             "gamma":optim_hp["gamma"], 
             "alpha":optim_hp["alpha"], 
             "patch_size":patch_size, 
             "min_thresh":min_thresh,
             "num_epochs":num_epochs, 
             "batch_size":batch_size}
    #directory where this script is located
    base_dir = Path()
    
    # print(torch.cuda.is_available())    
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    #Filter precalculated scaling factors required to assemble full heatmaps, on the im_type (axial, coronal, sagittal) that is selected     
    pre_calc_scaling_factors_df = pre_calc_scaling_factors_df.reset_index(drop = True)

    #Filter img coordinates for reference-standard (ground-truth) and coarse-localized AC and PC landmarks, on the selected im_type
    scan_info_df = scan_info_df.reset_index(drop = True)

    #attaching paths for scans and heatmaps 
    scan_info_df['data_path'] = img_dir
    scan_info_df['scan_path'] = scan_info_df['scan_path'].apply(lambda x: img_dir / 'ip_patches' /str(x).lstrip(os.sep))
    scan_info_df['gt_heatmap_path'] = scan_info_df['gt_heatmap_path'].apply(lambda x: img_dir / 'op_patches' /str(x).lstrip(os.sep))
    pre_calc_scaling_factors_df['gt_heatmap_path'] = pre_calc_scaling_factors_df['gt_heatmap_path'].apply(lambda x: img_dir / 'op_patches' /str(x).lstrip(os.sep))

    scan_info_df.rename(columns = {'ac_img':'ac_img_coarse', 'pc_img':'pc_img_coarse'}, inplace = True)
    num_scans = len(scan_info_df['scan_id'].unique())

    rot_ids = [x+1 for x in range(num_rotations)] 
    ###############################################################################################################################
    for aug_scale in augmentation_scales:
        
        experiment_name = f'AugmentedTimes{aug_scale}'

        results_dir = (base_dir / 'unet_results_aug' / experiment_name).resolve()
    
        #creating a results directory
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
            # raise FileExistsError(f"The folder '{results_dir}' already exists. Please rename it or delete it to avoid losing data.")
        # else:          
        os.makedirs(results_dir)
        print(f"Results will be saved to: {results_dir}")

        random.seed(35)
        rot_sample_df = pd.DataFrame(pd.DataFrame([[[0] + random.sample(rot_ids, aug_scale)] for _ in range(num_scans)])[0].to_list())
        rot_sample_df.columns = ['rot_' + str(x) for x in range(aug_scale+1)]
        rot_sample_df['scan_id'] = scan_info_df['scan_id'].drop_duplicates().values
        rot_sample_df['id'] = rot_sample_df.index
        rot_sample_df.to_csv(str((results_dir / f'augmentations_selected_times_{aug_scale}.csv').resolve()))
        rot_sample_df_long = pd.wide_to_long(rot_sample_df, "rot_", i="id", j="rot_id").reset_index()
        rot_sample_df_long.drop(columns = ['id','rot_id'], inplace = True)
        rot_sample_df_long.columns = ['scan_id', 'rot_id']

        scan_info_long_df = rot_sample_df_long.merge(scan_info_df, how = 'inner', on = ['scan_id','rot_id'])

        all_dataset = PreLoadDataset(scan_info_long_df['scan_path'].values, 
                                     scan_info_long_df['gt_heatmap_path'].values,                    
                                     scan_info_long_df[['ac_img_true','ac_img_coarse','pc_img_true','pc_img_coarse']],
                                     patch_size,
                                     pre_load_data_transform)  

        kwargs = {'num_workers': 4, 'pin_memory': True} 
        all_dataloader = DataLoader(all_dataset, batch_size = 64, shuffle = False, 
                                    **kwargs)
        inputs_list = []
        outputs_list = []
        labels_list = []
        scan_id_map = dict()
        
        for i, data in enumerate(all_dataset):    
            
            scan_id_map[i] = str((Path() / data['scan_path']).resolve())
            inputs_list.append(data['image'])
            outputs_list.append(data['target'])
            labels_list.append(data['label'])
         
        input_tensors = torch.stack(inputs_list)
        output_tensors = torch.stack(outputs_list)
        label_tensors = torch.stack(labels_list)
    
        scan_id_map = dict(zip(scan_id_map.values(), scan_id_map.keys()))  
        pd.DataFrame.from_dict(scan_id_map, orient = 'index').to_csv(results_dir / 'scan_id_map.csv')
        print(f"Wrote out scan-path and long tensor-id map")

        pre_calc_scaling_factors = pre_calc_scaling_factors_df.merge(scan_info_long_df[['scan_id','scan_path','rot_id']], how = 'inner', on = ['scan_id','rot_id']).reset_index(drop = True)
        pre_calc_scaling_factors['scan_path'] = pre_calc_scaling_factors['scan_path'].apply(lambda x: str((base_dir / x).resolve()))
        
        """If you have multiple scans per patient, do a patient-level cross-validation by using a stratified group k-fold split. If working with multiple neurological conditions, stratified splits ensure that the training and test sets contain similar distributions of each condition. If there is only 1    condition, stratified group k-fold defaults to just group-k-fold"""
        #outer_skf = StratifiedGroupKFold(n_splits=5, shuffle = True, random_state = 75)
        outer_groups = scan_info_long_df['scan_id'] 
    
        #going with group k-fold CV splits to keep train and test splits mutually exclusive in terms of scans, for the purpose of this demonstration
        outer_skf = StratifiedGroupKFold(n_splits=5, shuffle = True, random_state = 75)
        scan_info_long_df['dummy'] = 1 #indicating that there is only one class of scans in this dataset
        outer_fold_test_metrics = pd.DataFrame()
        outer_fold = 0
        
        for split in outer_skf.split(scan_info_long_df, scan_info_long_df['dummy'], groups = scan_info_long_df['scan_id']):
            
            train_index, test_index = split[0], split[1]
                     
            train_df, test_df = scan_info_long_df.iloc[train_index].reset_index(drop = True), scan_info_long_df.iloc[test_index].reset_index(drop = True)    

            #we only want to test on original unaugmented data
            test_df = test_df[test_df['rot_id'] == 0].reset_index(drop = True)
        
            print(f'Fold {outer_fold}: Train set size is {len(train_df)} and test set size is {len(test_df)}')      

            train_model(config, data_dict = {'input_tensors':input_tensors, 
                                             'output_tensors':output_tensors, 
                                             'label_tensors':label_tensors, 
                                             'scan_id_map':scan_id_map,
                                             'data_df_train': train_df,                                             
                                             'data_df_test': test_df,
                                             'outer_fold': outer_fold,
                                             'pre_calc_scaling_factors':pre_calc_scaling_factors,
                                             'results_folder':results_dir, 
                                             'img_dir':img_dir})

            outer_fold = outer_fold + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default= Path(), help='Root data directory')    
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--min_thresh', type=float, default=0.01)  
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hp_trial_results_path', type=str, default=str((Path() / 'unet_results_hp_optim/trial_results.csv').resolve()))
    parser.add_argument('--augmentation_scales', type=list, default=[x for x in range(2,21,2)])
    parser.add_argument('--num_rotations', type = int, default=48)
    
    args = parser.parse_args()

  
    main(
        data_dir = args.data_dir,
        num_epochs = args.num_epochs,
        patch_size = args.patch_size, 
        min_thresh = args.min_thresh,
        batch_size = args.batch_size,
        hp_trial_results_path = args.hp_trial_results_path, 
        augmentation_scales = args.augmentation_scales, 
        num_rotations = args.num_rotations
    )
    

