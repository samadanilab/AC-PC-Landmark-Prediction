#library imports
from data_utils import verify_preprocessed_data, PreLoadDataset,  DataTransform, ACPCLandMarkDataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import os, re, torch, time
import argparse
from pathlib import Path
import sys

from loss import BCEWithFocalLoss
from model import UNet3D
import torch.optim as optim
from torch.optim import lr_scheduler

import random

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
              img_dir, rot_id = 0): 

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
    
    selected_scans = [str((img_dir / f"ip_patches/{scan_id}/input_patches_Rot_{rot_id}.npy").resolve()) for scan_id in selected_scans_cv]
    
    selected_indices = [scan_id_map[x] for x in selected_scans]
    dataset = ACPCLandMarkDataset(input_tensors[selected_indices],
                                  output_tensors[selected_indices], 
                                  label_tensors[selected_indices], 
                                  selected_scans,     
                                  patch_size, 
                                  data_transform)

    return dataset

def train_model(config, data_dict = None):
    #unpack data_dict
    input_tensors = data_dict["input_tensors"] 
    output_tensors = data_dict["output_tensors"]                                                          
    label_tensors = data_dict["label_tensors"] 
    scan_id_map = data_dict["scan_id_map"]
                                                                 
    data_df = data_dict["data_df"]
    outer_fold = data_dict["outer_fold"]
    pre_calc_scaling_factors = data_dict["pre_calc_scaling_factors"]
 
    results_folder = data_dict["results_folder"]
    img_dir = data_dict["img_dir"]

    #fixed parameters
    patch_size = config["patch_size"]
    min_thresh = config["min_thresh"]
    num_epochs = config["num_epochs"] 
    num_batches = config["batch_size"]
    voxel_center_offset = config["patch_size"]//2
    
    #Hyperparameters to tune  
    sigma = config["sigma"]    
    alpha = config["alpha"]
    gamma = config["gamma"]
    initial_lr = config["lr"]

    #training device set-up (these settings allow for easier debugging. Feel free to modify according to your needs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {'num_workers': 0, 'pin_memory': False} if device=='cuda' else {}

    #Defining the loss function conditioned on the chosen class-imbalance (alpha), focus (gamma), and minimum threshold (min_thresh) parameters 
    #for active voxel selection
    criterion = BCEWithFocalLoss(alpha, gamma, min_thresh, "mean")
    
    #As with the outer cross validation splits, stratify if dealing with multiple neurological conditions. Will default to groupkfold if there is only one         condition. 
    # inner_groups = data_df['pat'] 
    # inner_gkf = StratifiedGroupKFold(n_splits=4, shuffle = True, random_state = 75)

    #going with k-fold splits for the sake of this demonstration
    inner_gkf = KFold(n_splits=4, shuffle = True, random_state = 75)

    #Setting up the inner cross validation for hyper-parameter tuning
    inner_fold_val_metrics = pd.DataFrame()
    inner_fold_val_metrics_file_path = os.path.join(results_folder, 'inner_fold_val_metrics.csv')
    
    fold = 0

    #start inner cross validation for hyperparameter tuning
    for train, valid in inner_gkf.split(data_df): 
        
        X_train, X_valid = data_df.iloc[train], data_df.iloc[valid]

        #create a mapping between the accession numbers and which patient the scan originated from
        # selected_accnum_pat_dict_train = dict(zip(X_train['acc_num'].values, X_train['pat'].values))
        # selected_accnum_pat_dict_valid = dict(zip(X_valid['acc_num'].values, X_valid['pat'].values))

        #create train and validation datasets
        selected_scans_cv_train = X_train['scan_id'].values
        selected_scans_cv_valid = X_valid['scan_id'].values


        train_dataset = load_data(input_tensors, output_tensors, label_tensors, scan_id_map, sigma, patch_size, pre_calc_scaling_factors, selected_scans_cv_train, img_dir)
        valid_dataset = load_data(input_tensors, output_tensors, label_tensors, scan_id_map, sigma, patch_size, pre_calc_scaling_factors, selected_scans_cv_valid, img_dir)

        #create train and validation dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size = num_batches, shuffle = True, **kwargs)
        val_dataloader = DataLoader(valid_dataset, batch_size = num_batches, shuffle = False, **kwargs)
        train_len, val_len = len(train_dataloader), len(val_dataloader)

        #initialize lists to track train and validation loss and localization error metrics (both radial and coordinate-wise) across the training epochs
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

        validation_loss_epochs = []
        validation_mre_epochs = []
        validation_ac_mre_epochs = []
        validation_pc_mre_epochs = []
        validation_ac_x_err_epochs = []
        validation_ac_y_err_epochs = []
        validation_ac_z_err_epochs = []
        validation_pc_x_err_epochs = []
        validation_pc_y_err_epochs = []
        validation_pc_z_err_epochs = []
        
        #initialize model with seed for reproducibility
        torch.manual_seed(46)
        unet_model = UNet3D(in_channels = 2, out_channels = 6)
        unet_model = unet_model.to(device)

        #initialize the SGD optimizer and the learning rate scheduler
        optimizer = optim.SGD(unet_model.parameters(), lr=initial_lr, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        #Iterate through epochs
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
            train_nf = 0 #Normalization factor (keeps track of how many training samples were seen in an epoch)

            #Track the mean hounsfield unit (HU) across all of the training data in this fold. Need to do this only once, and record it to apply
            #in the validation phase. 
            if epoch == 0:
                train_mean_hus_ac = []
                train_mean_hus_pc = []

            # Set model to training mode
            unet_model.train()  
            
            # Training phase per epoch
            torch.manual_seed(23 + epoch) #change how the order in which data is presented to the model in each epoch, but in a reproducible way
            for data in train_dataloader:
                #unstack the input, target (ground-truth), labels (AC-PC reference standard and coarse localized image landmarks), and image type (axial,                      coronal, sagittal) batches and move them to GPU 
                inputs_unnorm, targets, labels = data['image'].to(device), data['target'].to(device), data['label'].to(device)

                #Normalize the input AC and PC patches by their mean value. Returns the means of the AC and PC patches in this batch
                inputs, mean_hus = center_patch_hu(inputs_unnorm.float())

                #Record the mean AC and PC HU values for each batch, only in the first epoch. 
                if epoch == 0:                    
                    train_mean_hus_ac = train_mean_hus_ac + [mean_hus[0].item()]
                    train_mean_hus_pc = train_mean_hus_pc + [mean_hus[1].item()]
                
                inputs, targets = inputs.float(), targets.float() 

                #Unpack the reference-standard and coarse-localized AC and PC image coordinates from the labels tensor. Note that the reference standard
                #coordinates are required to compute the radial error statistics
                ac_true, ac = labels[:,:3], labels[:,3:6] 
                pc_true, pc = labels[:,6:9], labels[:,9:]

                # zero the parameter gradients
                optimizer.zero_grad()

                #make the forward pass to obtain predicted heatmaps for this batch of data
                outputs = unet_model(inputs) 
                #calculate the loss
                loss = criterion(outputs, targets)

                #backpropagate the loss to compute gradients and weight updates
                loss.backward()

                #make the weight updates
                optimizer.step()

                #determine the origin (image coordinates) of the AC and PC input patches in each sample (batch) 
                ac_sampling_origin = torch.tensor([[int(x - voxel_center_offset), 
                                                    int(y - voxel_center_offset), 
                                                    int(z - voxel_center_offset)] for x,y,z in ac]).cpu().numpy()
                pc_sampling_origin = torch.tensor([[int(x - voxel_center_offset), 
                                                    int(y - voxel_center_offset), 
                                                    int(z - voxel_center_offset)] for x,y,z in pc]).cpu().numpy()

                #infer AC and PC image landmark predictions (x,y,z coordinates)
                ac_pred, pc_pred = infer_landmark_preds(outputs, ac_sampling_origin, pc_sampling_origin, device)

                #calculate localization error
                mre_, ac_mre_, pc_mre_, ac_x_err, ac_y_err, ac_z_err, pc_x_err, pc_y_err, pc_z_err = mre(ac_true, pc_true, ac_pred, pc_pred)

                #compute the total loss and AC/PC MRE along with coordinate-specific errors for each batch
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

                train_nf = train_nf + inputs.size(0) #running total of the number of samples across the batches for each epoch

            #invoke the learning rate scheduler so it keeps track of the number of epochs.  
            exp_lr_scheduler.step() 

            #This computes the loss and radial error metrics per epoch
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

            #keep track of loss and MRE metrics over the epochs
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


            # Validation metrics initialization
            val_running_loss = 0.0
            val_running_mre = 0.0
            val_running_ac_mre = 0.0
            val_running_pc_mre = 0.0
            val_running_ac_x_err = 0.0
            val_running_ac_y_err = 0.0
            val_running_ac_z_err = 0.0
            val_running_pc_x_err = 0.0
            val_running_pc_y_err = 0.0
            val_running_pc_z_err = 0.0
            val_nf = 0
            best_mre = np.inf
            best_epoch = 0

            # get the mean vaues of the AC and PC patches in the training loop
            ac_mean_norm = np.mean(train_mean_hus_ac)
            pc_mean_norm = np.mean(train_mean_hus_pc)

            #set the model to evaluation mode
            unet_model.eval()
            for data in val_dataloader:
                #unpack the input, target, labels, and image types from each batch of data returned by the data loader
                inputs_unnorm, targets, labels = data['image'].to(device), data['target'].to(device), data['label'].to(device)

                #normalize the AC and PC patches by the training AC and PC means 
                inputs = torch.stack([inputs_unnorm[:,0,:,:,:] - pc_mean_norm, inputs_unnorm[:,1,:,:,:] - ac_mean_norm], axis = 1)
                
                inputs, targets = inputs.float(), targets.float()

                #make inference on the validation batch
                with torch.no_grad():            
                    outputs = unet_model(inputs)

                #Unpack the reference-standard and coarse-localized AC and PC image coordinates from the labels tensor. Note that the reference standard
                #coordinates are required to compute the radial error statistics
                ac_true, ac = labels[:,:3], labels[:,3:6] 
                pc_true, pc = labels[:,6:9], labels[:,9:]

                #determine the origin (image coordinates) of the AC and PC input patches in each sample (batch) 
                ac_sampling_origin = torch.tensor([[int(x - voxel_center_offset), 
                                                    int(y - voxel_center_offset), 
                                                    int(z - voxel_center_offset)] for x,y,z in ac]).cpu().numpy()
                pc_sampling_origin = torch.tensor([[int(x - voxel_center_offset), 
                                                    int(y - voxel_center_offset), 
                                                    int(z - voxel_center_offset)] for x,y,z in pc]).cpu().numpy()

                #infer AC and PC image landmark predictions (x,y,z coordinates)
                ac_pred, pc_pred = infer_landmark_preds(outputs, ac_sampling_origin, pc_sampling_origin, device)

                #calculate localization error
                mre_, ac_mre_, pc_mre_, ac_x_err, ac_y_err, ac_z_err, pc_x_err, pc_y_err, pc_z_err = mre(ac_true, pc_true, ac_pred, pc_pred)

                #compute the total loss and AC/PC MRE along with coordinate-specific errors for each batch                
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_mre += mre_ * inputs.size(0)
                val_running_ac_mre += ac_mre_ * inputs.size(0)
                val_running_pc_mre += pc_mre_ * inputs.size(0)
                val_running_ac_x_err += ac_x_err * inputs.size(0)
                val_running_ac_y_err += ac_y_err * inputs.size(0)
                val_running_ac_z_err += ac_z_err * inputs.size(0)
                val_running_pc_x_err += pc_x_err * inputs.size(0)
                val_running_pc_y_err += pc_y_err * inputs.size(0)
                val_running_pc_z_err += pc_z_err * inputs.size(0)
                val_nf += inputs.size(0)

            #This computes the loss and radial error metrics per epoch
            val_epoch_loss = val_running_loss/val_nf
            val_epoch_mre = (val_running_mre/val_nf).numpy().item()
            val_epoch_ac_mre = (val_running_ac_mre/val_nf).numpy().item()
            val_epoch_ac_x_err = (val_running_ac_x_err/val_nf).numpy().item()
            val_epoch_ac_y_err = (val_running_ac_y_err/val_nf).numpy().item()
            val_epoch_ac_z_err = (val_running_ac_z_err/val_nf).numpy().item()
            val_epoch_pc_mre = (val_running_pc_mre/val_nf).numpy().item()
            val_epoch_pc_x_err = (val_running_pc_x_err/val_nf).numpy().item()
            val_epoch_pc_y_err = (val_running_pc_y_err/val_nf).numpy().item()
            val_epoch_pc_z_err = (val_running_pc_z_err/val_nf).numpy().item()
            
            
            #Record best MRE and epoch where it was accomplished
            if val_epoch_mre < best_mre:
                best_mre = val_epoch_mre
                best_epoch = epoch

            #Track the loss across epochs for the validation dataset
            validation_loss_epochs.append(val_epoch_loss)
            validation_mre_epochs.append(val_epoch_mre)
            validation_ac_mre_epochs.append(val_epoch_ac_mre)
            validation_pc_mre_epochs.append(val_epoch_pc_mre)
            validation_ac_x_err_epochs.append(val_epoch_ac_x_err)       
            validation_ac_y_err_epochs.append(val_epoch_ac_y_err)
            validation_ac_z_err_epochs.append(val_epoch_ac_z_err)
            validation_pc_x_err_epochs.append(val_epoch_pc_x_err)
            validation_pc_y_err_epochs.append(val_epoch_pc_y_err)        
            validation_pc_z_err_epochs.append(val_epoch_pc_z_err)
            
            epoch_time = time.time() - epoch_start_time
            
            #print statements --- 
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
                
            print(f"""Train Loss: {train_epoch_loss:.4f} | MRE: {train_epoch_mre:.4f} | AC (MRE,X,Y,Z): {train_epoch_ac_mre:.4f},{train_epoch_ac_x_err:.4f},{train_epoch_ac_y_err:.4f},{train_epoch_ac_z_err:.4f} | PC(MRE,X,Y,Z): {train_epoch_pc_mre:.4f},{train_epoch_pc_x_err:.4f},{train_epoch_pc_y_err:.4f},{train_epoch_pc_z_err:.4f} """)

            print(f"""Validation Loss: {val_epoch_loss:.4f} | MRE: {val_epoch_mre:.4f} | AC (MRE,X,Y,Z): {val_epoch_ac_mre:.4f},{val_epoch_ac_x_err:.4f},{val_epoch_ac_y_err:.4f},{val_epoch_ac_z_err:.4f} | PC(MRE,X,Y,Z): {val_epoch_pc_mre:.4f},{val_epoch_pc_x_err:.4f},{val_epoch_pc_y_err:.4f},{val_epoch_pc_z_err:.4f} """)
            
            print(f'Epoch Run Time: {epoch_time// 60:.4f}')
        

        #After all epochs, for this HP config and split, what was the best validation MRE? This is almost always the last epoch.  
        
        val_metrics_dict = {"config":[config],
            "loss": validation_loss_epochs[best_epoch], 
            "mre" : validation_mre_epochs[best_epoch],
            "ac_mre" : validation_ac_mre_epochs[best_epoch],
            "pc_mre" : validation_pc_mre_epochs[best_epoch],
            "ac_x_err" : validation_ac_x_err_epochs[best_epoch],
            "ac_y_err" : validation_ac_y_err_epochs[best_epoch],
            "ac_z_err" : validation_ac_z_err_epochs[best_epoch],
            "pc_x_err" : validation_pc_x_err_epochs[best_epoch],
            "pc_y_err" : validation_pc_y_err_epochs[best_epoch],
            "pc_z_err" : validation_pc_z_err_epochs[best_epoch], 
                           "best_epoch": [best_epoch]}
        val_metrics = pd.DataFrame(val_metrics_dict, index =range(0,1))
        val_metrics['fold'] = fold        
        val_metrics['outer_fold'] = outer_fold
        
        
        inner_fold_val_metrics = pd.concat([inner_fold_val_metrics, val_metrics], ignore_index = True)       

        
        val_metrics.to_csv(inner_fold_val_metrics_file_path, mode='a', header=not os.path.exists(inner_fold_val_metrics_file_path)) #best metrics across all epochs for the 4 inner folds, 5 outer folds, and hyperparameter configurations evaluated.
        

        exp_metrics = pd.DataFrame({'config':[config], 'fold':[fold], 'outer_fold':[outer_fold],
                                    'train_losses':[train_loss_epochs], 'val_losses':[validation_loss_epochs],
                                   'train_ac_mre':[train_ac_mre_epochs], 'val_ac_mre':[validation_ac_mre_epochs], 
                                    'train_pc_mre':[train_pc_mre_epochs], 'val_pc_mre':[validation_pc_mre_epochs],
                                   'train_ac_x_err':[train_ac_x_err_epochs], 'val_ac_x_err':[validation_ac_x_err_epochs],
                                   'train_ac_y_err':[train_ac_y_err_epochs], 'val_ac_y_err':[validation_ac_y_err_epochs],
                                   'train_ac_z_err':[train_ac_z_err_epochs], 'val_ac_z_err':[validation_ac_z_err_epochs]})
        
        exp_metrics_file_path = os.path.join(results_folder, 'tracked_epoch_level_metrics.csv') #epoch level metrics for each fold
        exp_metrics.to_csv(exp_metrics_file_path, mode='a', header=not os.path.exists(exp_metrics_file_path))

        
        fold = fold + 1
    
    #avg metrics across all folds for a particular HP combination
    avg_metrics_folds = inner_fold_val_metrics.mean().to_dict() #averaging across the 4 validation folds within the train-valid splits
    

    print(f"Outer fold {outer_fold} : Finished training on all inner folds with avg. metrics for this experiment being {avg_metrics_folds}")
    return pd.DataFrame({'config':[config], 'avg_cv_mre':[avg_metrics_folds['mre']], 'avg_cv_loss':[avg_metrics_folds['loss']],
                        'avg_cv_ac_mre':[avg_metrics_folds['ac_mre']], 'avg_cv_pc_mre':[avg_metrics_folds['pc_mre']],
                        'avg_cv_ac_x_err': [avg_metrics_folds['ac_x_err']], 'avg_cv_ac_y_err': [avg_metrics_folds['ac_y_err']],
                        'avg_cv_ac_z_err': [avg_metrics_folds['ac_z_err']], 'avg_cv_pc_x_err': [avg_metrics_folds['pc_x_err']],
                        'avg_cv_pc_y_err': [avg_metrics_folds['pc_y_err']], 'avg_cv_pc_z_err': [avg_metrics_folds['pc_z_err']],
                        'avg_best_epoch':[avg_metrics_folds['best_epoch']]}
                       , index =range(0,1))

def test_accuracy(outer_fold, train_valid_dataloader, test_dataloader, unet_model, criterion, optimizer, exp_lr_scheduler, num_epochs, voxel_center_offset, results_folder, device):


    
    print("Retraining model based on best config for loss function on training set to eval on test set")
    
    train_losses = []
    train_mre = []
    train_mre_ac = []
    train_mre_pc = []
    train_ac_x_err = []
    train_ac_y_err = []
    train_ac_z_err = []
    train_pc_x_err = []
    train_pc_y_err = []
    train_pc_z_err = []
    
    track_train_set_predictions_path_outer = os.path.join(results_folder, 'tracked_train_set_predictions_outer.csv')
    track_train_loss_path_outer =  os.path.join(results_folder, 'tracked_train_loss_outer.csv')
    best_mre = np.inf
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
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

        batch_num = 0
        unet_model.train()  # Set model to training mode
        torch.manual_seed(23 + epoch)
        for data in train_valid_dataloader:
            
            inputs_unnorm, targets, labels = data['image'].to(device), data['target'].to(device), data['label'].to(device)
            
            inputs, mean_hus = center_patch_hu(inputs_unnorm.float())
            if epoch == 0:
                train_mean_hus_ac = train_mean_hus_ac + [mean_hus[0].item()]
                train_mean_hus_pc = train_mean_hus_pc + [mean_hus[1].item()]
                
            inputs, targets = inputs.float(), targets.float()
            
            ac_true, ac = labels[:,:3], labels[:,3:6] 
            pc_true, pc = labels[:,6:9], labels[:,9:]
     
            # zero the parameter gradients
            optimizer.zero_grad()
            
            outputs = unet_model(inputs) #2b x 3 x 32 x 32 x 32
        
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
                    
            ac_sampling_origin = torch.tensor([[int(x - voxel_center_offset), int(y - voxel_center_offset), int(z - voxel_center_offset)] for x,y,z in ac]).cpu().numpy()
            pc_sampling_origin = torch.tensor([[int(x - voxel_center_offset), int(y - voxel_center_offset), int(z - voxel_center_offset)] for x,y,z in pc]).cpu().numpy()

            ac_pred, pc_pred = infer_landmark_preds(outputs, ac_sampling_origin, pc_sampling_origin, device)

            if epoch == num_epochs - 1:
                ac_predicted_coordinate_df = pd.DataFrame(ac_pred)
                ac_predicted_coordinate_df.columns = ['ac_x','ac_y','ac_z']
                #this step parses the scan_id from the full scan path. Make sure that the data folders are organized similarly for correct functionality
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
        
        train_losses = train_losses + [train_epoch_loss]
        train_mre = train_mre + [train_epoch_mre]
        train_mre_ac = train_mre_ac + [train_epoch_ac_mre]
        train_mre_pc = train_mre_pc + [train_epoch_pc_mre]
        train_ac_x_err = train_ac_x_err + [train_epoch_ac_x_err]        
        train_ac_y_err = train_ac_y_err + [train_epoch_ac_y_err]
        train_ac_z_err = train_ac_z_err + [train_epoch_ac_z_err]
        train_pc_x_err = train_pc_x_err + [train_epoch_pc_x_err]
        train_pc_y_err = train_pc_y_err + [train_epoch_pc_y_err]        
        train_pc_z_err = train_pc_z_err + [train_epoch_pc_z_err]
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch Run Time: {epoch_time// 60:.3f}') 
        
        #print statements --- 
        print(f"""Train Loss: {train_epoch_loss:.4f} | MRE: {train_epoch_mre:.4f} | AC (MRE,X,Y,Z): {train_epoch_ac_mre:.4f},{train_epoch_ac_x_err:.4f},{train_epoch_ac_y_err:.4f},{train_epoch_ac_z_err:.4f} | PC(MRE,X,Y,Z): {train_epoch_pc_mre:.4f},{train_epoch_pc_x_err:.4f},{train_epoch_pc_y_err:.4f},{train_epoch_pc_z_err:.4f} """)

        train_loss_metrics_outer_df = pd.DataFrame({"epoch":[epoch],
                                                    "loss": [train_epoch_loss],
                                                    "mre": [train_epoch_mre],
                                                    "ac_mre": [train_epoch_ac_mre],
                                                    "ac_x_err": [train_epoch_ac_x_err],
                                                    "ac_y_err": [train_epoch_ac_y_err],
                                                    "ac_z_err": [train_epoch_ac_z_err],
                                                    "pc_mre": [train_epoch_pc_mre],
                                                    "pc_x_err": [train_epoch_pc_x_err],
                                                    "pc_y_err": [train_epoch_pc_y_err],
                                                    "pc_z_err": [train_epoch_pc_z_err]})
        
        train_loss_metrics_outer_df.to_csv(track_train_loss_path_outer, mode='a', 
                                           header=not os.path.exists(track_train_loss_path_outer))

        if train_epoch_mre < best_mre:
            best_mre = train_epoch_mre
            best_epoch = epoch
            torch.save({"model_state": unet_model.state_dict()}, 
                          os.path.join(results_folder, f'best_model_outer_fold {outer_fold}.pt'))

    
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
    
    test_running_mre = 0.0
    test_running_ac_mre = 0.0
    test_running_pc_mre = 0.0
    test_running_ac_x_err = 0.0
    test_running_ac_y_err = 0.0
    test_running_ac_z_err = 0.0
    test_running_pc_x_err = 0.0
    test_running_pc_y_err = 0.0
    test_running_pc_z_err = 0.0
    test_nf = 0
    
    ac_mean_norm = np.mean(train_mean_hus_ac)
    pc_mean_norm = np.mean(train_mean_hus_pc)
    
    unet_model.eval()
    torch.manual_seed(23 + epoch)
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
      

        test_running_mre += mre_ 
        test_running_ac_mre += ac_mre_
        test_running_pc_mre += pc_mre_ 
        test_running_ac_x_err += ac_x_err 
        test_running_ac_y_err += ac_y_err
        test_running_ac_z_err += ac_z_err 
        test_running_pc_x_err += pc_x_err
        test_running_pc_y_err += pc_y_err
        test_running_pc_z_err += pc_z_err
        test_nf += inputs.size(0)
        
        test_mre = test_mre + [mre_]
        test_mre_ac = test_mre_ac + [ac_mre_]
        test_mre_pc = test_mre_pc + [pc_mre_]
        test_ac_x_err = test_ac_x_err + [ac_x_err]
        test_ac_y_err = test_ac_y_err + [ac_y_err]
        test_ac_z_err = test_ac_z_err + [ac_z_err]
        test_pc_x_err = test_pc_x_err + [pc_x_err]
        test_pc_y_err = test_pc_y_err + [pc_y_err]
        test_pc_z_err = test_pc_z_err + [pc_z_err]
        
    
    track_test_set_predictions_path = os.path.join(results_folder, 'tracked_test_set_predictions.csv')
    pd.DataFrame({'outer_fold':[outer_fold], 'predicted_ac_coords':[predicted_ac_coords], 'predicted_pc_coords': [predicted_pc_coords],
                  'true_ac_coords':[test_ac_true_coords], 'true_pc_coords':[test_pc_true_coords],
                  'scan_path':[scan_path_list]
                 }).to_csv(track_test_set_predictions_path, mode='a', header=not os.path.exists(track_test_set_predictions_path))

    
    track_test_set_metrics_path = os.path.join(results_folder, 'tracked_test_set_metrics.csv')
    pd.DataFrame({'outer_fold':[outer_fold], 'test_mre': [test_mre], 'test_mre_ac': [test_mre_ac], 
                  'test_mre_pc':[test_mre_pc], 'test_ac_x_err':[test_ac_x_err], 'test_ac_y_err': [test_ac_y_err], 
                  'test_ac_z_err':[test_ac_z_err], 'test_pc_x_err':[test_pc_x_err], 'test_pc_y_err':[test_pc_y_err], 
                  'test_pc_z_err':[test_pc_z_err], 'scan_path':[scan_path_list]
                 }).to_csv(track_test_set_metrics_path, mode='a', header=not os.path.exists(track_test_set_metrics_path))
    

    test_metrics = {"mre" : (test_running_mre/test_nf).numpy(),
                    "ac_mre" : (test_running_ac_mre/test_nf).numpy(),
                    "pc_mre" : (test_running_pc_mre/test_nf).numpy(),
                    "ac_x_err" : (test_running_ac_x_err/test_nf).numpy(),
                    "ac_y_err" : (test_running_ac_y_err/test_nf).numpy(),
                    "ac_z_err" : (test_running_ac_z_err/test_nf).numpy(),
                    "pc_x_err" : (test_running_pc_x_err/test_nf).numpy(),
                    "pc_y_err" : (test_running_pc_y_err/test_nf).numpy(),
                    "pc_z_err" : (test_running_pc_z_err/test_nf).numpy()}   

    return test_metrics

def lognuniform(low, high, size, base=np.e):
    return np.power(base, np.random.uniform(np.log(low), np.log(high), size))

def random_sample_hps(sigma_choices, gamma_choices, alpha_choices, lr_bounds, num_samples):
    
    sigma_p = torch.tensor([1/len(sigma_choices)]*len(sigma_choices))
    alpha_p = torch.tensor([1/len(alpha_choices)]*len(alpha_choices))
    gamma_p = torch.tensor([1/len(gamma_choices)]*len(gamma_choices))

    hp_samples_df = pd.DataFrame({'sigma':[sigma_choices[ind] for ind in torch.multinomial(sigma_p, num_samples, replacement = True)],
                                  'alpha':[alpha_choices[ind] for ind in torch.multinomial(alpha_p, num_samples, replacement = True)], 
                                  'gamma':[gamma_choices[ind] for ind in torch.multinomial(gamma_p, num_samples, replacement = True)],
                                  'lr':lognuniform(lr_bounds[0], lr_bounds[1], num_samples)})
    return hp_samples_df

def main(data_dir, num_samples, num_epochs, patch_size, min_thresh, hp_search_space, batch_size):

    #validate data preprocessing and required data paths
    scan_info_df, pre_calc_scaling_factors_df, img_dir = verify_preprocessed_data(
        data_dir, 
        scan_info_name = "scan_info.csv", 
        pre_calc_scaling_factors_name = "scaling_factors_info.csv"
    )
    
    #directory where this script is located
    base_dir = Path()
    results_dir = (base_dir / str('unet_results_hp_optim')).resolve()
    
    #creating a results directory
    if os.path.exists(results_dir):
        raise FileExistsError(f"The folder '{results_dir}' already exists. Please rename it or delete it to avoid losing data.")
    else:          
        os.mkdir(results_dir)
    print(f"Results will be saved to: {results_dir}")
        
    #Filter precalculated scaling factors required to assemble full heatmaps, on the im_type (axial, coronal, sagittal) that is selected     
    pre_calc_scaling_factors_df = pre_calc_scaling_factors_df.reset_index(drop = True)

    #Filter img coordinates for reference-standard (ground-truth) and coarse-localized AC and PC landmarks, on the selected im_type
    scan_info_df = scan_info_df.reset_index(drop = True)
    
    #unpacking the hp search space
    lr_bounds = hp_search_space["lr_bounds"]
    sigma_choices = hp_search_space["sigma_choices"]
    gamma_choices = hp_search_space["gamma_choices"]
    alpha_choices = hp_search_space["alpha_choices"]
    
    #sample the required number of hyper-parameter combinations to evaluate
    hp_samples_df = random_sample_hps(sigma_choices, gamma_choices, alpha_choices, lr_bounds, num_samples)
    hp_samples_df.to_csv((results_dir / 'HyperParametersSelected.csv').resolve())
    hp_trial_results_file_path = os.path.join(results_dir, 'hp_trial_results.csv')
    print(f'Hyperparameter choices to evaluate:{hp_samples_df}')
    
    #attaching paths for scans and heatmaps 
    scan_info_df['data_path'] = img_dir
    scan_info_df['scan_path'] = scan_info_df['scan_path'].apply(lambda x: img_dir / 'ip_patches' /str(x).lstrip(os.sep))
    scan_info_df['gt_heatmap_path'] = scan_info_df['gt_heatmap_path'].apply(lambda x: img_dir / 'op_patches' /str(x).lstrip(os.sep))
    pre_calc_scaling_factors_df['gt_heatmap_path'] = pre_calc_scaling_factors_df['gt_heatmap_path'].apply(lambda x: img_dir / 'op_patches' /str(x).lstrip(os.sep))
    # scan_info_df['full_scan_path'] = scan_info_df['scan_path'].apply(lambda x: os.path.join(img_dir, str(x).lstrip(os.sep)))
    # scan_info_df['full_gt_heatmap_path'] = scan_info_df['gt_heatmap_path'].apply(lambda x: os.path.join(img_dir, str(x).lstrip(os.sep)))

    #No augmentation during hyper-parameter tuning 
    scan_info_df_0 = scan_info_df[scan_info_df['rot_id'] == 0].reset_index(drop = True) 
    scan_info_df_0.rename(columns = {'ac_img':'ac_img_coarse', 'pc_img':'pc_img_coarse'}, inplace = True)
    num_scans = len(scan_info_df_0['scan_id'].unique())
    print(f"Number of scans in dataset: {num_scans}")
    
    ##This contains the pre-calculated scaling factors for the heatmap patches with no rotations
    pre_calc_scaling_factors_0 = pre_calc_scaling_factors_df.merge(scan_info_df_0[['scan_id','scan_path','rot_id']], 
                                                                   how = 'inner', on = ['scan_id','rot_id']).reset_index(drop = True)
    pre_calc_scaling_factors_0['scan_path'] = pre_calc_scaling_factors_0['scan_path'].apply(lambda x: str((Path() / x).resolve()))
  

    assert len(pre_calc_scaling_factors_0) == num_scans*len(sigma_choices), "Number of precalculated scaling factors per scan do not correspond to number of sigmas you want to evaluate. Double-check preprocessing"

    #Create long tensors for preprocessed input, heatmap, and labels, along with a map between the scan path and its index in the long tensor ("scan_id_map") 
    #Load these long ternsors into memory for faster batching and moving to GPU during training. 
    all_dataset = PreLoadDataset(scan_info_df_0['scan_path'].values, 
                                 scan_info_df_0['gt_heatmap_path'].values,                    
                                 scan_info_df_0[['ac_img_true','ac_img_coarse','pc_img_true','pc_img_coarse']], 
                                 patch_size, 
                                 pre_load_data_transform)      
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


    """If you have multiple scans per patient, do a patient-level cross-validation by using a stratified group k-fold split. If working with multiple neurological conditions, stratified splits ensure that the training and test sets contain similar distributions of each condition. If there is only 1    condition, stratified group k-fold defaults to just group-k-fold"""
    
    #outer_skf = StratifiedGroupKFold(n_splits=5, shuffle = True, random_state = 75)
    #outer_groups = scan_info_df_0['pat'] 

    #going with k-fold CV splits for the purpose of this demonstration
    outer_skf = KFold(n_splits=5, shuffle = True, random_state = 75)

    outer_fold_test_metrics = pd.DataFrame()
    outer_fold = 0
    
    for split in outer_skf.split(scan_info_df_0):
    
        train_index, test_index = split[0], split[1]     
                          
        train_valid_df, test_df = scan_info_df_0.iloc[train_index].reset_index(drop = True), scan_info_df_0.iloc[test_index].reset_index(drop = True)    

        print(f'Num scans in train validation set is {len(train_valid_df)} and those in test set is {len(test_df)}')

        trial_results_df = pd.DataFrame()
        for i in range(len(hp_samples_df)):
            config = {"lr":  hp_samples_df['lr'].iloc[i],
                      "alpha":  hp_samples_df['alpha'].iloc[i],
                      "gamma":  hp_samples_df['gamma'].iloc[i],
                      "min_thresh": min_thresh, 
                      "sigma": hp_samples_df['sigma'].iloc[i],
                      "num_epochs": num_epochs,
                      "patch_size": patch_size,
                      "batch_size": batch_size
            }
            print(f'Current HP config : {config}')

            cur_trial_results_df = train_model(config, data_dict = {'input_tensors':input_tensors, 
                                                                    'output_tensors':output_tensors, 
                                                                    'label_tensors':label_tensors, 
                                                                    'scan_id_map':scan_id_map,
                                                                    'data_df':train_valid_df,
                                                                    'pre_calc_scaling_factors':pre_calc_scaling_factors_0,
                                                                    'outer_fold':outer_fold,
                                                                    'results_folder':results_dir, 
                                                                    'img_dir':img_dir})
            cur_trial_results_df['outer_fold'] = outer_fold

            
            cur_trial_results_df.to_csv(hp_trial_results_file_path, mode='a', header=not os.path.exists(hp_trial_results_file_path))

            #storing the current trial results in a df so we don't have to read them from the csv again
            trial_results_df = pd.concat([trial_results_df, cur_trial_results_df])
            

        pd.set_option('display.max_columns', None)
        print(f'All trial results for outer fold {outer_fold}: {trial_results_df}')

        #config that gave the best avg. metrics across val folds
        best_config = trial_results_df['config'][trial_results_df['avg_cv_mre'] == trial_results_df['avg_cv_mre'].min()] 
        print(f"Best trial config for outer fold {outer_fold}: {best_config.values[0]}")

        best_patch_size = best_config.values[0]["patch_size"]
        best_sigma = best_config.values[0]["sigma"]
        best_alpha = best_config.values[0]["alpha"]
        best_gamma = best_config.values[0]["gamma"]
        best_min_thresh = best_config.values[0]["min_thresh"]
        best_lr = best_config.values[0]["lr"]        
        voxel_center_offset = best_patch_size//2  

        scan_list_trainvalid_cv = train_valid_df['scan_id'].values
        scan_list_test_cv = test_df['scan_id'].values

        
        train_valid_dataset  = load_data(input_tensors, output_tensors, label_tensors, scan_id_map, best_sigma, best_patch_size, pre_calc_scaling_factors_0, scan_list_trainvalid_cv, img_dir)   
        test_dataset = load_data(input_tensors, output_tensors, label_tensors, scan_id_map, best_sigma, best_patch_size, pre_calc_scaling_factors_0, scan_list_test_cv, img_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {'num_workers': 0, 'pin_memory': False} if device=='cuda' else {}
        train_valid_dataloader = DataLoader(train_valid_dataset, batch_size=16, shuffle = True, **kwargs)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle = False, **kwargs)

        torch.manual_seed(46)
        unet_model = UNet3D(in_channels = 2, out_channels = 6)
        unet_model = unet_model.to(device)
        optimizer = optim.SGD(unet_model.parameters(), lr = best_lr, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        criterion = BCEWithFocalLoss(best_alpha, best_gamma, best_min_thresh, "mean")

        test_set_metrics = test_accuracy(outer_fold, train_valid_dataloader, test_dataloader, unet_model, criterion, optimizer,                                                      exp_lr_scheduler, num_epochs, voxel_center_offset, results_dir, device) 
        print(f"Best trial test set MRE for outer fold {outer_fold}: {test_set_metrics}")
        outer_fold = outer_fold + 1
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default= Path(), help='Root data directory')
    parser.add_argument('--num_samples', type=int, default=25)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--min_thresh', type=float, default=0.01)  
    parser.add_argument('--batch_size', type=int, default=16)
    # Hyper-parameter bounds/ranges to choose from
    parser.add_argument(
        '--lr_bounds', 
        type=float, 
        nargs=2,               
        default=(1e-4, 1e-1),  
        metavar=('MIN', 'MAX'), 
        help='Lower and upper bounds for log-uniform LR sampling')

    parser.add_argument(
        '--sigma_choices', 
        type=int,        
        nargs='+',
        default=[4, 6, 8, 10, 12, 14],  
        help='Range of choices for the std. deviation of Gaussian heatmap used for supervised training')

    parser.add_argument(
        '--gamma_choices', 
        type=int,    
        nargs='+',
        default=[0,1,2,3,4,5],  
        help='Range of choices for the focus parameter used in the loss function')

    parser.add_argument(
        '--alpha_choices', 
        type=float,    
        nargs='+',
        default=[0.1,0.2,0.3,0.4] ,  
        help='Range of choices for the class imbalance parameter used in the loss function')

    args = parser.parse_args()

    hp_search_space = {
        'lr_bounds': args.lr_bounds,
        'sigma_choices': args.sigma_choices,
        'gamma_choices': args.gamma_choices,
        'alpha_choices': args.alpha_choices
    }
  
    main(
        data_dir = args.data_dir,
        num_samples = args.num_samples,
        num_epochs = args.num_epochs,
        patch_size = args.patch_size, 
        min_thresh = args.min_thresh,
        hp_search_space = hp_search_space, 
        batch_size = args.batch_size
    )

