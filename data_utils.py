import sys
import os
import re
import pandas as pd
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

def verify_preprocessed_data(data_root, scan_info_name="scan_info.csv", pre_calc_scaling_factors_name ="scaling_factors_info.csv"):
    """
    Verifies existence and integrity of preprocessed data required to train the 3D-UNet.
    Returns: (scan_info_df, pre_calc_scaling_factors_df, img_dir)
    """
    root = Path(data_root)
    csv_dir = root / "files_for_unet"
    img_dir = root / "patched_data_4unet"
    
    #Check if the csv folder and required scan information and precalculated scaling factors files exist
    if not csv_dir.exists():
        sys.exit(f"ERROR: Splits folder not found at: {csv_dir}")

    scan_info_path = csv_dir / scan_info_name
    pre_calc_scaling_factors_path = csv_dir / pre_calc_scaling_factors_name

    if not scan_info_path.exists():
        sys.exit(f"ERROR: Scan info CSV missing: {scan_info_path}")
    if not pre_calc_scaling_factors_path.exists():
        sys.exit(f"ERROR: Precalculated scaling factors CSV missing: {pre_calc_scaling_factors_path}")
        
    print(f"Found split files: {scan_info_path}, {pre_calc_scaling_factors_path}")

    #Check if the preprocessed image folder exists
    if not img_dir.exists():
        sys.exit(f"ERROR: Processed images folder not found at: {img_dir}")
        
    print(f" Found image directory: {img_dir.name}")

    #Read the scan information CSV 
    try:
        scan_info_df = pd.read_csv(scan_info_path)
    except Exception as e:
        sys.exit(f"ERROR: Could not read {scan_info_name}. Reason: {e}")
    
    if scan_info_df.empty:
        sys.exit(f"ERROR: {scan_info_name} is empty. Please check your preprocessing.")
    
    #Read the precalculated scaling factors CSV
    try: 
        pre_calc_scaling_factors_df = pd.read_csv(pre_calc_scaling_factors_path)
    except Exception as e:
        sys.exit(f"ERROR: Could not read {pre_calc_scaling_factors_name}. Reason: {e}")

    if pre_calc_scaling_factors_df.empty:
        sys.exit(f"ERROR: {pre_calc_scaling_factors_name} is empty. Please check your preprocessing.")


    
    # Read the first row from scan_info.csv
    first_case_scan_path = scan_info_df.iloc[0]
    first_case_scaling_factors_path = pre_calc_scaling_factors_df.iloc[0]
    #If you used the preprocessing script we provide, this file should have a relative path pointing to your augmented scan
    scan_path = img_dir / 'ip_patches' /str(first_case_scan_path['scan_path']).lstrip(os.sep)
    gt_heatmap_path = img_dir / 'op_patches' /str(first_case_scaling_factors_path['gt_heatmap_path']).lstrip(os.sep)

    #Check for the required preprocessed data organization 
    required_structure = [
        scan_path, 
        gt_heatmap_path
    ]
    
    for path_to_check in required_structure:
        if not path_to_check.exists():
            print(f"PREPROCESSED DATA ERROR")
            print(f"The first case in your CSV exists, but is missing required subfolders.")
            print(f"Missing: {path_to_check}")
            print(f"Expected structure: pat > acc_num > [patched_inputs, gt_heatmaps] > [axial, coronal (optional), sagittal (optional)]")
            sys.exit(1)

    print(f"Data structure verified on the first case")
    print("------------------------------------------\n")

    return scan_info_df, pre_calc_scaling_factors_df, img_dir



def verify_preprocessed_data_aug(data_root, hp_trial_results_path, scan_info_name="scan_info.csv", 
                                 pre_calc_scaling_factors_name ="scaling_factors_info.csv", 
                                ):
    """
    Verifies existence and integrity of preprocessed data required to train the 3D-UNet with augmentations, based on hyperparameters selected from the nested CV experiments.
    Returns: (scan_info_df, pre_calc_scaling_factors_df, img_dir, optim_hp)
    """
    root = Path(data_root)
    csv_dir = root / "files_for_unet"
    img_dir = root / "patched_data_4unet"
    
    #Check if the csv folder and required scan information and precalculated scaling factors files exist
    if not csv_dir.exists():
        sys.exit(f"ERROR: Splits folder not found at: {csv_dir}")

    scan_info_path = csv_dir / scan_info_name
    pre_calc_scaling_factors_path = csv_dir / pre_calc_scaling_factors_name

    if not scan_info_path.exists():
        sys.exit(f"ERROR: Scan info CSV missing: {scan_info_path}")
    if not pre_calc_scaling_factors_path.exists():
        sys.exit(f"ERROR: Precalculated scaling factors CSV missing: {pre_calc_scaling_factors_path}")
    
    print(f"Found split files: {scan_info_path}, {pre_calc_scaling_factors_path}")

    #Check if the preprocessed image folder exists
    if not img_dir.exists():
        sys.exit(f"ERROR: Processed images folder not found at: {img_dir}")
        
    print(f" Found image directory: {img_dir.name}")

    #Read the scan information CSV 
    try:
        scan_info_df = pd.read_csv(scan_info_path)
    except Exception as e:
        sys.exit(f"ERROR: Could not read {scan_info_name}. Reason: {e}")
    
    if scan_info_df.empty:
        sys.exit(f"ERROR: {scan_info_name} is empty. Please check your preprocessing.")
    
    #Read the precalculated scaling factors CSV
    try: 
        pre_calc_scaling_factors_df = pd.read_csv(pre_calc_scaling_factors_path)
    except Exception as e:
        sys.exit(f"ERROR: Could not read {pre_calc_scaling_factors_name}. Reason: {e}")

    if pre_calc_scaling_factors_df.empty:
        sys.exit(f"ERROR: {pre_calc_scaling_factors_name} is empty. Please check your preprocessing.")

    
    # Read the first row from scan_info.csv
    first_case_scan_path = scan_info_df.iloc[0]
    first_case_scaling_factors_path = pre_calc_scaling_factors_df.iloc[0]
    #If you used the preprocessing script we provide, this file should have a relative path pointing to your augmented scan
    scan_path = img_dir / 'ip_patches' /str(first_case_scan_path['scan_path']).lstrip(os.sep)
    gt_heatmap_path = img_dir / 'op_patches' /str(first_case_scaling_factors_path['gt_heatmap_path']).lstrip(os.sep)

    #Check for the required preprocessed data organization 
    required_structure = [
        scan_path, 
        gt_heatmap_path
    ]
    
    for path_to_check in required_structure:
        if not path_to_check.exists():
            print(f"PREPROCESSED DATA ERROR")
            print(f"The first case in your CSV exists, but is missing required subfolders.")
            print(f"Missing: {path_to_check}")
            print(f"Expected structure: pat > acc_num > [patched_inputs, gt_heatmaps] > [axial, coronal (optional), sagittal (optional)]")
            sys.exit(1)

    #get the best hyperparameter config from the previous experiment
    try:
        trial_results_df = pd.read_csv(hp_trial_results_path)
        trial_results_df.sort_values(['outer_fold','avg_cv_mre'], inplace = True)
        trial_results_df['rank'] = [x for x in range(25)]*5    

        #select the best config across test folds
        num_wins_per_hp = trial_results_df[trial_results_df['rank'] == 1].groupby('config').agg({'outer_fold':'count'}).reset_index()
        num_wins_per_hp.columns = ['config','num_wins']

        print(num_wins_per_hp)

        optim_hp = num_wins_per_hp[num_wins_per_hp['num_wins'] == num_wins_per_hp['num_wins'].max()]['config'].values[0]
        #converting the dict stored in string representation (as it is read from excel) back into dict format
        optim_hp = dict(zip([x.strip("{} ").split(":")[0].strip("'") for x in optim_hp.split(",")], 
                             [float(x.strip("{} ").split(":")[1]) for x in optim_hp.split(",")]))
       
    except Exception as e:
        sys.exit(f"ERROR: Could not read get the optimal hyperparmeters due to : {e}") 
    

    print(f"Data structure verified on the first case")
    print("------------------------------------------\n")

    return scan_info_df, pre_calc_scaling_factors_df, img_dir, optim_hp
    


class PreLoadDataset(Dataset):
    
    """
    Represents a pre-processed dataset class for training the 3D-UNet for fine-localization of the AC-PC landmarks. 
    Returns the 2-channel input patch, unmodulated 4-channel output patches, and the reference-standard AC and PC landmarks (ground-truth labels). 
    The unmodulated 4-channel output patches are multiplied with sigma (std. deviation of Gaussian heatmap, which is a hyper-parameter) and fully assembled             into 6-channel output patches (the 4 channels and 2 background channels based on the specific sigma) on the fly during training.
    
    Args:
        scan_paths: Location of the patched skull-stripped (and augmented) 3D input (head CT) scans.
        heatmap_paths: Location of the patched 3D Gaussian heatmaps centered around the reference-standard AC-PC. AC and PC heatmap patches are assumed to be               stacked in the same file.
        img_coordinates: Contains the coordinates of the coarse-localized AC-PC landmarks (which are at the center of the patched i/p and o/p patches) 
            and the reference-standard AC-PC which are the ground-truth labels. Each of these coordinates are assumed to be stored as string representations of             lists (e.g., '[1.0, 2.0, 3.0]'). 
        patch_size: Predetermined patch-size. 
        transform: Represents the preprocessing function to load the patched i/p and o/p .npy files, convert them to pytorch tensors, and reshape them into                 [2 x patch_size x patch_size x patch_size] inputs and [4 x patch_size x patch_size x patch_size] outputs. The 'labels' arrays are also converted to             tensors.         
    """
    
    def __init__(self, scan_paths, heatmap_paths, img_coordinates, patch_size, transform): 
        self.scan_path = scan_paths
        self.heatmap_path = heatmap_paths        
        self.img_coordinates = img_coordinates 
        self.patch_size = patch_size
        self.transform = transform
        
    def __len__(self):
        return len(self.img_coordinates)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input_path = self.scan_path[idx] 
        output_path = self.heatmap_path[idx]
        patch_size = self.patch_size
        label = np.array([np.float64(x.strip("[]()")) for x in self.img_coordinates.iloc[idx]['ac_img_true'].split(",")] + 
                         [np.float64(x.strip("[]()")) for x in self.img_coordinates.iloc[idx]['ac_img_coarse'].split(",")] +
                         [np.float64(x.strip("[]()")) for x in self.img_coordinates.iloc[idx]['pc_img_true'].split(",")] +
                         [np.float64(x.strip("[]()")) for x in self.img_coordinates.iloc[idx]['pc_img_coarse'].split(",")])
           
        if self.transform:
            image, target, label = self.transform(input_path, output_path, label, patch_size) 
     
        return {'scan_path': input_path, 'label':label,'image':image, 'target':target}

class PreLoadExternalDataset(Dataset):
    
    """Represents the pre-processed dataset class for testing a pretrained 3D-UNet for fine-localization of the AC-PC landmarks. 

    Args:
        scan_paths: Location of the patched skull-stripped (and augmented) 3D input (head CT) scans.
        img_coordinates: Contains the coordinates of the coarse-localized AC-PC landmarks (which are at the center of the patched i/p and o/p patches). Each of             these coordinates are assumed to be stored as string representations of lists (e.g., '[1.0, 2.0, 3.0]'). 
        patch_size: Predetermined patch-size. 
        transform: Represents the preprocessing function to load the patched i/p .npy files, convert them to pytorch tensors, and reshape them into                         [2 x patch_size x patch_size x patch_size] inputs. The 'labels' arrays are also converted to tensors.         
    """
    
    def __init__(self, scan_paths, img_coordinates, patch_size, transform): 
        self.scan_path = scan_paths    
        self.img_coordinates = img_coordinates 
        self.patch_size = patch_size
        self.transform = transform
        
    def __len__(self):
        return len(self.img_coordinates)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input_path = self.scan_path[idx] 

        label = np.array([np.float64(x.strip("[]()")) for x in self.img_coordinates.iloc[idx]['ac_img_coarse'].split(",")] + 
                         [np.float64(x.strip("[]()")) for x in self.img_coordinates.iloc[idx]['pc_img_coarse'].split(",")])
           
        if self.transform:
            image, label = self.transform(input_path, label, self.patch_size) 
     
        return {'scan_path': input_path, 'label':label,'image':image}
    

class DataTransform(object):
    """Returns the full 6-channel ground-truth gaussian heatmaps modulated with specified variance, and adding background channels. 
    Args:
        sigma (float): Desired variance of the gaussian to use for heatmap regression. 
        pre_calc_scaling_factors: Dataframe that contains image specific scaling factors for the gaussian heatmaps for specific sigma vals
    """

    def __init__(self, sigma, pre_calc_scaling_factors):
        #initialize the configuration of this class with the chosen sigma (hyperparameter) and the pre-calculated scaling factors dataframe
        self.sigma = sigma
        self.pre_calc_scaling_factors = pre_calc_scaling_factors
        
    def __call__(self, sample): 
        #make this a callable class so that the transformation parameters (sigma and pre_calc_scaling_factors) need not be passed every time it's called
        target = sample['target']
        scan_path = sample['scan_path']
       
        patch_size = sample['patch_size']

        #get the precalculated scaling factors for the scan at scan_path and chosen sigma        
        filter_cond = (self.pre_calc_scaling_factors['sigma'] == self.sigma) & (self.pre_calc_scaling_factors['scan_path'] == scan_path)
        min_gaussian_heatmap_ac = self.pre_calc_scaling_factors['min_gaussian_heatmap_ac'][filter_cond].values[0]
        max_gaussian_heatmap_ac = self.pre_calc_scaling_factors['max_gaussian_heatmap_ac'][filter_cond].values[0]        
        min_gaussian_heatmap_pc = self.pre_calc_scaling_factors['min_gaussian_heatmap_pc'][filter_cond].values[0]
        max_gaussian_heatmap_pc = self.pre_calc_scaling_factors['max_gaussian_heatmap_pc'][filter_cond].values[0]

        #Modulate the half-computed AC and PC heatmap patches by the chosen sigma and scale them to construct the full Gaussians
        pc_patch_ac_hm = torch.exp(-1/(2*self.sigma**2) * target[0,:,:,:])
        pc_patch_ac_hm = (pc_patch_ac_hm - min_gaussian_heatmap_ac)/(max_gaussian_heatmap_ac - min_gaussian_heatmap_ac)
        ac_patch_ac_hm = torch.exp(-1/(2*self.sigma**2) * target[1,:,:,:])
        ac_patch_ac_hm = (ac_patch_ac_hm - min_gaussian_heatmap_ac)/(max_gaussian_heatmap_ac - min_gaussian_heatmap_ac)
        
        pc_patch_pc_hm = torch.exp(-1/(2*self.sigma**2) * target[2,:,:,:])
        pc_patch_pc_hm = (pc_patch_pc_hm - min_gaussian_heatmap_pc)/(max_gaussian_heatmap_pc - min_gaussian_heatmap_pc)
        ac_patch_pc_hm = torch.exp(-1/(2*self.sigma**2) * target[3,:,:,:])
        ac_patch_pc_hm = (ac_patch_pc_hm - min_gaussian_heatmap_pc)/(max_gaussian_heatmap_pc - min_gaussian_heatmap_pc)
        
        #Compute background heatmaps for the AC and PC patches based on the chosen sigma
        pc_patch_bg_hm = 1 - pc_patch_ac_hm - pc_patch_pc_hm
        ac_patch_bg_hm = 1 - ac_patch_ac_hm - ac_patch_pc_hm

        #Assemble the full 6-channel groundtruth heatmap for supervised training
        gt_pat = torch.cat((pc_patch_ac_hm, pc_patch_pc_hm, pc_patch_bg_hm, 
                            ac_patch_ac_hm, ac_patch_pc_hm, ac_patch_bg_hm)).reshape(6,patch_size,patch_size,patch_size)

        return gt_pat    
    
    
class ACPCLandMarkDataset(Dataset):
    
    """Represents the dataset class for training the 3D-UNet to predict the AC-PC landmarks. 
       Returns the input patch, ground-truth gaussian heatmap for heatmap regression, the image coordinates of the reference-standard and coarse-localized              AC-PC landmarks, scan_path of the input scan being considered, and the image type (axial, coronal, sagittal). 

    Args:
        input_tensors: The long tensor stacked with the patched/cropped input images, which is already read into CPU memory
        output_tensors: The long tensor stacked with the patched ground-truth heatmaps, already read into CPU memory. 
        label_tensors: The long tensor stacked with the image coordinates of the reference-standard and coarse-localized AC-PC landmarks, read into CPU memory. 
        scan_paths: Scan paths of the input_tensors, indexed in the same order as the input_tensors were stacked 
        patch_size: Patch-size of the input and output samples
        transform: The sigma-parameterized DataTransform object which will be used to assemble full 6-channel heatmaps  
    """
    
    def __init__(self, input_tensors, output_tensors, label_tensors, scan_paths, patch_size, transform):

        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
        self.label_tensors = label_tensors    
        self.scan_paths = scan_paths
        self.patch_size = patch_size
        self.transform = transform #will be parameterized based on sigma 

    def __len__(self):
        return len(self.label_tensors)    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.input_tensors[idx] 
        target = self.output_tensors[idx]
        label = self.label_tensors[idx]
        scan_path = self.scan_paths[idx]

        if self.transform:
            target = self.transform({'target':target, 'scan_path':scan_path, 'patch_size':self.patch_size})
     
        return {'image':image, 'target':target, 'label':label, 'scan_path':scan_path}



class ExternalTestDataset(Dataset):
    
    """Represents the dataset class for testing the pretrained 3D-UNet for fine-localization of the AC-PC landmarks on an external dataset. 
       Returns the input patch, the image coordinates of the coarse-localized AC-PC landmarks, scan_path of the input scan being considered, and the image type         (axial, coronal, sagittal). 
    Args:
        input_tensors: The long tensor stacked with the patched/cropped input images, which is already read into CPU memory
        label_tensors: The long tensor stacked with the image coordinates of the reference-standard and coarse-localized AC-PC landmarks, read into CPU memory. 
        scan_paths: Scan paths of the input_tensors, indexed in the same order as the input_tensors were stacked 
        patch_size: Patch-size of the input and output samples
    """
    
    def __init__(self, input_tensors, label_tensors, scan_paths, patch_size):

        self.input_tensors = input_tensors
        self.label_tensors = label_tensors    
        self.scan_paths = scan_paths
        self.patch_size = patch_size

    def __len__(self):
        return len(self.label_tensors)    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.input_tensors[idx] 
        label = self.label_tensors[idx]
        scan_path = self.scan_paths[idx]

     
        return {'image':image, 'label':label, 'scan_path':scan_path}

