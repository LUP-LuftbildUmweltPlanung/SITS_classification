# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:30:26 2023

@author: Admin
"""

import os
import glob
import sys
sys.path.append('../sits_force')
from sits_force.force_backend.force_class_main import *
from utils.class_run import sample_to_ref_onefile, sample_to_ref_sepfiles
from pytorch.train import train_init

base_params = {
    "force_dir": "/force:/force",
    "local_dir": "/uge_mount:/uge_mount",
    "force_skel": "/uge_mount/FORCE/new_struc/scripts/force/skel/force_cube_sceleton",
    "scripts_skel": "/uge_mount/FORCE/new_struc/scripts/force/skel",
    "temp_folder": "/uge_mount/FORCE/new_struc/process/temp",
    "mask_folder": "/uge_mount/FORCE/new_struc/process/mask",
    "proc_folder": "/uge_mount/FORCE/new_struc/process/result",
    "data_folder": "/uge_mount/FORCE/new_struc/data",
    ###BASIC PARAMS###
    "project_name": "uge_class_gv_10dint_3year_nodatafilled",
    "hold": False,  # execute cmd
    }


#FORCE
#preprocess_params['aois'] = glob.glob(f"{base_params['data_folder']}/_ReferencePoints/{base_params['project_name']}/*shp")
preprocess_params['aois'] = glob.glob(f"/uge_mount/FORCE/new_struc/data/uge_class_20230823/Reference_Points_UGE_Chrisadjusted/Sat_20pct_Pix_Sen20m_NewClasses_GV/*.shp")
preprocess_params['years'] = [int(re.search(r'(\d{4})', os.path.basename(f)).group(1)) for f in preprocess_params['aois'] if re.search(r'(\d{4})', os.path.basename(f))]
preprocess_params['date_ranges'] = [f"{year - 3}-10-01 {year}-09-30" for year in preprocess_params['years']]
preprocess_params['sample'] = True ##for sample csv named like shapes are necessary
preprocess_params['INT_DAY'] = 10
preprocess_params['column_name'] = 'gv'



sampleref_param = {
    #### sample_toref
    "output_folder": f"{base_params['proc_folder']}/_SITSrefdata/{base_params['project_name']}",
    "seed": 42,
    "bands": 10,
    "split_train": 0.9,
    }


args_train = {
    'batchsize': 256,  # batch size
    'epochs': 100,  # number of training epochs
    'workers': 10,  # number of CPU workers to load the next batch
    'data_root': f"{base_params['proc_folder']}/_SITSrefdata/{base_params['project_name']}/sepfiles/train/", # folder with CSV or cached NPY folder
    'store': f"{base_params['proc_folder']}/_SITSModels/{base_params['project_name']}",  # store run logger results
    'valid_every_n_epochs': 5,  # skip some valid epochs for faster overall training
    'checkpoint_every_n_epochs': 5,  # save checkpoints during training
    'seed': 0,  # seed for batching and weight initialization
    'ref_split': 0.8, # split ratio for training, other part is validation
    'model': "transformer",  # "tempcnn","rnn","msresnet","transformer"
    'tune': False, #Hyperparameter Tune?
    'study_name':"tempcnn_rmse_tcd",
    'response': "regression_relu",  # "classification", "regression_relu" Use ReLU for 0 to infinity output, "regression_sigmoid" Use sigmoid for 0 to 1 output
    'classes_lst': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'order': ["BLU", "GRN", "RED", "NIR", "SW1", "SW2", "RE1", "RE2", "RE3", "BNR"],
    'normalizing_factor': 1e-4,
    'padding_value': -1
}

if __name__ == '__main__':
    force_class(**base_params, **preprocess_params)
    sample_to_ref_sepfiles(**base_params, **sampleref_param) # splits for single domain then goes to next

    #base_params['project_name'] = 'uge_class_gv_10dint_2year_nodatafilled'
    #args_train['data_root'] = f"{base_params['proc_folder']}/_SITSrefdata/{base_params['project_name']}/sepfiles/train/"
    #args_train['store'] = f"{base_params['proc_folder']}/_SITSModels/{base_params['project_name']}"
    train_init(args_train)

