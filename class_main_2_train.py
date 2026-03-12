# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:30:26 2023

@author: benjaminstoeckigt
"""

import glob
from utils.class_run import force_sample
from pytorch.train import train_init

#FORCE
preprocess_params = {
    "project_name": "test", #Project Name that will be the name of output folder in temp & result subfolder
    "process_folder": "/uge_mount/Freddy/process/", # Folder where Data and Results will be processed (will be created if not existing)
    "aois": glob.glob(f"/uge_mount/Freddy/data/referenzpunkte/biotoptypen/test4/*.shp"),## reference points shape as single file or file list ## should have YYYY in name
    "years": None,  ###Oberservation Year (last year of the timeseries, e. g. [2025]), that should be defined for every Point Shapefile - if "None" Years will be extracted from aoi FileName YYYY
    "time_range": ["3", "03-01"],  # [time_range in years, start and end MM-DD for timeseries]
    "column_name": "class", #column name for response variable in points
    "Interpolation" : False, ## Classification based on not interpolated Data just possible with Transformer
    "INT_DAY" : 10, ## interpolation time steps
    ###########################################
    ########Advanced Parameters################
    ###########################################
    "force_dir": "/force", # mount directory for FORCE-Datacube - should look like /force_mount/FORCE/C1/L2/..
    "thermal_time": "/uge_mount/Freddy/data/thermal_encoding/concatenated_gdd_start2015_3035.tif", #set None if not using, take care of starting date from gdd -> class_run.py def(calculate_band_index)
    "hold": False,  # if True, FORCE cmd must be closed manually ## recommended for debugging FORCE
    "Sensors": "SEN2A SEN2B",  # LND04 LND05 LND07 LND08 LND09 SEN2A SEN2B,
    "Indices": "BLUE GREEN RED NIR SWIR1 SWIR2 RE1 RE2 RE3 BNIR", # Type: Character list. Valid values: {BLUE,GREEN,RED,NIR,SWIR1,SWIR2,RE1,RE2,RE3,BNIR,NDVI,EVI,NBR,NDTI,ARVI,SAVI,SARVI,TC-BRIGHT,TC-GREEN,TC-WET,TC-DI,NDBI,NDWI,MNDWI,NDMI,NDSI,SMA,kNDVI,NDRE1,NDRE2,CIre,NDVIre1,NDVIre2,NDVIre3,NDVIre1n,NDVIre2n,NDVIre3n,MSRre,MSRren,CCI},
    "SPECTRAL_ADJUST": "FALSE", # spectral adjustment will be necessary by using Sentinel 2 & Landsat together
    "INTERPOLATE": 'RBF',  # NONE,LINEAR,MOVING,RBF,HARMONIC ## Just necessary if OUPUT_TSI == True
    "ABOVE_NOISE": 3, # noise filtering in spectral values above 3 x std
    "BELOW_NOISE": 1, # get back values from qai masking below single std
    #Streaming Mechnism FORCE
    "NTHREAD_READ": 7,  # 4,
    "NTHREAD_COMPUTE": 7,  # 11,
    "NTHREAD_WRITE": 2,  # 2,
    "BLOCK_SIZE": 3000,
    ############################################################
    #################Data Splitting#############################
    ############################################################
    'split_method': "user_defined",  # "user_defined": every shapefile must have "train", "val" or "test" in file name, "random": random split into train, val and test, "random_test": random split intop train, test, "no_split": no split (note: if 'final_training = False' and split_method = random_test OR no_split data will be splitted random into train, val)
    'split_ratio': 0.8,  # split ratio for training, other part is validation/test # only used if split_method = random
    "seed": 42,  # seed for train validation split
    "feature_order": ["BLU", "GRN", "RED", "NIR", "SW1", "SW2", "RE1", "RE2", "RE3", "BNR"], # feature order related to FORCE output [x.split('_')[-2]] --> naming convention e.g.: 2022-2023_001-365_HL_TSA_SEN2L_SW2_TSS.tif
    "start_doy_month": None, ### Define start date [YYYY-MM-DD], If "None" DOY will be starting from individual timeseries start
}


args_train = {
    'epochs': 10,  # number of training epochs
    'valid_every_n_epochs': 2,  # skip some valid epochs for faster overall training
    'checkpoint_every_n_epochs': 2,  # save checkpoints during training
    'model': "transformer",  # "tempcnn","rnn","msresnet","transformer", "rf"
    'response': "classification",  # "classification" -> softmax, "regression" -> raw output, "regression_relu" -> 0 to infinity output, "regression_sigmoid" -> 0 to 1 output
    'final_training': False,  # all data is used for training. use for final inference model
    ###########################################
    ########Advanced Parameters################
    ###########################################
    'augmentation': 1, # Percentage x*100 % for augmenting Training Data with DOY Day Shifting / annual Gaussian Scaling / Zero Out
    'augmentation_plot': None, #Plotting for Augmentations; either None or BandNumber [None, 1, 2, 3, 4, 5, ...]
    'classes_lst': [0, 1],
    #'classes_lst': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48], #classification classes
    'use_class_weights': "train", # None: use for balanced datasets, 'train': use for imbalanced dataset, 'valid': only use when train and valid dataset have same imbabalance # only used if response = classification
    'tune': False,  # Hyperparameter Tune? True: new folder next to the model will be created named optuna. You can easily visualize statistics with optuna-dashboard /path/to/optuna/config
    'study_name': "test", # Name for Hyperparameter Trial
    'seed': 42,  # seed for batching and weight initialization
    'validation_metric': "f1", # metric used for hyperparameter tuning: "f1": use for imbalance data, "acc": use for stratified data -> only used if response = classification
    'max_seq_length': int(preprocess_params["time_range"][0])*366,
}

if __name__ == '__main__':
    force_sample(preprocess_params) # splits for single domain then goes to next
    train_init(args_train, preprocess_params)
