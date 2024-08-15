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
    "project_name" : "class_vv_final", #Project Name that will be the name of output folder in temp & result subfolder
    "process_folder": "/uge_mount/FORCE/new_struc/process/", # Folder where Data and Results will be processed (will be created if not existing)
    "aois" : glob.glob(f"/uge_mount/FORCE/new_struc/process/results/_SamplingPoints/uge_tcd_30m_equalized/*.shp"),## reference points shape as single file or file list ## should have YYYY in name
    "years": None,  ###Oberservation Year (last year of the timeseries), that should be defined for every Point Shapefile - if "None" Years will be extracted from aoi FileName YYYY
    "time_range": ["1", "06-01"],  # [time_range in years, start and end MM-DD for timeseries]
    "column_name": 'tcd', #column name for response variable in points
    "Interpolation" : False, ## Classification based on not interpolated Data just possible with Transformer
    "INT_DAY" : 10, ## interpolation time steps
    ###########################################
    ########Advanced Parameters################
    ###########################################
    "force_dir": "/force", # mount directory for FORCE-Datacube - should look like /force_mount/FORCE/C1/L2/..
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
    ########Postprocessing Samples for Reference################
    ############################################################
    "split_train": 0.8,  ### [0-1] for random split | [2010, ..., 2024, ..] for year test split (shapefile folder name)
    "seed": 42,  # seed for train validation split
    "feature_order": ["BLU", "GRN", "RED", "NIR", "SW1", "SW2", "RE1", "RE2", "RE3", "BNR"], # feature order related to FORCE output [x.split('_')[-2]] --> naming convention e.g.: 2022-2023_001-365_HL_TSA_SEN2L_SW2_TSS.tif
    "start_doy_month": None, ### Define start date [YYYY-MM-DD], If "None" DOY will be starting from individual timeseries start
}


args_train = {
    'epochs': 100,  # number of training epochs
    'valid_every_n_epochs': 2,  # skip some valid epochs for faster overall training
    'checkpoint_every_n_epochs': 2,  # save checkpoints during training
    'ref_split': 0.8, # split ratio for training, other part is validation
    'model': "transformer",  # "tempcnn","rnn","msresnet","transformer", "rf"
    'response': "regression",  # "classification" -> softmax, "regression" -> raw output, "regression_relu" -> 0 to infinity output, "regression_sigmoid" -> 0 to 1 output
    ###########################################
    ########Advanced Parameters################
    ###########################################
    'augmentation': 1, # Percentage x*100 % for augmenting Training Data with DOY Day Shifting / annual Gaussian Scaling / Zero Out
    'augmentation_plot': None, #Plotting for Augmentations; either None or BandNumber [None, 1, 2, 3, 4, 5, ...]
    'classes_lst': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], #classification classes
    'tune': False,  # Hyperparameter Tune? True: new folder next to the model will be created named optuna. You can easily visualize statistics with optuna-dashboard /path/to/optuna/config
    'study_name': "test2", # Name for Hyperparameter Trial
    'seed': 42,  # seed for batching and weight initialization
    'max_seq_length': int(preprocess_params["time_range"][0])*367,
    'norm_factor_features': 1e-4,
    'norm_factor_response': "log10", #1e-2,#"log10", Response Scaling will be done after Caching, Should be None for Classification. Can be a Value e.g. 1e-3, None or "log10"
    ## take cre for norm_factor_response and regression_relu / regression_sigmoid
}

if __name__ == '__main__':

    #force_sample(preprocess_params) # splits for single domain then goes to next
    train_init(args_train, preprocess_params)


