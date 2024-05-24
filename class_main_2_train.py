# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:30:26 2023

@author: benjaminstoeckigt
"""

import glob
from utils.class_run import force_sample
from pytorch.train import train_init
from config_path import path_params

#FORCE
preprocess_params = {
    "project_name" : "test2", #Project Name that will be the name of output folder in temp & result subfolder
    "time_range" : ["1","10-01"], # [time_range in years, start MM-DD for doy] !!
    "aois" : glob.glob(f"/uge_mount/FORCE/new_struc/data/_SamplingPoints/test_workshop/potsdam_2023_points_extract.shp"), ## reference points shape as single file or file list ## should have YYYY in name
    "column_name": 'vgh', #column name for response variable in points
    "Interpolation" : False, ## Classification based on not interpolated Data just possible with Transformer
    "INT_DAY" : 10, ## interpolation time steps
    ###########################################
    ########Advanced Parameters################
    ###########################################
    "years": None,  ###PLACEHOLDER, if None Years will be extracted from aoi FileName
    "date_ranges": None,  ###PLACEHOLDER, if None date_ranges will be extracted from years and time_ranges
    "sample": True, ## Always True for Training
    "hold": False,  # if True, cmd must be closed manually ## recommended for debugging FORCE
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
    }

sampleref_param = {
    "split_train": 0.9, ### [0-1] for random split | [2010, ..., 2024, ..] for year test split (shapefile folder name)
    "del_emptyTS": True, # if True empty timesteps gets deleted (TSS / Transformer), if False empty timesteps gets interpolated over bands (TSI)
    ###########################################
    ########Advanced Parameters################
    ###########################################
    "output_folder": f'{path_params["proc_folder"]}/_SITSrefdata/{preprocess_params["project_name"]}',
    "band_names": ["BLU", "GRN", "RED", "NIR", "SW1", "SW2", "RE1", "RE2", "RE3", "BNR"],
    "seed": 42, # seed for train validation split
    "start_doy_month": preprocess_params["time_range"],###PLACEHOLDER #[time_range in years, start MM-DD for doy] !! reference to the processing year has to be in shapefile name folder !!
    }

args_train = {
    'epochs': 10,  # number of training epochs
    'valid_every_n_epochs': 2,  # skip some valid epochs for faster overall training
    'checkpoint_every_n_epochs': 2,  # save checkpoints during training
    'ref_split': 0.8, # split ratio for training, other part is validation
    'model': "transformer",  # "tempcnn","rnn","msresnet","transformer", "rf"
    'response': "regression_relu",  # "classification", "regression_relu" Use ReLU for 0 to infinity output, "regression_sigmoid" Use sigmoid for 0 to 1 output
    ###########################################
    ########Advanced Parameters################
    ###########################################
    'augmentation': 1, # Percentage x*100 % for augmenting Training Data with DOY Day Shifting / annual Gaussian Scaling / Zero Out
    'augmentation_plot': 4, #Plotting for Augmentations; either None or BandNumber [None, 1, 2, 3, 4, 5, ...]
    'classes_lst': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], #classification classes
    'tune': False,  # Hyperparameter Tune?
    'study_name': "test2", # Name for Hyperparameter Trial
    'seed': 42,  # seed for batching and weight initialization
    'years': int(preprocess_params["time_range"][0]),  ###PLACEHOLDER #time series years for doy max sequence length
    'norm_factor_features': 1e-4,
    'norm_factor_response': None,#Response Scaling will be done before Caching, Should be None for Classification. Can be a Value e.g. 1e-3, None or "log10"
    'order': sampleref_param["band_names"],
}

if __name__ == '__main__':

    force_sample(sampleref_param, preprocess_params, path_params) # splits for single domain then goes to next
    train_init(args_train, preprocess_params, path_params)


