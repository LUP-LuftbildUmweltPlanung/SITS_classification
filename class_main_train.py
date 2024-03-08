# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:30:26 2023

@author: Admin
"""

import os
import glob
import sys
import re
sys.path.append('../sits_force')
from sits_force.force_backend.force_class_utils import force_class
from utils.class_run import sample_to_ref_sepfiles
from pytorch.train import train_init
from config_path import path_params


#FORCE
preprocess_params = {
    "project_name" : "phd_vgh_tss_3years_EQUAL_noOutlier",
    "time_range" : ["3","10-01"], # [time_range in years, start MM-DD for doy] !!
    "aois" : glob.glob(f"/uge_mount/FORCE/new_struc/data/phd_publi1/Reference_Points_UGE_Chrisadjusted_equalized/Sat_20pct_Pix_Sen20m_NewClasses_TCD/*extract.shp"), ## should have YYYY in name  ###PLACEHOLDER ### should have YYYY in name
    "column_name": 'tcd',
    "years" : None, ###PLACEHOLDER, if None Years will be extracted from aoi FileName
    "date_ranges" : None, ###PLACEHOLDER, if None date_ranges will be extracted from years and time_ranges
    "NTHREAD_READ" : 7, #4,
    "NTHREAD_COMPUTE" : 7, #11,
    "NTHREAD_WRITE" : 2, #2,
    "BLOCK_SIZE" : 3000,
    "Indices" : "BLUE GREEN RED NIR SWIR1 SWIR2 RE1 RE2 RE3 BNIR",#Type: Character list. Valid values: {BLUE,GREEN,RED,NIR,SWIR1,SWIR2,RE1,RE2,RE3,BNIR,NDVI,EVI,NBR,NDTI,ARVI,SAVI,SARVI,TC-BRIGHT,TC-GREEN,TC-WET,TC-DI,NDBI,NDWI,MNDWI,NDMI,NDSI,SMA,kNDVI,NDRE1,NDRE2,CIre,NDVIre1,NDVIre2,NDVIre3,NDVIre1n,NDVIre2n,NDVIre3n,MSRre,MSRren,CCI},
    "Sensors" : "SEN2A SEN2B", #LND04 LND05 LND07 LND08 LND09 SEN2A SEN2B,
    "SPECTRAL_ADJUST" : "FALSE",
    "ABOVE_NOISE" : 0,
    "BELOW_NOISE" : 0,
    "OUTPUT_TSS" : 'TRUE',
    "OUTPUT_TSI": 'FALSE',
    "INTERPOLATE" : 'RBF', # NONE,LINEAR,MOVING,RBF,HARMONIC
    "INT_DAY" : 5,
    "hold" : False,  # execute cmd
    "sample": True,
    }

sampleref_param = {
    "project_name" : preprocess_params["project_name"],
    #### sample_toref
    "output_folder": f'{path_params["proc_folder"]}/_SITSrefdata/{preprocess_params["project_name"]}',
    "seed": 42,
    "band_names": ["BLU", "GRN", "RED", "NIR", "SW1", "SW2", "RE1", "RE2", "RE3", "BNR"],
    "split_train": 0.9, ### [0-1] for random split | [2010, ..., 2024, ..] for year test split (shapefile folder name)
    "start_doy_month": preprocess_params["time_range"], ###PLACEHOLDER #[start MMDD for doy, previous years for doy start] !! reference to the processing year has to be in shapefile name folder !!
    "del_emptyTS": True, # if True empty timesteps gets deleted (TSS / Transformer), if False empty timesteps gets interpolated over bands (TSI)
    }

args_train = {
    'epochs': 1,  # number of training epochs
    'workers': 10,  # number of CPU workers to load the next batch
    'data_root': f'{path_params["proc_folder"]}/_SITSrefdata/{preprocess_params["project_name"]}/sepfiles/train/', # folder with CSV or cached NPY folder
    'store': f'{path_params["proc_folder"]}/_SITSModels/{preprocess_params["project_name"]}/',  # store run logger results
    #'data_root': f'/uge_mount/FORCE/new_struc/process/result/_SITSrefdata/tcd_1year_pe/phd_tcd_tss_1years_EQUAL_noOutlier/sepfiles/train/', # folder with CSV or cached NPY folder
    #'store': f'/uge_mount/FORCE/new_struc/process/result/_SITSrefdata/tcd_1year_pe/phd_tcd_tss_1years_EQUAL_noOutlier/',  # store run logger results
    'valid_every_n_epochs': 5,  # skip some valid epochs for faster overall training
    'checkpoint_every_n_epochs': 5,  # save checkpoints during training
    'seed': 42,  # seed for batching and weight initialization
    'ref_split': 0.8, # split ratio for training, other part is validation
    'model': "transformer",  # "tempcnn","rnn","msresnet","transformer"
    'tune': False, #Hyperparameter Tune?
    'study_name':"transformer_1year_equal_vh_80eptest2",
    'response': "regression_sigmoid",  # "classification", "regression_relu" Use ReLU for 0 to infinity output, "regression_sigmoid" Use sigmoid for 0 to 1 output
    'classes_lst': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'order': sampleref_param["band_names"],
    'normalizing_factor': 1e-4,
    #'years': int(preprocess_params["time_range"][0]), ###PLACEHOLDER #time series years for doy max sequence length
    'years': 3,
}

if __name__ == '__main__':

    #force_class(preprocess_params, **path_params)
    #sample_to_ref_sepfiles(sampleref_param, **path_params) # splits for single domain then goes to next
    #train_init(args_train)


    for year in [2018, 2019, 2020, 2021, 2022]:

        preprocess_params["project_name"] = "phd_vgh_tss_3years_EQUAL_noOutlier"
        #preprocess_params["project_name"] = f'{preprocess_params["project_name"]}_test{year}'
        #force_class(preprocess_params, **path_params)
        sampleref_param["project_name"] = preprocess_params["project_name"]
        sampleref_param["output_folder"] = f'{path_params["proc_folder"]}/_SITSrefdata/{preprocess_params["project_name"]}_test{year}'
        sampleref_param["split_train"] = year
        sample_to_ref_sepfiles(sampleref_param, **path_params) # splits for single domain then goes to next
        try:
            del args_train['nclasses']
            del args_train['input_dims']
            del args_train['seqlength']
        except:
            print("probably first run ... no entries to delete")
        args_train["data_root"] = f'{path_params["proc_folder"]}/_SITSrefdata/{preprocess_params["project_name"]}_test{year}/sepfiles/train/'
        args_train["store"] = f'{path_params["proc_folder"]}/_SITSModels/{preprocess_params["project_name"]}_test{year}/'

        train_init(args_train)
