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
    "project_name" : "class_vh_thermal_3y", #Project Name that will be the name of output folder in temp & result subfolder
    "process_folder": "/uge_mount/FORCE/new_struc/process/", # Folder where Data and Results will be processed (will be created if not existing)
    "aois" : glob.glob(f"/uge_mount/FORCE/new_struc/process/results/_SamplingPoints/uge_vgh_30m_equalized/*extract.shp"),## reference points shape as single file or file list ## should have YYYY in name
    "years": None,  ###Oberservation Year (last year of the timeseries), that should be defined for every Point Shapefile - if "None" Years will be extracted from aoi FileName YYYY
    "time_range": ["3", "06-01"],  # [time_range in years, start and end MM-DD for timeseries]
    "column_name": 'seal', #column name for response variable in points
    "Interpolation" : False, ## Classification based on not interpolated Data just possible with Transformer
    "INT_DAY" : 10, ## interpolation time steps
    ###########################################
    ########Advanced Parameters################
    ###########################################
    "force_dir": "/force", # mount directory for FORCE-Datacube - should look like /force_mount/FORCE/C1/L2/..
    "thermal_time": "/uge_mount/FORCE/new_struc/process/data/gdd/concatenated_gdd_start2015_3035.tif", #set None if not using, take care of starting date from gdd -> class_run.py def(calculate_band_index)
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
    "split_train": 5000,  ### [0-1] for random split | [2010, ..., 2024, ..] for year test split (shapefile folder name)
    "seed": 42,  # seed for train validation split
    "feature_order": ["BLU", "GRN", "RED", "NIR", "SW1", "SW2", "RE1", "RE2", "RE3", "BNR"], # feature order related to FORCE output [x.split('_')[-2]] --> naming convention e.g.: 2022-2023_001-365_HL_TSA_SEN2L_SW2_TSS.tif
    "start_doy_month": None, ### Define start date [YYYY-MM-DD], If "None" DOY will be starting from individual timeseries start
}


args_train = {
    'epochs': 160,  # number of training epochs
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
    'study_name': "Tune_TCD_3Years", # Name for Hyperparameter Trial
    'seed': 42,  # seed for batching and weight initialization
    'max_seq_length': int(preprocess_params["time_range"][0])*366,
}

if __name__ == '__main__':

    #force_sample(preprocess_params) # splits for single domain then goes to next
    train_init(args_train, preprocess_params)


    #preprocess_params["project_name"] = "class_cc_thermal_3y"
    #train_init(args_train, preprocess_params)

    # model_f = "/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vh_thermal_3y_2022DElo_JoThermalJoAug/transformer"
    # model_new_f = "/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vh_thermal_3y_2022DElo_JoThermalJoAug/transformer_normal_Aug"
    # import os
    # try:
    #     #os.rename(refdata_f, refdata_new_f)
    #     os.rename(model_f, model_new_f)
    # except OSError as e:
    #     print(f"Error renaming")
    # preprocess_params["thermal_time"] = "/uge_mount/FORCE/new_struc/process/data/gdd/concatenated_gdd_start2015_3035.tif"
    # train_init(args_train, preprocess_params)
    #
    #
    # preprocess_params["tmp"] = False
    # preprocess_params["split_train"] = "2022"
    # force_sample(preprocess_params) # splits for single domain then goes to next
    # train_init(args_train, preprocess_params)
    #
    # refdata_f = "/uge_mount/FORCE/new_struc/process/results/_SITSrefdata/class_vh_thermal_3y"
    # refdata_new_f = "/uge_mount/FORCE/new_struc/process/results/_SITSrefdata/class_vh_thermal_3y_trainNoDuisEssen1822Gießen_noAugjoThermal"
    # model_f = "/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vh_thermal_3y"
    # model_new_f = "/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vh_thermal_3y_trainNoDuisEssen1822Gießen_noAugjoThermal"
    #
    # import os
    # try:
    #     os.rename(refdata_f, refdata_new_f)
    #     os.rename(model_f, model_new_f)
    # except OSError as e:
    #     print(f"Error renaming")
    #
    # preprocess_params["tmp"] = False
    # preprocess_params["split_train"] = "guetersloh"
    #
    # force_sample(preprocess_params) # splits for single domain then goes to next
    # train_init(args_train, preprocess_params)
    #
    # refdata_f = "/uge_mount/FORCE/new_struc/process/results/_SITSrefdata/class_vh_thermal_3y"
    # refdata_new_f = "/uge_mount/FORCE/new_struc/process/results/_SITSrefdata/class_vh_thermal_3y_trainNoDuisEssen1822Guetersloh_noAugjoThermal"
    # model_f = "/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vh_thermal_3y"
    # model_new_f = "/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vh_thermal_3y_trainNoDuisEssen1822Guetersloh_noAugjoThermal"
    # import os
    # try:
    #     os.rename(refdata_f, refdata_new_f)
    #     os.rename(model_f, model_new_f)
    # except OSError as e:
    #     print(f"Error renaming")
    #
    #
    # preprocess_params["split_train"] = "marburg"
    #
    # force_sample(preprocess_params)  # splits for single domain then goes to next
    # train_init(args_train, preprocess_params)
    #
    # refdata_f = "/uge_mount/FORCE/new_struc/process/results/_SITSrefdata/class_vh_thermal_3y"
    # refdata_new_f = "/uge_mount/FORCE/new_struc/process/results/_SITSrefdata/class_vh_thermal_3y_trainNoDuisEssen1822Marburg_noAugjoThermal"
    # model_f = "/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vh_thermal_3y"
    # model_new_f = "/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vh_thermal_3y_trainNoDuisEssen1822Marburg_noAugjoThermal"
    # import os
    # try:
    #     os.rename(refdata_f, refdata_new_f)
    #     os.rename(model_f, model_new_f)
    # except OSError as e:
    #     print(f"Error renaming")
    #
    #
    # preprocess_params["split_train"] = "vechta"
    #
    # force_sample(preprocess_params)  # splits for single domain then goes to next
    # train_init(args_train, preprocess_params)
    #
    # refdata_f = "/uge_mount/FORCE/new_struc/process/results/_SITSrefdata/class_vh_thermal_3y"
    # refdata_new_f = "/uge_mount/FORCE/new_struc/process/results/_SITSrefdata/class_vh_thermal_3y_trainNoDuisEssen1822Vechta_noAugjoThermal"
    # model_f = "/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vh_thermal_3y"
    # model_new_f = "/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vh_thermal_3y_trainNoDuisEssen1822Vechta_noAugjoThermal"
    # import os
    # try:
    #     os.rename(refdata_f, refdata_new_f)
    #     os.rename(model_f, model_new_f)
    # except OSError as e:
    #     print(f"Error renaming")
    #
    #
    # preprocess_params["split_train"] = "berlin"
    #
    # force_sample(preprocess_params)  # splits for single domain then goes to next
    # train_init(args_train, preprocess_params)
    #
    # refdata_f = "/uge_mount/FORCE/new_struc/process/results/_SITSrefdata/class_vh_thermal_3y"
    # refdata_new_f = "/uge_mount/FORCE/new_struc/process/results/_SITSrefdata/class_vh_thermal_3y_trainNoDuisEssen1822Berlin_noAugjoThermal"
    # model_f = "/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vh_thermal_3y"
    # model_new_f = "/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vh_thermal_3y_trainNoDuisEssen1822Berlin_noAugjoThermal"
    # import os
    # try:
    #     os.rename(refdata_f, refdata_new_f)
    #     os.rename(model_f, model_new_f)
    # except OSError as e:
    #     print(f"Error renaming")
    #
    #
    # preprocess_params["split_train"] = "dresden"
    #
    # force_sample(preprocess_params)  # splits for single domain then goes to next
    # train_init(args_train, preprocess_params)
    #
    # refdata_f = "/uge_mount/FORCE/new_struc/process/results/_SITSrefdata/class_vh_thermal_3y"
    # refdata_new_f = "/uge_mount/FORCE/new_struc/process/results/_SITSrefdata/class_vh_thermal_3y_trainNoDuisEssen1822VDresden_noAugjoThermal"
    # model_f = "/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vh_thermal_3y"
    # model_new_f = "/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vh_thermal_3y_trainNoDuisEssen1822Dresden_noAugjoThermal"
    # import os
    # try:
    #     os.rename(refdata_f, refdata_new_f)
    #     os.rename(model_f, model_new_f)
    # except OSError as e:
    #     print(f"Error renaming")
    #
    #
    # preprocess_params["split_train"] = "d"
    #
    # force_sample(preprocess_params)  # splits for single domain then goes to next
    # train_init(args_train, preprocess_params)
    #
    # refdata_f = "/uge_mount/FORCE/new_struc/process/results/_SITSrefdata/class_vh_thermal_3y"
    # refdata_new_f = "/uge_mount/FORCE/new_struc/process/results/_SITSrefdata/class_vh_thermal_3y_trainNoDuisEssen1822VDuisburg_noAugjoThermal"
    # model_f = "/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vh_thermal_3y"
    # model_new_f = "/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vh_thermal_3y_trainNoDuisEssen1822Duisburg_noAugjoThermal"
    # import os
    # try:
    #     os.rename(refdata_f, refdata_new_f)
    #     os.rename(model_f, model_new_f)
    # except OSError as e:
    #     print(f"Error renaming")
    #
    #
    # preprocess_params["split_train"] = "e"
    #
    # force_sample(preprocess_params)  # splits for single domain then goes to next
    # train_init(args_train, preprocess_params)
    #
    # refdata_f = "/uge_mount/FORCE/new_struc/process/results/_SITSrefdata/class_vh_thermal_3y"
    # refdata_new_f = "/uge_mount/FORCE/new_struc/process/results/_SITSrefdata/class_vh_thermal_3y_trainNoDuisEssen1822VEssen_noAugjoThermal"
    # model_f = "/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vh_thermal_3y"
    # model_new_f = "/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vh_thermal_3y_trainNoDuisEssen1822Essen_noAugjoThermal"
    # import os
    # try:
    #     os.rename(refdata_f, refdata_new_f)
    #     os.rename(model_f, model_new_f)
    # except OSError as e:
    #     print(f"Error renaming")
