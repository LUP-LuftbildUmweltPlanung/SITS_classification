# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:30:26 2023

@author: benjaminstoeckigt
"""

import os
import glob
import time

from pytorch.predict import predict


args_predict = {
    'project_name': "vc_germany_2024",
    "process_folder": "/uge_mount/FORCE/new_struc/process/",
    #'model_path': '/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_cc_thermal_3y/transformer/model_e88.pth', # Path to Model
    'model_path': '/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vc_thermal_test_invekos_ackerone_3y_nodropout/transformer/model_e62.pth', # Path to Model,
    'model_path2': None, # if multiple models are based on the same time series, you can safe preprocessing satellite data
    'model_path3': None,
    'aois': glob.glob("/uge_mount/FORCE/new_struc/process/data/germany_exact_annual/germany_exact_2024.shp"), # aois can be path or list. Path for Force Tile folder or list // process structure and shapefiles must be correct
    #'aois': None,
    'years': None, ###Oberservation Year (last year of the timeseries), that should be defined for every AOI Shapefile - if "None" Years will be extracted from aoi FileName YYYY
    'reference_folder': None,
    #'reference_folder' : "/uge_mount/FORCE/new_struc/process/results/_SITSrefdata/class_vh_thermal_3y_2022DElo_JoThermalJoAug/",#None, #Set Path if you want to predict the Test CSV File /path/to/_SITSrefdata/projectname
    'probability': False, # just gets recognized if classification
    'chunksize': 6000,  # 5years ts -> 2000
    'tmp_cleanup': True, # clean processed sentinel-2 timeSeries after prediction of individual tile
    'thermal_time_prediction': "/uge_mount/FORCE/new_struc/process/data/gdd/concatenated_gdd_start20150101_end20250630_3035.tif", #set None if not using
    'force_dir': "/force",
    }

if __name__ == '__main__':
    predict(args_predict)











