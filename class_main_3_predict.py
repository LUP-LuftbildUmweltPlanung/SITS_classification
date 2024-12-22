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
    'project_name': "tcd_v2_devtest_double_dingodiffheight",
    "process_folder": "/uge_mount/FORCE/new_struc/process/",
    'model_path': '/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_cc_thermal_3y/transformer/model_e88.pth', # Path to Model
    'model_path2': '/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_vh_thermal_3y/transformer_respNone/model_e110.pth', # Path to Model,
    'aois': glob.glob("/uge_mount/FORCE/new_struc/process/data/devtest_veg_aois/dingo_dif*.shp"), # aois can be path or list. Path for Force Tile folder or list // process structure and shapefiles must be correct
    #'aois': None,
    'years': None, ###Oberservation Year (last year of the timeseries), that should be defined for every AOI Shapefile - if "None" Years will be extracted from aoi FileName YYYY
    'reference_folder' : None,
    #'reference_folder' : "/uge_mount/FORCE/new_struc/process/results/_SITSrefdata/class_vh_thermal_3y_2022DElo_JoThermalJoAug/",#None, #Set Path if you want to predict the Test CSV File /path/to/_SITSrefdata/projectname
    'probability' : False, # just gets recognized if classification
    'chunksize': 6000,  # 5years ts -> 2000
    'tmp_cleanup': True,
    'thermal_time_prediction': "/uge_mount/FORCE/new_struc/process/data/gdd/concatenated_gdd_start2015_3035.tif", #set None if not using
    'force_dir': "/force",
    }

if __name__ == '__main__':
    predict(args_predict)











