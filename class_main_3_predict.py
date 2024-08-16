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
    'project_name': "prediction_VV_2020",
    "process_folder": "/nne_mount/sits_framework/process",
    'model_path': '/nne_mount/sits_framework/process/data/Models/vv_transformer_1year/model_e46.pth', # Path to Model
    'aois': glob.glob(f"/nne_mount/sits_framework/process/data/Germany_AOIs/germany_2020.shp"), # aois can be path or list. Path for Force Tile folder or list // process structure and shapefiles must be correct
    #'aois': None,
    'years': None, ###Oberservation Year (last year of the timeseries), that should be defined for every AOI Shapefile - if "None" Years will be extracted from aoi FileName YYYY
    #'reference_folder' : '/uge_mount/FORCE/new_struc/process/result/_SITSrefdata/envilink_vv_1years/',
    'reference_folder' : None, #Set Path if you want to predict the Test CSV File /path/to/_SITSrefdata/projectname
    'probability' : False, # just gets recognized if classification
    'chunksize': 6000,  # 5years ts -> 2000
    'force_dir': "/force",
    }

if __name__ == '__main__':
    startzeit = time.time()
    predict(args_predict)
    endzeit = time.time()
    print(f"{(endzeit-startzeit)/60} minutes")









