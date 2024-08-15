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
    'project_name': "ogi_tcd",
    "process_folder": "/uge_mount/FORCE/new_struc/process/",
    'model_path': '/uge_mount/FORCE/new_struc/process/results/_SITSModels/class_tcd_final/transformer/model_e52.pth', # Path to Model
    'aois': glob.glob(f"/uge_mount/FORCE/new_struc/process/data/aois_test_class_ogi/*.shp"), # aois can be path or list. Path for Force Tile folder or list // process structure and shapefiles must be correct
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









