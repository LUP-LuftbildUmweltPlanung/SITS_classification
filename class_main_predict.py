# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:30:26 2023

@author: Admin
"""

import os
import glob
import time
import sys
sys.path.append('../sits_force')
from sits_force.force_backend.force_class_utils import force_class
from pytorch.predict import predict_init, load_preprocess_settings
from config_path import path_params


args_predict = {
    'project_name': "5year_sbs_pilot",
    'model_path': '/uge_mount/FORCE/new_struc/process/result/_SITSModels/sbs_pilot/transformer/model_e30.pth',
    'aois': glob.glob(f"/uge_mount/FORCE/new_struc/data/sbs_pilot/Testgebiet_2023.shp"), # aois can be path or list. Path for Force Tile folder or list for process structure and shapefiles must be correct
    #'aois': None,
    'chunksize': 10000,
    #'reference_folder' : '/uge_mount/FORCE/new_struc/process/result/_SITSrefdata/phd_vgh_tss_3years_EQUAL_noOutlier/',
    'reference_folder' : None,
    'probability' : True, # just gets recognized if classification
}


if __name__ == '__main__':
    startzeit = time.time()

    preprocess_params = load_preprocess_settings(os.path.dirname(args_predict["model_path"]))
    preprocess_params["sample"] = False
    preprocess_params["project_name"] = args_predict["project_name"]
    preprocess_params["aois"] = args_predict["aois"]
    preprocess_params["date_ranges"] = None
    preprocess_params["years"] = None
    args_predict["time_range"] = preprocess_params["time_range"]

    force_class(preprocess_params, **path_params)
    predict_init(args_predict, **path_params)

    endzeit = time.time()
    print(f"{(endzeit-startzeit)/60} minutes")
