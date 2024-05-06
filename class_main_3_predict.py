# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:30:26 2023

@author: Admin
"""

import os
import glob
import time

from pytorch.predict import predict_init, load_preprocess_settings
from config_path import path_params
from force.force_class_utils import force_class

args_predict = {
    'project_name': "predict_test_workshop",
    'model_path': '/nne_mount/sits_framework/process/results/_SITSModels/test_workshop/transformer/model_e10.pth', # Path to Model
    'aois': glob.glob(f"/nne_mount/sits_framework/process/data/workshop_aoi/potsdam_2023_laea.shp"), # aois can be path or list. Path for Force Tile folder or list // process structure and shapefiles must be correct
    #'aois': None,
    'chunksize': 10000,
    #'reference_folder' : '/uge_mount/FORCE/new_struc/process/result/_SITSrefdata/phd_vgh_tss_3years_EQUAL_noOutlier/',
    'reference_folder' : None, #Set Path if you want to predict the Test CSV File
    'probability' : False, # just gets recognized if classification
}
##########################################
### Additional Setting - Can Be Ignored###
##########################################
preprocess_params = load_preprocess_settings(os.path.dirname(args_predict["model_path"]))
preprocess_params["sample"] = False
preprocess_params["project_name"] = args_predict["project_name"]
preprocess_params["aois"] = args_predict["aois"]
preprocess_params["date_ranges"] = None
preprocess_params["years"] = None
args_predict["time_range"] = preprocess_params["time_range"]


if __name__ == '__main__':
    startzeit = time.time()

    force_class(preprocess_params, **path_params)
    predict_init(args_predict, **path_params)

    endzeit = time.time()
    print(f"{(endzeit-startzeit)/60} minutes")









