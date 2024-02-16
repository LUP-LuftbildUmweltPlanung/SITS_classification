# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:30:26 2023

@author: Admin
"""

import os
import glob
from utils.sampling_run import sampling

#####################################
###########SAMPLING##################
#####################################


sampling_params = {
    #input aoi (can be list or single files) --> CONDITION: "City"_"Year"_*.shp
    "aoi_files": None, #need to have column area [m²] and has to be single part
    #if output_folder : None --> base_params
    #"output_folder" : r"O:\+DeepLearning_Extern\vegetationshoehen\Validierung\test"
    "output_folder" : None,
    "cities" : None,
    "years" : None,
    ## area percentage for amount of points
    "percent": 0.2,
    #####################################
    ### conditional raster stratification
    #####################################
    #Ranges
    "value_ranges_vegh": [(0, 0.1, 0.25), (0.1, 10, 0.25), (10, 20, 0.25), (20, 100, 0.25)],
    "value_ranges_tcd": [(0, 0.05, 0.2), (0.05, 0.33, 0.2), (0.33, 0.66, 0.2), (0.66, 0.95, 0.2), (0.95, 1, 0.2)],
    "sentinel20m": False,
    #conditional raster file paths (can be lists with same file assignments or single files) --> IF both None shape doesnt have to have area column
    #"vegh_files" : glob.glob(
        #r"O:\+DeepLearning_Extern\stacks_and_masks\masks\_force_10m_aggregation\+vegetation_height\augsburg_2022_vegetation_height_average_10m_FORCE.tif"),  # sampling raster 1
    #"tcd_files" : glob.glob(
        #r"O:\+DeepLearning_Extern\stacks_and_masks\masks\_force_10m_aggregation\+canopy_cover\augsburg_2022_canopy_cover_fraction_10m_FORCE.tif"),  # sampling raster 2, if just raster 1 --> [None]
    #####################################
    ### without stratification tcd & vegh has to be [None]
    #####################################
    "tcd_files" : [None,None],
    "vegh_files" : [None,None],
    "distance" : 10,
    }


base_params = {
    "force_dir": "/force:/force",
    "local_dir": "/uge_mount:/uge_mount",
    "force_skel": "/uge_mount/FORCE/new_struc/scripts_sits/sits_force/skel/force_cube_sceleton",
    "scripts_skel": "/uge_mount/FORCE/new_struc/scripts_sits/sits_force/skel",
    "temp_folder": "/uge_mount/FORCE/new_struc/process/temp",
    "mask_folder": "/uge_mount/FORCE/new_struc/process/mask",
    "proc_folder": "/uge_mount/FORCE/new_struc/process/result",
    "data_folder": "/uge_mount/FORCE/new_struc/data",
    ###BASIC PARAMS###
    "project_name": "aoi_sbs_class_202402",
    "hold": False,  # execute cmd
}
sampling_params["aoi_files"] = glob.glob(r"/uge_mount/FORCE/new_struc/data/sbs_pilot/*.shp")
sampling_params["years"] = [2023]#[os.path.basename(file).split('_')[1] for file in sampling_params["aoi_files"]]
sampling_params["cities"] = ["sbs"]#[os.path.basename(file).split('_')[0] for file in sampling_params["aoi_files"]]
sampling_params["percent"] = 100

sampling(**base_params,**sampling_params)

##EXTRACT VALUES
from utils.sampling_run import extract_ref
extract_param = {
    "script_path" : r"/uge_mount/FORCE/new_struc/scripts_sits/sits_classification/skel/zonal_rasterstats_mp.py",
    "shapefile_path" : sorted(glob.glob(f"{base_params['data_folder']}/_ReferencePoints/{base_params['project_name']}/*shp")),
    "raster_path" : sorted(glob.glob(r"/uge_mount/FORCE/new_struc/data/aoi_test_sbs/*.tif")),
    "o_folder" : f"{base_params['data_folder']}/_ReferencePoints/{base_params['project_name']}",
    "column_name" : "vh",
    }
extract_ref(**extract_param)

