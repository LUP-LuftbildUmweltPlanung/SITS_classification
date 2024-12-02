# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:30:26 2023

@author: benjaminstoeckigt
"""

import glob
from utils.sampling_run import sampling
from utils.sampling_run import analyze_shapefiles
import os
#####################################
###########SAMPLING##################
#####################################


sampling_params = {
    ###BASIC PARAMS###
    "project_name": "uge_sealed_strat", #Project Name that will be the name of the output folder in temp & result subfolder
    "process_folder": "/uge_mount/FORCE/new_struc/process/",
    #input aoi (can be list or single files) --> CONDITION: "City"_"Year"_*.shp
    "aoi_files": sorted(glob.glob(r"/uge_mount/FORCE/new_struc/process/data/sealed/training/*.shp")), #aoi as shapefile ## can be multiple shapefiles /path/to/shapes/*.shp
    "output_n" : ["berlin","frankfurt","leipzig","potsdam"], #naming of output sampling points xx_**.shp, e.g. ["sbs"]
    "output_n_m" : ["2021","2021","2022","2022"], #naming of output sampling points **_xx.shp, e.g. ["2023"]
    "percent": 0.4, # [>0 - 1] percentage for amount of possible 10 m based points x*100 %
    "distance": 20, # distance between points in m
    #####################################
    ### conditional raster stratification
    #####################################
    #conditional raster file paths (lists with same file order related to aoi_files), For excluding Stratification set vegh_files and tcd_files variables to None
    "raster_files1" : sorted(glob.glob(r"/uge_mount/FORCE/new_struc/process/data/sealed/training/*.tif")),  # sampling raster 1
    #"raster_files2" : sorted(glob.glob(r"/uge_mount/FORCE/new_struc/process/data/sealed/training/*.tif")),  # sampling raster 2, if just raster 1 --> [None]
    #"raster_files1": None,
    "raster_files2" : None,
    # Ranges for Stratification [(range_min, range_max, x*100 %), (...), ...]
    "value_ranges_raster1": [(0, 0.33, 0.33), (0.33, 0.66, 0.33), (0.66, 1, 0.34)],
    "value_ranges_raster2": [(0, 0.1, 0.33), (0.1, 10, 0.33), (10, 100, 0.34)],
    }

##EXTRACT VALUES
# Parameters for extracting Values with sampled Points (Output Folder will be stored in path_params['data_folder']/_SamplingPoints/sampling_params['project_name']
from utils.sampling_run import extract_ref
extract_param = {
    "raster_path" : sorted(glob.glob(r"/uge_mount/FORCE/new_struc/process/data/sealed/training/*.tif")),
    "column_name" : "seal", #attribute column name to store the response variable
    }

sampling_params["output_n_m"] = [os.path.basename(file).split('_')[1].split('.')[0] for file in sampling_params["aoi_files"]]#
sampling_params["output_n"] = [os.path.basename(file).split('_')[0] for file in sampling_params["aoi_files"]]#

if __name__ == '__main__':

    sampling(**sampling_params)
    extract_ref(**sampling_params, **extract_param)
