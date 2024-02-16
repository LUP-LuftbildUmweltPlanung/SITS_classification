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
from sits_force.force_backend.force_class_main import *
from utils.class_run import mosaic_rasters
#os.chdir(r"E:\++++Promotion\Verwaltung\Publikation_1\Workflow_Scripts\final")
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
    "project_name": "sbs_pilot_result_test",
    "hold": False,  # execute cmd
    }

startzeit = time.time()

#FORCE
preprocess_params['aois'] = glob.glob(f"/uge_mount/FORCE/new_struc/data/sbs_pilot_test/sbs_pilot_test_2023.shp")
preprocess_params['years'] = [int(re.search(r'(\d{4})', os.path.basename(f)).group(1)) for f in preprocess_params['aois'] if re.search(r'(\d{4})', os.path.basename(f))]
preprocess_params['date_ranges'] = [f"{year - 5}-10-01 {year}-09-30" for year in preprocess_params['years']]
preprocess_params['sample'] = False
preprocess_params['INT_DAY'] = 15
#force_class(**base_params, **preprocess_params)


from pytorch.predict import predict_init

for basen in preprocess_params['aois']:
    basename = os.path.basename(basen)

    args_predict = {
        'model_path': '/uge_mount/FORCE/new_struc/process/result/_SITSModels/sbs_pilot/transformer/model_e35.pth',
        'folder_path': f"{base_params['temp_folder']}/{base_params['project_name']}/FORCE/{basename}/tiles_tss/X*",
        'chunksize' : 10000,
    }

    predict_init(args_predict)

    files = glob.glob(f"{base_params['temp_folder']}/{base_params['project_name']}/FORCE/{basename}/tiles_tss/X*/predicted.tif")
    output_filename = f"{base_params['proc_folder']}/{base_params['project_name']}/{os.path.basename(basen.replace('.shp','.tif'))}"

    mosaic_rasters(files, output_filename)


endzeit = time.time()
print(f"{(endzeit-startzeit)/60} minutes")
