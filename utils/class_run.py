# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:24:28 2023

@author: Admin
"""

import random
import rasterio
import rasterio.mask
import glob
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import shapely
import os
#shapely.speedups.disable()
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from rasterio.merge import merge
import shutil
from tqdm import tqdm
import time

def sample_to_ref_onefile(force_dir,local_dir,force_skel,scripts_skel,temp_folder,mask_folder,
 proc_folder,data_folder,project_name,hold,response_lst,features_lst,response_out,features_out,bands,split_train):
    o_folder = f"{os.path.dirname(response_out)}/onefile"
    if not os.path.exists(o_folder):
        print(f"output folder doesnt exist ... creating {o_folder}")
        os.makedirs(o_folder)
    # Merge txt files for features
    df_features = pd.concat([pd.read_csv(f, sep=' ', header=None) for f in features_lst])
    # Merge txt files for response
    df_response = pd.concat([pd.read_csv(f, sep=' ', header=None) for f in response_lst])

    features_out = f"{o_folder}/{os.path.basename(features_out)}"
    response_out = f"{o_folder}/{os.path.basename(response_out)}"
    features = df_features.replace(-9999, np.nan)
    df_features.to_csv(features_out, sep=' ', header=False, index=False)
    df_response.to_csv(response_out, sep=' ', header=False, index=False)
    #df_coords.to_csv(coords_out, sep=' ', header=False, index=False)
    ##interpolate for every band of every point
    band_length = features.shape[1] // bands
    for i in range(bands):
        start = i * band_length
        end = start + band_length
        band = features.iloc[:, start:end]
        band_interpolated = band.interpolate(axis=1, limit_direction='both')
        features.iloc[:, start:end] = band_interpolated
        # print(band)

    # Add row ID as a new column
    features['row_id'] = features.index

    # Concatenate response and interpolated_features DataFrames
    result = pd.concat([df_response, features['row_id'], features.drop(columns=['row_id'])], axis=1)
    result.columns = range(result.shape[1])
    result = result.loc[result[0] != -9999]

    result.columns = result.columns.astype(str)

    # Define the train percentage
    train_perc = split_train

    train_result, test_result = train_test_split(result, train_size=train_perc, random_state=42)
    result = None

    train_result = train_result.dropna()
    test_result = test_result.dropna()

    train_output = features_out.replace(".txt", "_train.csv")
    test_output = features_out.replace(".txt", "_test.csv")

    train_result.to_csv(train_output, index=False, header=False, sep=",")
    test_result.to_csv(test_output, index=False, header=False, sep=",")

def sample_to_ref_sepfiles(
        force_dir, local_dir, force_skel, scripts_skel, temp_folder, mask_folder,
        proc_folder, data_folder, project_name, hold, output_folder, seed, bands, split_train):

    response_lst = glob.glob(f"{temp_folder}/{project_name}/FORCE/*/tiles_tss/response.txt")
    features_lst = glob.glob(f"{temp_folder}/{project_name}/FORCE/*/tiles_tss/features.txt")

    output_folder = f"{output_folder}/sepfiles"
    if not os.path.exists(output_folder):
        print(f"Output folder does not exist ... creating {output_folder}")
        os.makedirs(output_folder)
    global_idx = 0
    # Process each file pair individually
    f_len = len(features_lst)

    for idx, (feature_file, response_file) in enumerate(zip(features_lst, response_lst)):
        print(f"Processing Samples {idx+1} of {f_len}")
        feature = pd.read_csv(feature_file, sep=' ', header=None)
        response = pd.read_csv(response_file, sep=' ', header=None)

        raster_path = glob.glob(f"{os.path.dirname(response_file)}/X*/*.tif")[0]
        #print(raster_path)
        timesteps_per_band = int(feature.shape[1] / bands)

        with rasterio.open(raster_path) as src:
            timesteps = [src.descriptions[i] for i in range(src.count)]

        band_names = ["B2", "B3", "B4", "B8", "B11", "B12", "B5", "B6", "B7", "B8A"]

        feature = feature.replace(-9999, np.nan)

        # Calculate the total number of items for tqdm to track progress accurately
        total_items = len(feature)
        with tqdm(total=total_items, desc="Processing Rows") as pbar:
            for idx, (feat_row, resp_row) in enumerate(zip(feature.iterrows(), response.iterrows())):
                feat_row = feat_row[1].values
                resp_row = resp_row[1].values

                pixel_data = np.reshape(feat_row, (bands, timesteps_per_band)).T
                pixel_df = pd.DataFrame(pixel_data, columns=band_names, dtype=float)
                pixel_df.insert(0, 'label', resp_row[0])
                pixel_df.insert(0, 'doa', timesteps[:timesteps_per_band])
                pixel_df.insert(0, 'id', idx)

                if pixel_df[band_names].isna().any().any():
                    pixel_df[band_names] = pixel_df[band_names].interpolate(method='linear', limit_direction='both', axis=0)

                output_file_path = os.path.join(output_folder, f"{global_idx}.csv")
                pixel_df.to_csv(output_file_path, index=False)
                global_idx += 1
                # Update the progress bar after each iteration
                pbar.update(1)
        #Seed for reproducibility


        # Ensure that your train, valid, and test folders exist
        train_folder = os.path.join(output_folder, "train/csv")
        # valid_folder = os.path.join(output_folder, "valid")
        test_folder = os.path.join(output_folder, "test/csv")

        os.makedirs(train_folder, exist_ok=True)
        # os.makedirs(valid_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        # Getting list of all .csv files in the output_folder
        csv_files = [f for f in os.listdir(output_folder) if f.endswith(".csv")]

        # Shuffle the list to ensure random distribution of files
        random.seed(seed)  # Set the seed before shuffling
        random.shuffle(csv_files)

        # Calculating split indices
        num_files = len(csv_files)
        train_idx = int(num_files * split_train)
        # valid_idx = int(num_files * (train_perc + valid_perc))

        # Splitting files
        train_files = csv_files[:train_idx]
        # valid_files = csv_files[train_idx:valid_idx]
        # test_files = csv_files[valid_idx:]
        test_files = csv_files[train_idx:]

        # Function to move files
        def move_files(file_list, dest_folder):
            for file in file_list:
                shutil.move(os.path.join(output_folder, file), os.path.join(dest_folder, file))

        # Moving files
        move_files(train_files, train_folder)
        # move_files(valid_files, valid_folder)
        move_files(test_files, test_folder)

def mosaic_rasters(input_pattern, output_filename):
    """
    Mosaic rasters matching the input pattern and save to output_filename.

    Parameters:
    - input_pattern: str, a wildcard pattern to match input raster files (e.g., "./tiles/*.tif").
    - output_filename: str, the name of the output mosaic raster file.
    """

    # Find all files matching the pattern
    src_files_to_mosaic = [rasterio.open(fp) for fp in input_pattern]

    # Mosaic the rasters
    mosaic, out_transform = merge(src_files_to_mosaic)
    #mosaic[mosaic == 0] = -9999
    # Get metadata from one of the input files
    out_meta = src_files_to_mosaic[0].meta.copy()

    # Update metadata with new dimensions, transform, and compression (optional)
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        #"compress": "lzw"
    })
    if not os.path.exists(os.path.dirname(output_filename)):
        print(f"output folder doesnt exist ... creating {os.path.dirname(output_filename)}")
        os.makedirs(os.path.dirname(output_filename))
    # Write the mosaic raster to disk
    with rasterio.open(output_filename, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Close the input files
    for src in src_files_to_mosaic:
        src.close()


