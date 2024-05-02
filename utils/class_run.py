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
import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import datetime

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

# Function to move files
def move_files(output_folder, file_list, dest_folder):
    for file in file_list:
        shutil.move(os.path.join(output_folder, file), os.path.join(dest_folder, file))

def sample_to_ref_sepfiles(sampleref_param, preprocess_param, temp_folder, **kwargs):

    sampleref_param["project_name"] = preprocess_param["project_name"]

    bands = len(sampleref_param["band_names"])

    response_lst = sorted(glob.glob(f'{temp_folder}/{sampleref_param["project_name"]}/FORCE/*/tiles_tss/response*.txt'))
    features_lst = sorted(glob.glob(f'{temp_folder}/{sampleref_param["project_name"]}/FORCE/*/tiles_tss/features*.txt'))
    coordinates_lst = sorted(glob.glob(f'{temp_folder}/{sampleref_param["project_name"]}/FORCE/*/tiles_tss/coordinates*.txt'))

    output_folder_sep = f'{sampleref_param["output_folder"]}/sepfiles'
    print(f"Output folder does not exist ... creating {output_folder_sep}")
    os.makedirs(output_folder_sep, exist_ok=True)
    try:
        shutil.copy(f'{temp_folder}/{sampleref_param["project_name"]}/preprocess_settings.json', f'{sampleref_param["output_folder"]}/preprocess_settings.json')
    except:
        print("Couldnt Copy preprocess_settings.json")

    global_idx = 0
    nan_idx = 0
    # Process each file pair individually
    f_len = len(features_lst)

    # Initialize an empty DataFrame for storing coordinates with a global index
    coordinates_df = pd.DataFrame(columns=['global_idx', 'x', 'y', 'aoi'])
    coordinates_list = []
    for idx, (feature_file, response_file, coordinates_file) in enumerate(zip(features_lst, response_lst, coordinates_lst)):
        print(f"Processing Samples {idx+1} of {f_len}")

        folder_year = os.path.basename(os.path.dirname(os.path.dirname(feature_file)))  # Move up two levels in the directory path
        procyear_match = re.search(r'(\d{4})',folder_year)
        assert procyear_match, "Error: Year not found in folder name"
        procyear = int(procyear_match.group(1))
        start_year = procyear - int(sampleref_param["start_doy_month"][0])


        feature = pd.read_csv(feature_file, sep=' ', header=None)
        response = pd.read_csv(response_file, sep=' ', header=None)
        coordinates = pd.read_csv(coordinates_file, sep=' ', header=None, names=['x', 'y'])

        tile_folder = os.path.basename(response_file)[9:-4] # X*_Y* force tile folder
        raster_path = glob.glob(f"{os.path.dirname(response_file)}/{tile_folder}/*.tif")[0]

        #print(raster_path)
        timesteps_per_band = int(feature.shape[1] / bands)

        #with rasterio.open(raster_path) as src:
            #timesteps = [src.descriptions[i] for i in range(src.count)]
        with rasterio.open(raster_path) as src:
            timesteps = [src.descriptions[i][:8] for i in range(src.count)]


        feature = feature.replace(-9999, np.nan)

        # Calculate the total number of items for tqdm to track progress accurately
        total_items = len(feature)
        with tqdm(total=total_items, desc="Processing Rows") as pbar:
            for row_idx, (feat_row, resp_row, coord_row) in enumerate(zip(feature.iterrows(), response.iterrows(), coordinates.iterrows())):
                feat_row = feat_row[1].values
                resp_row = resp_row[1].values
                coord_row_data = coord_row[1].values

                if np.all(np.isnan(feat_row)):
                    nan_idx += 1
                    continue  # Skip the current iteration and move to the next array

                pixel_data = np.reshape(feat_row, (bands, timesteps_per_band)).T
                pixel_df = pd.DataFrame(pixel_data, columns=sampleref_param["band_names"], dtype=float)

                # Step 1: Extract year, month, day from 'doa' and calculate 'doy'
                doa_dates = [datetime.datetime.strptime(str(doa), '%Y%m%d') for doa in timesteps[:timesteps_per_band]]
                ###earliest_year = min(doa_dates, key=lambda x: x.year).year  # Step 2: Find the earliest year
                start_date = datetime.datetime.strptime(f'{start_year}{sampleref_param["start_doy_month"][1]}', '%Y%m-%d')
                #start_date = datetime.datetime.strptime(f'{2015}{sampleref_param["start_doy_month"][1]}','%Y%m-%d')
                doy = [(doa_date - start_date).days + 1 for doa_date in doa_dates]  # Step 3: Calculate 'doy'
                # New approach: Calculate doy for each date, resetting at the start of each year
                #doy = []
                # for doa_date in doa_dates:
                #     year_start_date = datetime.datetime(doa_date.year, 1,1)  # First day of the year for the current date
                #     doy_value = (doa_date - year_start_date).days + 1
                #     doy.append(doy_value)

                pixel_df.insert(0, 'year', timesteps[:timesteps_per_band])
                pixel_df.insert(1, 'doy', doy)  # Step 4: Insert 'doy' into DataFrame
                pixel_df.insert(2, 'label', resp_row[0])

                # delete timesteps with only nan values
                if sampleref_param["del_emptyTS"] == True:
                    pixel_df = pixel_df.dropna(axis=0, how='all', subset=pixel_df.columns[3:])
                else:
                    if pixel_df[sampleref_param["band_names"]].isna().any().any():
                        pixel_df[sampleref_param["band_names"]] = pixel_df[sampleref_param["band_names"]].interpolate(method='linear', limit_direction='both',axis=0)

                output_file_path = os.path.join(output_folder_sep, f"{global_idx}.csv")

                pixel_df.to_csv(output_file_path, index=False)

                temp_df = {'global_idx': global_idx, 'x': coord_row_data[0], 'y': coord_row_data[1], 'aoi': folder_year}
                coordinates_list.append(temp_df)
                global_idx += 1
                # Update the progress bar after each iteration
                pbar.update(1)

        # Ensure that your train, valid, and test folders exist
        train_folder = os.path.join(output_folder_sep, "train/csv")
        test_folder = os.path.join(output_folder_sep, "test/csv")

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        # Getting list of all .csv files in the output_folder
        csv_files = [f for f in os.listdir(output_folder_sep) if f.endswith(".csv")]

        if sampleref_param["split_train"] <= 1:
            # Shuffle the list to ensure random distribution of files
            random.seed(sampleref_param["seed"])  # Set the seed before shuffling
            random.shuffle(csv_files)
            # Calculating split indices
            num_files = len(csv_files)
            train_idx = int(num_files * sampleref_param["split_train"])
            # Splitting files
            train_files = csv_files[:train_idx]
            test_files = csv_files[train_idx:]
            # Moving files
            move_files(output_folder_sep, train_files, train_folder)
            # move_files(valid_files, valid_folder)
            move_files(output_folder_sep, test_files, test_folder)
        else:
            print(sampleref_param["split_train"])
            if procyear == sampleref_param["split_train"]:
                move_files(output_folder_sep, csv_files, test_folder)
            else:
                move_files(output_folder_sep, csv_files, train_folder)

    temp_df = pd.DataFrame(coordinates_list)
    temp_df.to_csv(os.path.join(sampleref_param["output_folder"], f"meta.csv"), index=False)
    print(f"Process finished - deleted {nan_idx} samples cause their were no values.")




