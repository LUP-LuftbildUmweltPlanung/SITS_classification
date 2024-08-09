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
from force.force_class_utils import force_class

def force_sample(preprocess_params):

    if preprocess_params["years"] == None:
        preprocess_params["years"] = [int(re.search(r'(\d{4})', os.path.basename(f)).group(1)) for f in preprocess_params["aois"] if re.search(r'(\d{4})', os.path.basename(f))]
    time_range = preprocess_params["time_range"]
    preprocess_params["date_ranges"] = [f"{year - int(time_range[0])}-{time_range[1]} {year}-{time_range[1]}" for year in preprocess_params["years"]]

    force_class(preprocess_params)
    sample_to_ref_sepfiles(preprocess_params) # splits for single domain then goes to next

# Function to move files
def move_files(output_folder, file_list, dest_folder):
    for file in file_list:
        shutil.move(os.path.join(output_folder, file), os.path.join(dest_folder, file))

def sample_to_ref_sepfiles(preprocess_params, **kwargs):

    output_folder = f'{preprocess_params["process_folder"]}/results/_SITSrefdata/{preprocess_params["project_name"]}'
    temp_folder = preprocess_params['process_folder'] + "/temp"
    preprocess_params["project_name"] = preprocess_params["project_name"]

    bands = len(preprocess_params["feature_order"])

    output_folder_sep = f'{output_folder}/sepfiles'
    print(f"Output folder does not exist ... creating {output_folder_sep}")
    os.makedirs(output_folder_sep, exist_ok=True)
    try:
        shutil.copy(f'{temp_folder}/{preprocess_params["project_name"]}/preprocess_settings.json', f'{output_folder}/preprocess_settings.json')
    except:
        print("Couldnt Copy preprocess_settings.json")

    # Initialize an empty DataFrame for storing coordinates
    coordinates_list = []
    global_idx = 0
    nan_idx = 0
    # Process each file pair individually

    # Get the list of all FORCE directories
    force_dirs = sorted(glob.glob(f'{temp_folder}/{preprocess_params["project_name"]}/FORCE/*'))

    # Extract just the filenames from preprocess_params["points"]
    point_filenames = [os.path.basename(point) for point in preprocess_params["aois"]]

    # Create a mapping from points to years
    points_years_mapping = dict(zip(point_filenames, preprocess_params["years"]))
    for force_dir in force_dirs:
        # Extract the folder name (basename)
        folder_name = os.path.basename(force_dir)

        if folder_name not in points_years_mapping:
            raise ValueError(f"Folder name '{folder_name}' does not have a corresponding year in preprocess_params.")

        # Get the corresponding year for the folder
        related_year = points_years_mapping[folder_name]
        # Construct the pattern to match the specific files within each FORCE directory
        response_lst = sorted(glob.glob(os.path.join(force_dir, 'tiles_tss/response*.txt')))
        features_lst = sorted(glob.glob(os.path.join(force_dir, 'tiles_tss/features*.txt')))
        coordinates_lst = sorted(glob.glob(os.path.join(force_dir, 'tiles_tss/coordinates*.txt')))


        f_len = len(features_lst)


        for idx, (feature_file, response_file, coordinates_file) in enumerate(zip(features_lst, response_lst, coordinates_lst)):
            print(f"Processing Samples {idx+1} of {f_len}")

            feature = pd.read_csv(feature_file, sep=' ', header=None)
            response = pd.read_csv(response_file, sep=' ', header=None)
            coordinates = pd.read_csv(coordinates_file, sep=' ', header=None, names=['x', 'y'])

            tile_folder = os.path.basename(response_file)[9:-4] # X*_Y* force tile folder
            raster_path = glob.glob(f"{os.path.dirname(response_file)}/{tile_folder}/*.tif")[0]

            #print(raster_path)
            timesteps_per_band = int(feature.shape[1] / bands)


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
                    pixel_df = pd.DataFrame(pixel_data, columns=preprocess_params["feature_order"], dtype=float)

                    # Step 1: Extract year, month, day from 'doa' and calculate 'doy'
                    doa_dates = [datetime.datetime.strptime(str(doa), '%Y%m%d') for doa in timesteps[:timesteps_per_band]]
                    ###earliest_year = min(doa_dates, key=lambda x: x.year).year  # Step 2: Find the earliest year


                    if preprocess_params["start_doy_month"] == None:
                        start_year = related_year - int(preprocess_params["time_range"][0])
                        start_date = datetime.datetime.strptime(f'{start_year}{preprocess_params["time_range"][1]}', '%Y%m-%d')
                    else:
                        start_date = datetime.datetime.strptime(f'{preprocess_params["start_doy_month"][0]}','%Y-%m-%d')

                    doy = [(doa_date - start_date).days + 1 for doa_date in doa_dates]  # Step 3: Calculate 'doy'

                    pixel_df.insert(0, 'year', timesteps[:timesteps_per_band])
                    pixel_df.insert(1, 'doy', doy)  # Step 4: Insert 'doy' into DataFrame
                    pixel_df.insert(2, 'label', resp_row[0])

                    # delete timesteps with only nan values
                    if preprocess_params["Interpolation"] == False:
                        pixel_df = pixel_df.dropna(axis=0, how='all', subset=pixel_df.columns[3:])
                    else:
                        if pixel_df[preprocess_params["feature_order"]].isna().any().any():
                            pixel_df[preprocess_params["feature_order"]] = pixel_df[preprocess_params["feature_order"]].interpolate(method='linear', limit_direction='both',axis=0)

                    output_file_path = os.path.join(output_folder_sep, f"{global_idx}.csv")

                    pixel_df.to_csv(output_file_path, index=False)

                    temp_df = {'global_idx': global_idx, 'x': coord_row_data[0], 'y': coord_row_data[1], 'aoi': folder_name}
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

            if preprocess_params["split_train"] <= 1:
                # Shuffle the list to ensure random distribution of files
                random.seed(preprocess_params["seed"])  # Set the seed before shuffling
                random.shuffle(csv_files)
                # Calculating split indices
                num_files = len(csv_files)
                train_idx = int(num_files * preprocess_params["split_train"])
                # Splitting files
                train_files = csv_files[:train_idx]
                test_files = csv_files[train_idx:]
                # Moving files
                move_files(output_folder_sep, train_files, train_folder)
                # move_files(valid_files, valid_folder)
                move_files(output_folder_sep, test_files, test_folder)
            else:
                print(preprocess_params["split_train"])
                if related_year == preprocess_params["split_train"]:
                    move_files(output_folder_sep, csv_files, test_folder)
                else:
                    move_files(output_folder_sep, csv_files, train_folder)

    temp_df = pd.DataFrame(coordinates_list)
    temp_df.to_csv(os.path.join(output_folder, f"meta.csv"), index=False)
    print(f"Process finished - deleted {nan_idx} samples cause their were no values.")




