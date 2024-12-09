# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:30:26 2023

@author: benjaminstoeckigt
"""

from pytorch.train import getModel
from pytorch.utils.hw_monitor import HWMonitor, disk_info, squeeze_hw_info
from force.force_class_utils_inference import force_class, force_class_pre
import subprocess
import time
import os
import torch
import rasterio
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
import glob
import json
import datetime
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask

def predict(args_predict):

    preprocess_params = load_preprocess_settings(os.path.dirname(args_predict["model_path"]))
    preprocess_params["aois"] = args_predict["aois"]

    assert (preprocess_params["thermal_time"] is None and args_predict["thermal_time_prediction"] is None) or \
           (preprocess_params["thermal_time"] is not None and args_predict["thermal_time_prediction"] is not None), "Different Positional Encoding used for Training and Prediction"
    if args_predict["thermal_time_prediction"] is not None:
        print("Predicting with Thermal Time!")
    if args_predict["years"] == None:
        preprocess_params["years"] = [int(re.search(r'(\d{4})', os.path.basename(f)).group(1)) for f in preprocess_params["aois"] if re.search(r'(\d{4})', os.path.basename(f))]
    else:
        preprocess_params["years"] = args_predict["years"]

    time_range = preprocess_params["time_range"]
    preprocess_params["date_ranges"] = [f"{year - int(time_range[0])}-{time_range[1]} {year}-{time_range[1]}" for year in preprocess_params["years"]]
    preprocess_params["hold"] = False
    preprocess_params["sample"] = False
    preprocess_params["project_name"] = args_predict["project_name"]
    preprocess_params["force_dir"] = args_predict["force_dir"]
    preprocess_params["process_folder"] = args_predict["process_folder"]

    args_predict["time_range"] = preprocess_params["time_range"]
    args_predict["feature_order"] = preprocess_params["feature_order"]

    scripts_skel = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/force/skel"
    temp_folder = preprocess_params['process_folder'] + "/temp"
    project_name = preprocess_params["project_name"]
    proc_folder = args_predict['process_folder'] + "/results"
    hyp = load_hyperparametersplus(os.path.dirname(args_predict["model_path"]))
    args_predict.update(hyp)

    ###LOADING MODEL
    if args_predict["thermal_time_prediction"] is not None and args_predict["thermal_time"] is not None:
        print("\nApplying Transformer Model with Thermal Positional Encoding!")
    else:
        print("\nApplying Transformer Model with Calendar Positional Encoding!")
    args_predict['store'] = os.path.dirname(args_predict['model_path'])
    model_path = args_predict['model_path']
    model = load_model(model_path,args_predict)

    #preprocess_params["date_ranges"] = ['2015-01-01 2024-12-31']
    ###save preprocessing settings for prediction
    os.makedirs(f'{temp_folder}/{project_name}/FORCE', exist_ok=True)
    # List of keys you want to save
    keys_to_save = ["time_range","Interpolation","INT_DAY","Sensors","Indices","SPECTRAL_ADJUST","INTERPOLATE","ABOVE_NOISE","BELOW_NOISE","NTHREAD_READ","NTHREAD_COMPUTE","NTHREAD_WRITE","BLOCK_SIZE","band_names","start_doy_month","feature_order","thermal_time"]  # replace these with the actual keys you want to save
    # Create a new dictionary with only the specified keys
    filtered_params = {key: preprocess_params[key] for key in keys_to_save if key in preprocess_params}
    # Save the filtered dictionary to the JSON file
    with open(f"{temp_folder}/{project_name}/preprocess_settings.json", 'w') as file:
        json.dump(filtered_params, file, indent=4)

    #subprocess.run(['sudo', 'chmod', '-R', '777', f"{Path(temp_folder).parent}"])
    subprocess.run(['sudo', 'chmod', '-R', '777', f"{Path(scripts_skel).parent}"])
    startzeit = time.time()
    if not args_predict['reference_folder']:
        for aoi,DATE_RANGE in zip(preprocess_params["aois"], preprocess_params["date_ranges"]):
            print(f"INFERENCE FOR {aoi} WITHIN TIME RANGE {DATE_RANGE}")

            X_TILES, Y_TILES = force_class_pre(preprocess_params, aoi)

            for idx, (X_TILE, Y_TILE) in enumerate(tqdm(zip(X_TILES, Y_TILES), total=len(X_TILES), desc="Overall Progress (Processing FORCE Tiles)", position=0, leave=True)):

                last_iteration = (idx == len(X_TILES) - 1)
                force_class(preprocess_params, aoi, DATE_RANGE, X_TILE, Y_TILE)

                if last_iteration:
                    # create hw_monitor output dir if it doesn't exist
                    drive_name = ["sdb1"]
                    Path(os.path.dirname(args_predict['model_path']) + '/hw_monitor').mkdir(parents=True, exist_ok=True)
                    hw_predict_logs_file = os.path.dirname(args_predict['model_path']) + '/hw_monitor/hw_monitor_predict.csv'
                    # Instantiate monitor with a 1-second delay between updates
                    hwmon_p = HWMonitor(1, hw_predict_logs_file, drive_name)
                    hwmon_p.start()
                    hwmon_p.start_averaging()


                basename = os.path.basename(aoi)
                args_predict['folder_path'] = f"{temp_folder}/{args_predict['project_name']}/FORCE/{basename}/tiles_tss/X00{X_TILE}_Y00{Y_TILE}"

                tile = args_predict['folder_path']
                prediction = predict_singlegrid(model, tile, args_predict)
                reshape_and_save(prediction, tile, args_predict)

                if args_predict["tmp_cleanup"] == True:
                    for f in os.listdir(args_predict['folder_path']):
                        if f != "predicted.tif":
                            os.remove(os.path.join(args_predict['folder_path'], f))

                if last_iteration:
                    hwmon_p.stop_averaging()
                    avgs = hwmon_p.get_averages()
                    squeezed = squeeze_hw_info(avgs)
                    mean_data = {key: round(value, 1) for key, value in squeezed.items() if "mean" in key}
                    hwmon_p.stop()

            print(f"##################\nMean Values Hardware Monitoring (Last Tile):\n{mean_data}\n##################")
            files = glob.glob(f"{temp_folder}/{args_predict['project_name']}/FORCE/{basename}/tiles_tss/X*/predicted.tif")
            output_filename = f"{proc_folder}/{args_predict['project_name']}/{os.path.basename(aoi.replace('.shp','.tif'))}"
            mosaic_rasters(files, output_filename)
    else:
        predict_csv(args_predict)
    endzeit = time.time()
    print("Processing beendet nach "+str((endzeit-startzeit)/60)+" Minuten")


def end_padding(batch_tensor, doy_tensor, thermal_tensor=None):
    ###################################################################################
    ###################################################################################
    # Get a mask of non-zero values across the features for each time step
    non_zero_mask_pad = torch.any(batch_tensor != 0, dim=1)  # Shape: [batch_size, seq_len]
    # Convert the boolean mask to integers for sorting (True -> 1, False -> 0)
    # Count the number of non-zero time steps for each sample
    non_zero_counts = non_zero_mask_pad.sum(dim=1)  # Shape: [batch_size]
    # Get indices that sort non-zero time steps to the front
    _, indices = torch.sort(non_zero_mask_pad.int(), descending=True, dim=1)  # Shape: [batch_size, seq_len]
    # Expand indices to match the dimensions of batch_tensor for gathering
    indices_expanded = indices.unsqueeze(1).expand(-1, batch_tensor.size(1),
                                                   -1)  # Shape: [batch_size, features, seq_len]
    # Rearrange batch_tensor to move zeros to the end along the sequence dimension
    batch_tensor = torch.gather(batch_tensor, dim=2, index=indices_expanded)
    # Rearrange doy_tensor and thermal_tensor similarly
    doy_tensor = torch.gather(doy_tensor, dim=1, index=indices)
    # Now, set the padding positions in doy_tensor and thermal_tensor to zeros
    batch_size, seq_len = doy_tensor.shape
    device = doy_tensor.device
    # Create a tensor of sequence indices
    seq_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size,
                                                                           -1)  # Shape: [batch_size, seq_len]
    # Expand non_zero_counts to match seq_len
    non_zero_counts_expanded = non_zero_counts.unsqueeze(1).expand(-1, seq_len)  # Shape: [batch_size, seq_len]
    # Create a mask where positions beyond non_zero_counts are padding positions
    padding_mask = seq_indices >= non_zero_counts_expanded  # Shape: [batch_size, seq_len]
    # Set the padding positions in doy_tensor and thermal_tensor to zeros
    doy_tensor[padding_mask] = 0  # Use your desired padding value if different
    if thermal_tensor is not None:
        thermal_tensor = torch.gather(thermal_tensor, dim=1, index=indices)
        thermal_tensor[padding_mask] = 0  # Use your desired padding value if different
    ###################################################################################
    ###################################################################################
    # Remove time steps where all features are zero
    # Requirement Sample Size 1
    # non_zero_mask = torch.any(batch_tensor != 0, dim=1)  # Shape: [1, seq_len]
    # non_zero_mask = non_zero_mask.squeeze(0)  # Shape: [seq_len]
    # batch_tensor = batch_tensor[:, :, non_zero_mask]
    # doy_tensor = doy_tensor[:, non_zero_mask]
    # if thermal_tensor is not None:
    #     thermal_tensor = thermal_tensor[:, non_zero_mask]
    return batch_tensor, doy_tensor, thermal_tensor


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


def load_model(model_path,args):

    # Load a PyTorch model from the given path
    saved_state = torch.load(model_path)
    model_state_dict = saved_state["model_state"]
    args['nclasses'] = saved_state["nclasses"]
    args['seqlength'] = args['max_seq_length']
    args['input_dims'] = saved_state["ndims"]
    #print(f"Sequence Length: {args['seqlength']}")
    print(f"Input Dims: {args['input_dims']}")
    print(f"Prediction Classes: {args['nclasses']}")
    model = getModel(args)

    model.load_state_dict(model_state_dict)

    model.eval()  # Set the model to evaluation mode
    return model

# Function to match and load necessary thermal bands based on doa_dates, spatial extent, and resample them
def load_and_resample_thermal_data(thermal_file_path, bands_data_extent, bands_data_res, doa_dates):
    """
    Matches necessary thermal bands based on doa_dates, loads and resamples thermal data to match the spatial
    extent and resolution of the bands_data.

    Parameters:
    - thermal_file_path (str): Path to the thermal raster dataset.
    - bands_data_extent (tuple): The spatial extent (min_x, min_y, max_x, max_y) of the bands_data.
    - bands_data_res (float): The spatial resolution of the bands_data.
    - doa_dates (list): List of acquisition dates in '%Y%m%d' format for the bands_data.

    Returns:
    - resampled_thermal_data (numpy.ndarray): The resampled thermal data that matches bands_data's extent and resolution.
    - resampled_transform (Affine): The affine transform of the resampled thermal data.
    """
    from shapely.geometry import box
    # Convert doa_dates into a set for quick lookup
    required_dates = set([datetime.datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d') for date in doa_dates])

    with rasterio.open(thermal_file_path) as thermal_dataset:
        # Get the band descriptions and map them to their indices
        band_to_date = {i + 1: thermal_dataset.descriptions[i] for i in range(thermal_dataset.count)}

        # Find the bands that match the required doa_dates
        matched_bands = [band for band, date in band_to_date.items() if date in required_dates]

        if not matched_bands:
            raise ValueError("No matching bands found in the thermal dataset for the provided DOA dates.")

        # Check the nodata value of the thermal dataset
        nodata_value = thermal_dataset.nodata

        # Convert the bounding box (extent) to a Shapely geometry polygon
        min_x, min_y, max_x, max_y = bands_data_extent
        geometry = [box(min_x, min_y, max_x, max_y)]  # Create a polygon from the bounding box

        thermal_data, thermal_transform = mask(thermal_dataset, geometry, crop=True, indexes=matched_bands, all_touched=True, nodata=nodata_value)

        # Define the new shape for the resampled data based on the bands_data resolution
        resampled_height = int((bands_data_extent[3] - bands_data_extent[1]) / bands_data_res)
        resampled_width = int((bands_data_extent[2] - bands_data_extent[0]) / bands_data_res)

        # Create an empty array to hold the resampled thermal data
        resampled_thermal_data = np.empty((len(matched_bands), resampled_height, resampled_width), dtype=np.float32)
        # Resample the cropped thermal data to match the resolution of bands_data
        reproject(
            source=thermal_data,
            destination=resampled_thermal_data,
            src_transform=thermal_transform,
            src_crs=thermal_dataset.crs,
            dst_transform=rasterio.transform.from_bounds(*bands_data_extent, resampled_width, resampled_height),
            dst_crs=thermal_dataset.crs,
            resampling=Resampling.nearest,
            src_nodata=nodata_value,  # Specify the nodata value to ignore it during reprojection
            dst_nodata=nodata_value   # Set the destination nodata value to the same, or None if no nodata
        )

    return resampled_thermal_data, rasterio.transform.from_bounds(*bands_data_extent, resampled_width,
                                                                  resampled_height)
def read_tif_files(folder_path, order, year, month, day, thermal_dataset=None):
    bands_data = []
    timesteps = []
    #print(folder_path)
    for band in order:
        # Find the file that matches the pattern
        file_pattern = os.path.join(folder_path, f"*{band}_*.tif")
        file_list = glob.glob(file_pattern)

        if len(file_list) == 0:
            raise FileNotFoundError(f"No file found for pattern {file_pattern}")

        # Assuming there's only one file per pattern
        file_path = file_list[0]
        with rasterio.open(file_path) as src:
            # Read all bands from the TIFF file
            band_data = src.read()
            band_data[band_data == -9999] = 0
            bands_data.append(band_data)

            # Capture transform and resolution for bands_data
            bands_data_transform = src.transform
            bands_data_res = src.res[0]  # Assuming square pixels, use the first value for resolution

    with rasterio.open(file_path) as src:
        timestamp = src.descriptions  # Assuming you need the first description for DOY

    doa_dates = [datetime.datetime.strptime(str(doa[:8]), '%Y%m%d') for doa in timestamp]
    # New DOY calculation that resets at the beginning of each year
    # doy = []
    # for doa_date in doa_dates:
    #     year_start_date = datetime.datetime(doa_date.year, 1, 1)  # First day of the year for the current date
    #     doy_value = (doa_date - year_start_date).days + 1
    #     #print(doy_value)
    #     doy.append(doy_value)

    latest_year = max(doa_dates, key=lambda x: x.year).year
    start_date = datetime.datetime(latest_year-year, month, day)
    doy = [(doa_date - start_date).days + 1 for doa_date in doa_dates]

    # Calculate the spatial extent of bands_data using the transform
    height, width = bands_data[0][0].shape  # Assuming consistent shape for all bands
    min_x, min_y = rasterio.transform.xy(bands_data_transform, height, 0)  # Lower-left corner
    max_x, max_y = rasterio.transform.xy(bands_data_transform, 0, width)  # Upper-right corner
    bands_data_extent = (min_x, min_y, max_x, max_y)


    # If thermal_file_path is provided, extract thermal data for each pixel and timestep
    if thermal_dataset is not None:
        # Load and resample the thermal data to match the spatial extent and resolution of bands_data
        resampled_thermal_data, resampled_transform = load_and_resample_thermal_data(
            thermal_dataset, bands_data_extent, bands_data_res, doa_dates
        )

        # Initialize the thermal_grid to hold the thermal values [timesteps][height][width]
        thermal_grid = np.zeros((len(doy), height, width), dtype=np.float32)

        # Loop over each timestep and use the resampled thermal data directly
        for t in range(len(doy)):
            #print(len(doy))
            #print(f"Processing timestep {t + 1}/{len(doy)}")
            #print(resampled_thermal_data.shape)
            # Directly assign the resampled thermal data to the thermal_grid since it's aligned
            thermal_grid[t] = resampled_thermal_data[t]

    return bands_data, doy, thermal_grid if thermal_dataset is not None else None


def predict_singlegrid(model, tiles, args_predict):
    #print(f"Preprocessing the Data for Prediction ...")
    # Read TIFF files
    order = args_predict["feature_order"]
    normalizing_factor = args_predict["norm_factor_features"]
    chunksize = args_predict["chunksize"]
    year = int(args_predict['time_range'][0])
    month = int(args_predict['time_range'][1].split('-')[0])
    day = int(args_predict['time_range'][1].split('-')[1])
    thermal_dataset = args_predict["thermal_time_prediction"]
    if thermal_dataset is not None:
        data, doy_single, thermal = read_tif_files(tiles, order, year, month, day, thermal_dataset)
        data_thermal = np.stack(thermal, axis=0)
        data = np.stack(data, axis=1)
        # Reshape data for PyTorch [sequence length, number of bands, height, width]
        seq_len, num_bands, height, width = data.shape
        XY = height * width
        data_thermal = data_thermal.reshape(seq_len, XY)
        data_thermal = np.transpose(data_thermal, (1, 0))
    else:
        # Stack the bands and perform initial reshape in NumPy
        data, doy_single, thermal = read_tif_files(tiles, order, year, month, day, thermal_dataset)
        data = np.stack(data, axis=1)
        # Reshape data for PyTorch [sequence length, number of bands, height, width]
        seq_len, num_bands, height, width = data.shape
        XY = height * width
    data = data.reshape(seq_len, num_bands, XY)
    # Reorder dimensions for PyTorch [XY, number of bands, sequence length] using NumPy
    data = np.transpose(data, (2, 1, 0))

    # Move model to the appropriate device
    device = next(model.parameters()).device

    predictions = []
    #print(f"Predicting with Chunksize {chunksize} from {data.shape[0]}")
    with torch.no_grad():
        for i in tqdm(range(0, data.shape[0], chunksize), desc=f"Predicting FORCE Tile with Chunksize {chunksize}", position=0, leave=False):
            batch = data[i:i + chunksize] * normalizing_factor
            if thermal_dataset is not None:
                batch_thermal = data_thermal[i:i + chunksize]
            non_zero_mask = np.any(batch != 0, axis=(1, 2))
            doy = np.array(doy_single)
            doy = np.tile(doy, (batch.shape[0], 1))
            cls = 1
            if args_predict["response"] == "classification":
                cls = len(args_predict["classes_lst"])
            if not np.any(non_zero_mask):
                batch_predictions = torch.full((batch.shape[0], cls), -9999, dtype=torch.float32, device=device)
            else:
                batch_non_zero = batch[non_zero_mask]
                doy_non_zero = doy[non_zero_mask]
                batch_tensor = torch.tensor(batch_non_zero, dtype=torch.float32, device=device)
                doy_tensor = torch.tensor(doy_non_zero, dtype=torch.long, device=device)
                if thermal_dataset is not None:
                    batch_thermal = batch_thermal[non_zero_mask]
                    thermal_tensor = torch.tensor(batch_thermal, dtype=torch.long, device=device)
                    batch_tensor, doy_tensor, thermal_tensor = end_padding(batch_tensor, doy_tensor, thermal_tensor)
                    predictions_non_zero = model(batch_tensor, doy_tensor, thermal_tensor)[0]
                else:
                    batch_tensor, doy_tensor, _ = end_padding(batch_tensor, doy_tensor)
                    predictions_non_zero = model(batch_tensor, doy_tensor,thermal = None)[0]

                # Handle normalization response factor
                norm_factor_response = args_predict.get("norm_factor_response")
                if norm_factor_response == "log10":
                    predictions_non_zero = torch.pow(10, predictions_non_zero) - 1
                elif norm_factor_response is not None and norm_factor_response != 0:
                    predictions_non_zero = predictions_non_zero / norm_factor_response

                if args_predict["response"] == "classification" and not args_predict["probability"]:
                    predictions_non_zero = torch.argmax(predictions_non_zero, dim=1).unsqueeze(1)  # Ensure 2D

                batch_predictions = torch.full((batch.shape[0], *predictions_non_zero.shape[1:]), -9999, dtype=torch.float32, device=device)
                batch_predictions[non_zero_mask] = predictions_non_zero

            predictions.append(batch_predictions.cpu())  # Move predictions back to CPU if needed

    return torch.cat(predictions, dim=0)


def read_tif_file(file_path):
    with rasterio.open(file_path) as src:
        # Read the specific band data from the TIFF file
        band_data = src.read()
        band_data[band_data == -9999] = 0
        return band_data


def reshape_and_save(predictions, tiles, args_predict):
    # Reshape the predictions to 3000x3000
    num_bands = predictions.shape[1]
    if num_bands >1:
        reshaped_predictions = predictions.reshape(3000, 3000, num_bands).cpu().numpy()
        reshaped_predictions = reshaped_predictions.transpose((2, 0, 1))
    else:
        reshaped_predictions = predictions.reshape(3000, 3000).cpu().numpy()

    # Find one of the existing TIFF files to copy the metadata
    existing_tif_path = glob.glob(os.path.join(tiles, "*.tif"))[0]
    if not existing_tif_path:
        raise FileNotFoundError("No TIFF files found in the folder to copy metadata.")


    # Read the existing TIFF file to get metadata
    with rasterio.open(existing_tif_path) as src:
        metadata = src.meta

    if args_predict["response"] == "classification" and not args_predict["probability"]:
        reshaped_predictions = reshaped_predictions.astype('uint8')
        metadata.update(dtype=rasterio.uint8, nodata=255, count=1)
    elif args_predict["response"] == "classification" and args_predict["probability"]:
        metadata.update(dtype=rasterio.float32, count=num_bands)
    else:
        metadata.update(dtype=rasterio.float32, count=1)

    # Write the predictions to a new TIFF file
    output_path = os.path.join(tiles, "predicted.tif")
    with rasterio.open(output_path, 'w', **metadata) as dst:
        if num_bands == 1:
            # Write a single band
            dst.write(reshaped_predictions, 1)
        else:
            # Iterate over each band and write
            for band in range(num_bands):
                dst.write(reshaped_predictions[band], band + 1)

def load_hyperparametersplus(model_name):
    """
    Load the hyperparameters from a JSON file.
    """
    file_path = os.path.join(model_name, "hyperparameters.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No hyperparameters file found at {file_path}")
    with open(file_path, 'r') as file:
        hyperparameters = json.load(file)
    return hyperparameters

def load_preprocess_settings(model_name):
    """
    Load the hyperparameters from a JSON file.
    """
    file_path = os.path.join(model_name, "preprocess_settings.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No hyperparameters file found at {file_path}")
    with open(file_path, 'r') as file:
        hyperparameters = json.load(file)
    return hyperparameters


def predict_csv(args_predict):

    output_path = args_predict["reference_folder"]
    reference_folder = args_predict["reference_folder"]
    folder_path = reference_folder+"/sepfiles/test/csv"
    meta_path = reference_folder+"/meta.csv"

    hyp = load_hyperparametersplus(os.path.dirname(args_predict["model_path"]))
    args_predict.update(hyp)
    model_path = args_predict['model_path']
    order = args_predict["feature_order"]
    model = load_model(model_path, args_predict)

    # Load metadata
    metadata_df = pd.read_csv(meta_path)

    # Prepare a DataFrame to hold all predictions
    predictions_df = pd.DataFrame(columns=['x', 'y', 'label', 'prediction','aoi'])

    # Iterate over CSV files
    for idx, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0]):
    #for idx, row in metadata_df.iterrows():
        #if (row['aoi'] != "duisburg_2022_extract.shp") and (row['aoi'] != "essen_2022_extract.shp"):
            #continue

        csv_file_path = os.path.join(folder_path, f"{row['global_idx']}.csv")
        if not os.path.exists(csv_file_path):
            continue  # Skip if file doesn't exist

        # Read spectral data
        pixel_df = pd.read_csv(csv_file_path)

        # Extract relevant columns and normalize if necessary
        data = pixel_df[order].values
        doy = pixel_df['doy'].values

        # *Handle thermal data if available
        if args_predict["thermal_time_prediction"] is not None:
            thermal_data = pixel_df["thermal"].values  # Modify according to how thermal data is structured in your CSV
            thermal_data = torch.tensor(thermal_data, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device)

        # Preprocess data for the model
        data = data * args_predict['norm_factor_features']  # Normalize data
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        data = data.permute(0, 2, 1)  # This swaps the second and third dimensions

        doy = torch.tensor(doy, dtype=torch.long)
        device = next(model.parameters()).device
        data = data.to(device)  # Move data tensor to the correct device
        doy = doy.to(device)  # Ensure 'doy' tensor is also on the correct device
        # Predict
        with torch.no_grad():
            if args_predict["thermal_time_prediction"] is not None:
                prediction = model(data, doy, thermal_data)[0]
                #print(prediction)
            else:
                prediction = model(data, doy)[0]

            if args_predict["response"] == "classification":
                prediction = torch.argmax(prediction, dim=1)
            prediction = prediction.squeeze().item()  # Assuming single prediction
            if args_predict["norm_factor_response"] == "log10":
                prediction = 10 ** prediction - 1  # Reverse log10(x + 1) using Python's scalar operations
            elif args_predict["norm_factor_response"] != None:
                prediction = prediction / (args_predict["norm_factor_response"])

            #print(pixel_df['label'].iloc[0])
            #print(prediction)

            # Create a DataFrame for the new row you want to add
            new_row_df = pd.DataFrame([{
                'x': row['x'],
                'y': row['y'],
                'label': pixel_df['label'].iloc[0],  # Assuming label is constant
                'prediction': prediction,
                'aoi': row['aoi']
            }])
        # Use concat to add the new row to predictions_df
        predictions_df = pd.concat([predictions_df, new_row_df], ignore_index=True)
    # Save predictions
    predictions_df.to_csv(os.path.join(output_path, "predictions.csv"), index=False)

    # Convert to a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        predictions_df,
        geometry=[Point(xy) for xy in zip(predictions_df.x, predictions_df.y)],
        crs="EPSG:3035"  # Assuming original coordinates are in WGS84
    )

    # Save to a shapefile
    gdf.to_file(os.path.join(output_path, "predictions.shp"))
    return predictions_df