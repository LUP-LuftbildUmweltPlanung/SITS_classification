import os
import torch
import rasterio
import sys
import numpy as np

from pathlib import Path

from pytorch.train import getModel
from pytorch.utils.hw_monitor import HWMonitor, disk_info, squeeze_hw_info
from tqdm import tqdm
import glob
import json
import datetime
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from rasterio.merge import merge

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
    args['seqlength'] = 366*int(args["time_range"][0])
    args['input_dims'] = saved_state["ndims"]
    #print(f"Sequence Length: {args['seqlength']}")
    print(f"Input Dims: {args['input_dims']}")
    print(f"Prediction Classes: {args['nclasses']}")
    model = getModel(args)

    model.load_state_dict(model_state_dict)

    model.eval()  # Set the model to evaluation mode
    return model

def read_tif_files(folder_path, order, year, month, day):
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
            band_data = band_data
            bands_data.append(band_data)

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
    return bands_data,doy


def predict(model, tiles, args_predict):
    print(f"Preprocessing the Data for Prediction ...")
    # Read TIFF files
    order = args_predict["order"]
    normalizing_factor = args_predict["norm_factor_features"]
    chunksize = args_predict["chunksize"]
    year = int(args_predict['time_range'][0])
    month = int(args_predict['time_range'][1].split('-')[0])
    day = int(args_predict['time_range'][1].split('-')[1])

    data, doy_single = read_tif_files(tiles, order, year, month, day)

    # Stack the bands and perform initial reshape in NumPy
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
    print(f"Predicting with Chunksize {chunksize} from {data.shape[0]}")
    with torch.no_grad():
        for i in tqdm(range(0, data.shape[0], chunksize)):
            batch = data[i:i + chunksize] * normalizing_factor
            non_zero_mask = np.any(batch != 0, axis=(1, 2))
            doy = np.array(doy_single)
            doy = np.tile(doy, (batch.shape[0], 1))

            if not np.any(non_zero_mask):
                batch_predictions = torch.full((batch.shape[0], 1), -9999, dtype=torch.float32, device=device)
            else:
                batch_non_zero = batch[non_zero_mask]
                doy_non_zero = doy[non_zero_mask]
                batch_tensor = torch.tensor(batch_non_zero, dtype=torch.float32, device=device)
                doy_tensor = torch.tensor(doy_non_zero, dtype=torch.long, device=device)

                predictions_non_zero = model(batch_tensor, doy_tensor)[0]

                # Handle normalization response factor
                norm_factor_response = args_predict.get("norm_factor_response")
                if norm_factor_response == "log":
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

def predict_raster(args_predict):

    hyp = load_hyperparametersplus(os.path.dirname(args_predict["model_path"]))
    args_predict.update(hyp)

    # create hw_monitor output dir if it doesn't exist
    drive_name = ["sdb1"]
    Path(args_predict['store'] + '/' + args_predict['model'] + '/hw_monitor').mkdir(parents=True, exist_ok=True)

    hw_predict_logs_file = args_predict['store'] + '/' + args_predict['model'] + '/hw_monitor/hw_monitor_predict.csv'
    # Instantiate monitor with a 1-second delay between updates
    hwmon_p = HWMonitor(1,hw_predict_logs_file,drive_name)
    hwmon_p.start()
    hwmon_p.start_averaging()

    model_path = args_predict['model_path']
    tiles = args_predict['folder_path']
    glob_tiles = glob.glob(tiles)
    model = load_model(model_path,args_predict)

    for tile in glob_tiles:
        print("###" * 15)
        print(f"Started Prediction for Tile {os.path.basename(tile)}")
        prediction = predict(model,tile,args_predict)
        reshape_and_save(prediction,tile,args_predict)


    hwmon_p.stop_averaging()
    avgs = hwmon_p.get_averages()
    squeezed = squeeze_hw_info(avgs)
    mean_data = {key: round(value, 1) for key, value in squeezed.items() if "mean" in key}
    print(f"##################\nMean Values Hardware Monitoring (Prediction):\n{mean_data}\n##################")
    hwmon_p.stop()

def predict_csv(args_predict):

    output_path = args_predict["reference_folder"]
    reference_folder = args_predict["reference_folder"]
    folder_path = reference_folder+"/sepfiles/test/csv"
    meta_path = reference_folder+"/meta.csv"

    hyp = load_hyperparametersplus(os.path.dirname(args_predict["model_path"]))
    args_predict.update(hyp)
    model_path = args_predict['model_path']
    order = args_predict["order"]
    model = load_model(model_path, args_predict)

    # Load metadata
    metadata_df = pd.read_csv(meta_path)

    # Prepare a DataFrame to hold all predictions
    predictions_df = pd.DataFrame(columns=['x', 'y', 'label', 'prediction'])

    # Iterate over CSV files
    for idx, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0]):
    #for idx, row in metadata_df.iterrows():
        csv_file_path = os.path.join(folder_path, f"{row['global_idx']}.csv")
        if not os.path.exists(csv_file_path):
            continue  # Skip if file doesn't exist

        # Read spectral data
        pixel_df = pd.read_csv(csv_file_path)

        # Extract relevant columns and normalize if necessary
        data = pixel_df[order].values
        doy = pixel_df['doy'].values

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
            prediction = model(data, doy)[0]
            if args_predict["response"] == "classification":
                prediction = torch.argmax(prediction, dim=1)
            prediction = prediction.squeeze().item()  # Assuming single prediction
            if args_predict["norm_factor_response"] != None:
                prediction = prediction / (args_predict["norm_factor_response"])
            elif args_predict["norm_factor_response"] == "log":
                prediction = 10 ** prediction - 1  # Reverse log10(x + 1) using Python's scalar operations


            # Create a DataFrame for the new row you want to add
            new_row_df = pd.DataFrame([{
                'x': row['x'],
                'y': row['y'],
                'label': pixel_df['label'].iloc[0],  # Assuming label is constant
                'prediction': prediction
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


def predict_init(args_predict, proc_folder, temp_folder, **kwargs):

    if not args_predict['reference_folder']:
        if isinstance(args_predict['aois'], list):
            for basen in args_predict['aois']:
                basename = os.path.basename(basen)
                args_predict['folder_path'] = f"{temp_folder}/{args_predict['project_name']}/FORCE/{basename}/tiles_tss/X*"

                predict_raster(args_predict)

                files = glob.glob(f"{temp_folder}/{args_predict['project_name']}/FORCE/{basename}/tiles_tss/X*/predicted.tif")
                output_filename = f"{proc_folder}/{args_predict['project_name']}/{os.path.basename(basen.replace('.shp','.tif'))}"

                mosaic_rasters(files, output_filename)
        else:
            predict_raster(args_predict)

    else:
        predict_csv(args_predict)
