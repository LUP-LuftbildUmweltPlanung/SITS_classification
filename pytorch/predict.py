import os
import torch
import rasterio
import sys
import numpy as np

from pathlib import Path

from utils.class_run import mosaic_rasters
from pytorch.train import getModel
from pytorch.utils.hw_monitor import HWMonitor, disk_info, squeeze_hw_info
from tqdm import tqdm
import glob
import json
import datetime
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def load_model(model_path,args):

    # Load a PyTorch model from the given path
    saved_state = torch.load(model_path)
    model_state_dict = saved_state["model_state"]
    args['nclasses'] = saved_state["nclasses"]
    args['seqlength'] = saved_state["sequencelength"]
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
    latest_year = max(doa_dates, key=lambda x: x.year).year
    start_date = datetime.datetime(latest_year-year, month, day)

    doy = [(doa_date - start_date).days + 1 for doa_date in doa_dates]
    return bands_data,doy


def predict(model, tiles, args_predict):
    print(f"Preprocessing the Data for Prediction ...")
    # Read TIFF files
    order = args_predict["order"]
    normalizing_factor = args_predict["normalizing_factor"]
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
            #print(i)  # Debugging print to check the batch index
            # Select batch and perform normalization and conversion to tensor here
            batch = data[i:i + chunksize] * normalizing_factor

            # Correctly casting DOY to long
            doy = np.array(doy_single)

            # ####### Preprocess step: Filter and Pad START
            # # Updated Preprocess step: Filter and Pad
            # # Updated filter and pad logic
            # mask = np.any(batch != 0, axis=1)  # Identify non-zero spectral values across timesteps
            # max_timesteps = max(np.sum(mask, axis=1)) if np.sum(mask) > 0 else 1  # Ensure at least 1
            # # Initialize new_batch with zeros to ensure a tensor is always created
            # new_batch = np.zeros((batch.shape[0], batch.shape[1], max_timesteps))
            # new_doy = np.zeros((batch.shape[0], max_timesteps))
            # for j in range(batch.shape[0]):
            #     valid_timesteps = np.where(mask[j])[0]
            #     if len(valid_timesteps) > 0:
            #         # Correctly copy valid timesteps
            #         # The correct shape should be maintained, so make sure dimensions match
            #         new_batch[j, :, :len(valid_timesteps)] = batch[j, :, valid_timesteps].transpose()
            #         new_doy[j, :len(valid_timesteps)] = doy[valid_timesteps].transpose()  # Copy valid DOYs
            # ####### Preprocess step: Filter and Pad ENDE

            current_batch_size = batch.shape[0]  # Actual batch size may be less than chunksize for the last chunk
            # expanding it
            doy = np.tile(doy, (current_batch_size, 1))

            # Create a boolean mask for samples where all values across all spectral bands are 0
            #all_zero_spectral = np.all(batch == 0, axis=1)
            # Apply the mask: Set doy values to 0 where the condition is true
            #doy[all_zero_spectral] = 0
            batch = torch.tensor(batch, dtype=torch.float32, device=device)
            batch_doy = torch.tensor(doy, dtype=torch.long, device=device)
            # Predict
            #print(batch.shape)
            #print(batch_doy.shape)
            #import time
            #time.sleep(10000)
            batch_predictions = model.forward(batch, batch_doy)[0]

            # Handle classification or regression
            if args_predict["response"] == "classification" and not args_predict["probability"]:
                batch_predictions = torch.argmax(batch_predictions, dim=1)

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
        data = data * args_predict['normalizing_factor']  # Normalize data
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
