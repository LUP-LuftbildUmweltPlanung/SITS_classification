import os
from pathlib import Path
import torch
import rasterio
import sys
import numpy as np

from pytorch.train import getModel
from pytorch.utils.hw_monitor import HWMonitor, disk_info
from tqdm import tqdm
import glob
import json

def load_model(model_path,args):

    # Load a PyTorch model from the given path
    saved_state = torch.load(model_path)
    model_state_dict = saved_state["model_state"]
    args['nclasses'] = saved_state["nclasses"]
    args['seqlength'] = saved_state["sequencelength"]
    args['input_dims'] = saved_state["ndims"]
    print(f"Sequence Length: {args['seqlength']}")
    print(f"Input Dims: {args['input_dims']}")
    print(f"Prediction Classes: {args['nclasses']}")
    model = getModel(args)
    model.load_state_dict(model_state_dict)
    model.eval()  # Set the model to evaluation mode
    return model

def read_tif_files(folder_path, order,normalizing_factor):
    bands_data = []
    #print(folder_path)
    for band in order:
        # Find the file that matches the pattern
        file_pattern = os.path.join(folder_path, f"*{band}_TSI.tif")
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
            #print(band_data[band_data!=0])
            bands_data.append(band_data)
    # Stack the bands along a new axis (bands axis)
    return bands_data


def predict(model, tiles, args_predict):
    print(f"Preprocessing the Data for Prediction ...")
    # Read TIFF files
    order = args_predict["order"]
    normalizing_factor = args_predict["normalizing_factor"]
    chunksize = args_predict["chunksize"]

    data = read_tif_files(tiles, order, normalizing_factor)

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
            batch = torch.tensor(batch, dtype=torch.float32, device=device)

            # Predict
            batch_predictions = model.forward(batch)[0]

            # Handle classification or regression
            if args_predict["response"] == "classification":
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
    reshaped_predictions = predictions.reshape(3000, 3000).cpu().numpy()

    # Find one of the existing TIFF files to copy the metadata
    existing_tif_path = glob.glob(os.path.join(tiles, "*.tif"))[0]
    if not existing_tif_path:
        raise FileNotFoundError("No TIFF files found in the folder to copy metadata.")

    # Read the existing TIFF file to get metadata
    with rasterio.open(existing_tif_path) as src:
        metadata = src.meta

    if args_predict["response"] == "classification":
        reshaped_predictions = reshaped_predictions.astype('uint8')
        metadata.update(dtype=rasterio.uint8, nodata=255, count=1)
    else:
        metadata.update(dtype=rasterio.float32, count=1)

    # Write the predictions to a new TIFF file
    output_path = os.path.join(tiles, "predicted.tif")
    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(reshaped_predictions, 1)


def load_hyperparametersplus(model_name):
    """
    Load the hyperparameters from a JSON file.
    """
    # Path where the file is saved
    file_path = os.path.join(model_name, "hyperparameters.json")

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No hyperparameters file found at {file_path}")

    # Reading from the file
    with open(file_path, 'r') as file:
        hyperparameters = json.load(file)

    return hyperparameters


def predict_init(args_predict):

    hw_args = args_predict['hw_monitor']

    # create hw_monitor output dir if it doesn't exist
    Path(hw_args['hw_logs_dir']).mkdir(parents=True, exist_ok=True)

    hw_predict_logs_file = hw_args['hw_logs_dir'] + '/' + hw_args['hw_predict_logs_file_name']

    # Instantiate monitor with a 1-second delay between updates
    hwmon_p = HWMonitor(1,hw_predict_logs_file,hw_args['disks_to_monitor'])
    # start monitoring
    hwmon_p.start()

    hyp = load_hyperparametersplus(os.path.dirname(args_predict["model_path"]))
    args_predict.update(hyp)

    model_path = args_predict['model_path']
    tiles = args_predict['folder_path']
    glob_tiles = glob.glob(tiles)

    model = load_model(model_path,args_predict)


    for tile in glob_tiles:
        print("###" * 15)
        print(f"Started Prediction for Tile {os.path.basename(tile)}")
        prediction = predict(model,tile,args_predict)
        reshape_and_save(prediction,tile,args_predict)

    # stop monitoring
    hwmon_p.stop()