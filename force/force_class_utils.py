# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:30:26 2023

@author: benjaminstoeckigt
"""

import os
import subprocess
import time
import shutil
import rasterio
import glob
import geopandas as gpd
import re
import json
from pathlib import Path
def generate_input_feature_line(tif_path, num_layers):
    sequence = ' '.join(str(i) for i in range(1, num_layers + 1))
    return f"INPUT_FEATURE = {tif_path} {sequence}"

def replace_parameters_feature(filename, tif_file, order):
    #tif_files = glob.glob(f"{tif_file}/*.tif")
    bandnumbers = rasterio.open(glob.glob(f"{tif_file}/*.tif")[1]).count
    tif_files = [os.path.basename(f) for f in glob.glob(f"{tif_file}/*.tif")]

    tif_files.sort(key=lambda x: order.index(x.split('_')[-2]))
    input_features = [generate_input_feature_line(tif_file, bandnumbers) for tif_file in tif_files]

    # Now let's write these lines to the file
    with open(filename, "r") as file:
        lines = file.readlines()

    # Find index where the original INPUT_FEATURE lines start and end
    start_index = next(i for i, line in enumerate(lines) if "INPUT_FEATURE =" in line)
    end_index = next(i for i, line in enumerate(lines[start_index:]) if "INPUT_FEATURE =" not in line) + start_index

    # Replace the original INPUT_FEATURE lines with the new ones
    lines = lines[:start_index] + input_features + ['\n'] + lines[end_index:]

    # Write the updated lines back to the file
    with open(filename, "w") as file:
        file.writelines(line if line.endswith('\n') else line + '\n' for line in lines)

def replace_parameters(filename, replacements):
    with open(filename, 'r') as f:
        content = f.read()
        for key, value in replacements.items():
            content = content.replace(key, value)
    with open(filename, 'w') as f:
        f.write(content)

def extract_coordinates(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    #Skip the first line
    lines = lines[1:]
    #Extract X and Y values
    x_values = [int(line.split('_')[0][1:]) for line in lines]
    y_values = [int(line.split('_')[1][1:]) for line in lines]
    #Extract the desired values
    x_str = f"{min(x_values)} {max(x_values)}"
    y_str = f"{min(y_values)} {max(y_values)}"

    return x_str, y_str

def check_and_reproject_shapefile(shapefile_path, target_epsg=3035):
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    # Check the current CRS of the shapefile
    if gdf.crs.to_epsg() != target_epsg:
        print("Reprojecting shapefile to EPSG: 3035")
        # Reproject the shapefile
        gdf = gdf.to_crs(epsg=target_epsg)
        # Define the new file path
        new_shapefile_path = shapefile_path.replace(".shp", "_3035.shp")
        # Save the reprojected shapefile
        gdf.to_file(new_shapefile_path, driver='ESRI Shapefile')
        print(f"Shapefile reprojected and saved to {new_shapefile_path}")
        return new_shapefile_path
    else:
        print("Shapefile is already in EPSG: 3035")
        return shapefile_path

def force_class(preprocess_params):
    force_dir = f"{preprocess_params['force_dir']}:{preprocess_params['force_dir']}"
    local_dir = f"{os.sep + preprocess_params['process_folder'].split(os.sep)[1]}:{os.sep + preprocess_params['process_folder'].split(os.sep)[1]}"
    scripts_skel = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/force/skel"
    force_skel = f"{scripts_skel}/force_cube_sceleton"
    temp_folder = preprocess_params['process_folder'] + "/temp"
    mask_folder = preprocess_params['process_folder'] + "/temp/_mask"

    preprocess_params.setdefault("sample", True) ## True for Training

    #defining parameters outsourced from main script
    if preprocess_params["Interpolation"] == False:
        OUTPUT_TSS = 'TRUE'  ## Classification based on TSS just possible for Transformer
        OUTPUT_TSI = 'FALSE'  ## Classification based on TSI possible for Transformer, TempCNN, , LSTM, MSResnet
    elif preprocess_params["Interpolation"] == True:
        OUTPUT_TSS = 'FALSE'  ## Classification based on TSS just possible for Transformer
        OUTPUT_TSI = 'TRUE'  ## Classification based on TSI possible for Transformer, TempCNN, , LSTM, MSResnet

    project_name = preprocess_params["project_name"]

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
    for aoi,DATE_RANGE in zip(preprocess_params["aois"], preprocess_params["date_ranges"]):
        print(f"FORCE PROCESSING FOR {aoi} WITHIN TIME RANGE {DATE_RANGE}")

        basename = os.path.basename(aoi)
        aoi = check_and_reproject_shapefile(aoi)


        ### get force extend
        os.makedirs(f'{temp_folder}/{project_name}/FORCE/{basename}', exist_ok=True)

        subprocess.run(['sudo', 'chmod', '-R', '777', f"{temp_folder}/{project_name}/FORCE/{basename}"])

        shutil.copy(f"{force_skel}/datacube-definition.prj",f"{temp_folder}/{project_name}/FORCE/{basename}/datacube-definition.prj")

        cmd = f"sudo docker run -v {local_dir} -v {force_dir} davidfrantz/force " \
               f"force-tile-extent {aoi} {force_skel} {temp_folder}/{project_name}/FORCE/{basename}/tile_extent.txt"

        if preprocess_params["hold"] == True:
            subprocess.run(['xterm','-hold','-e', cmd])
        else:
            subprocess.run(['xterm', '-e', cmd])

        subprocess.run(['sudo','chmod','-R','777',f"{temp_folder}/{project_name}/FORCE/{basename}"])

        ### mask
        os.makedirs(f"{mask_folder}/{project_name}/{basename}", exist_ok=True)

        subprocess.run(['sudo', 'chmod', '-R', '777', f"{mask_folder}"])

        shutil.copy(f"{force_skel}/datacube-definition.prj",f"{mask_folder}/{project_name}/{basename}/datacube-definition.prj")
        cmd = f"sudo docker run -v {local_dir} davidfrantz/force " \
              f"force-cube -o {mask_folder}/{project_name}/{basename} " \
              f"{aoi}"

        if preprocess_params["hold"] == True:
            subprocess.run(['xterm','-hold','-e', cmd])
        else:
            subprocess.run(['xterm', '-e', cmd])
        subprocess.run(['sudo','chmod','-R','777',f"{mask_folder}/{project_name}/{basename}"])

        ###mask mosaic
        cmd = f"sudo docker run -v {local_dir} davidfrantz/force " \
              f"force-mosaic {mask_folder}/{project_name}/{basename}"

        if preprocess_params["hold"] == True:
            subprocess.run(['xterm','-hold','-e', cmd])
        else:
            subprocess.run(['xterm', '-e', cmd])

        subprocess.run(['sudo','chmod','-R','777',f"{temp_folder}/{project_name}/FORCE/{basename}"])

        ###force param

        os.makedirs(f"{temp_folder}/{project_name}/FORCE/{basename}/provenance", exist_ok=True)
        os.makedirs(f"{temp_folder}/{project_name}/FORCE/{basename}/tiles_tss", exist_ok=True)

        shutil.copy(f"{force_skel}/datacube-definition.prj",f"{temp_folder}/{project_name}/FORCE/{basename}/datacube-definition.prj")
        shutil.copy(f"{force_skel}/datacube-definition.prj",f"{temp_folder}/{project_name}/FORCE/{basename}/tiles_tss/datacube-definition.prj")
        shutil.copy(f"{scripts_skel}/TSA_NoCom.prm", f"{temp_folder}/{project_name}/FORCE/{basename}/tsa.prm")


        X_TILE_RANGE, Y_TILE_RANGE = extract_coordinates(f"{temp_folder}/{project_name}/FORCE/{basename}/tile_extent.txt")
        # Define replacements
        replacements = {
            # INPUT/OUTPUT DIRECTORIES
            f'DIR_LOWER = NULL':f'DIR_LOWER = {force_dir.split(":")[0]}/FORCE/C1/L2/ard',
            f'DIR_HIGHER = NULL':f'DIR_HIGHER = {temp_folder}/{project_name}/FORCE/{basename}/tiles_tss',
            f'DIR_PROVENANCE = NULL':f'DIR_PROVENANCE = {temp_folder}/{project_name}/FORCE/{basename}/provenance',
            # MASKING
            f'DIR_MASK = NULL':f'DIR_MASK = {mask_folder}/{project_name}/{basename}',
            f'BASE_MASK = NULL':f'BASE_MASK = {os.path.basename(aoi).replace(".shp",".tif")}',
            # PARALLEL PROCESSING
            f'NTHREAD_READ = 8':f'NTHREAD_READ = {preprocess_params["NTHREAD_READ"]}',
            f'NTHREAD_COMPUTE = 22':f'NTHREAD_COMPUTE = {preprocess_params["NTHREAD_COMPUTE"]}',
            f'NTHREAD_WRITE = 4':f'NTHREAD_WRITE = {preprocess_params["NTHREAD_WRITE"]}',
            # PROCESSING EXTENT AND RESOLUTION
            f'X_TILE_RANGE = 0 0':f'X_TILE_RANGE = {X_TILE_RANGE}',
            f'Y_TILE_RANGE = 0 0':f'Y_TILE_RANGE = {Y_TILE_RANGE}',
            f'BLOCK_SIZE = 0':f'BLOCK_SIZE = {preprocess_params["BLOCK_SIZE"]}',
            # SENSOR ALLOW-LIST
            f'SENSORS = LND08 LND09 SEN2A SEN2B':f'SENSORS = {preprocess_params["Sensors"]}',
            f'SPECTRAL_ADJUST = FALSE':f'SPECTRAL_ADJUST = {preprocess_params["SPECTRAL_ADJUST"]}',
            # QAI SCREENING
            f'SCREEN_QAI = NODATA CLOUD_OPAQUE CLOUD_BUFFER CLOUD_CIRRUS CLOUD_SHADOW SNOW SUBZERO SATURATION':f'SCREEN_QAI = NODATA CLOUD_OPAQUE CLOUD_BUFFER CLOUD_CIRRUS CLOUD_SHADOW SNOW SUBZERO SATURATION',
            f'ABOVE_NOISE = 3':f'ABOVE_NOISE = {preprocess_params["ABOVE_NOISE"]}',
            f'BELOW_NOISE = 1':f'BELOW_NOISE = {preprocess_params["BELOW_NOISE"]}',
            # PROCESSING TIMEFRAME
            f'DATE_RANGE = 2010-01-01 2019-12-31':f'DATE_RANGE = {DATE_RANGE}',
            # SPECTRAL INDEX
            f'INDEX = NDVI EVI NBR':f'INDEX = {preprocess_params["Indices"]}',
            f'OUTPUT_TSS = FALSE':f'OUTPUT_TSS = {OUTPUT_TSS}',
            # INTERPOLATION PARAMETERS
            f'INTERPOLATE = RBF':f'INTERPOLATE = {preprocess_params["INTERPOLATE"]}',
            f'INT_DAY = 16':f'INT_DAY = {preprocess_params["INT_DAY"]}',
            f'OUTPUT_TSI = FALSE':f'OUTPUT_TSI = {OUTPUT_TSI}',
        }
        # Replace parameters in the file
        replace_parameters(f"{temp_folder}/{project_name}/FORCE/{basename}/tsa.prm", replacements)

        subprocess.run(['sudo','chmod','-R','777',f"{temp_folder}/{project_name}/FORCE/{basename}"])

        cmd = f"sudo docker run -it -v {local_dir} -v {force_dir} davidfrantz/force " \
              f"force-higher-level {temp_folder}/{project_name}/FORCE/{basename}/tsa.prm"

        if preprocess_params["hold"] == True:
            subprocess.run(['xterm', '-hold', '-e', cmd])
        else:
            subprocess.run(['xterm', '-e', cmd])

        subprocess.run(['sudo', 'chmod', '-R', '777', f"{temp_folder}/{project_name}/FORCE/{basename}"])

        if preprocess_params["sample"] == True:

            if not os.path.exists(aoi.replace(".shp",".csv")):
                #Read the shapefile with geopandas
                gdf = gpd.read_file(aoi)
                # Check if the CRS matches EPSG:3035
                if gdf.crs != "EPSG:3035":
                    print(f"detected crs {gdf.crs} reprojecting to EPSG:3035")
                    gdf = gdf.to_crs("EPSG:3035")
                # Create new columns for X and Y coordinates
                gdf['X'] = gdf.geometry.x
                gdf['Y'] = gdf.geometry.y

                # Reorder the columns
                gdf = gdf[['X', 'Y', preprocess_params["column_name"], 'geometry']]
                # Drop the CID column
                gdf_csv = gdf.drop('geometry', axis=1)
                # Export the file as CSV without index and header
                gdf_csv.to_csv(aoi.replace(".shp",".csv"), index=False, header=False, sep=' ')

            tile_lst = glob.glob(f"{temp_folder}/{project_name}/FORCE/{basename}/tiles_tss/X*")
            for tile in tile_lst:
                t_name = os.path.basename(tile)
                x_c = int(t_name[3:5])
                y_c = int(t_name[9:11])

                basename = os.path.basename(aoi)
                shutil.copy(f"{scripts_skel}/SAMPLE_NoCom.prm", f"{temp_folder}/{project_name}/FORCE/{basename}/sample_X00{x_c}_Y00{y_c}.prm")


                X_TILE_RANGE, Y_TILE_RANGE = extract_coordinates(f"{temp_folder}/{project_name}/FORCE/{basename}/tile_extent.txt")
                # Define replacements
                replacements = {
                    # INPUT/OUTPUT DIRECTORIES
                    f'DIR_LOWER = NULL':f'DIR_LOWER = {temp_folder}/{project_name}/FORCE/{basename}/tiles_tss/',
                    f'DIR_HIGHER = NULL':f'DIR_HIGHER = {temp_folder}/{project_name}/FORCE/{basename}/',
                    f'DIR_PROVENANCE = NULL':f'DIR_PROVENANCE = {temp_folder}/{project_name}/FORCE/{basename}/provenance/',
                    # MASKING
                    f'DIR_MASK = NULL':f'DIR_MASK = {mask_folder}/{project_name}/{basename}',
                    f'BASE_MASK = NULL':f'BASE_MASK = {os.path.basename(aoi).replace(".shp",".tif")}',
                    # PARALLEL PROCESSING
                    f'NTHREAD_READ = 8':f'NTHREAD_READ = {preprocess_params["NTHREAD_READ"]}',
                    f'NTHREAD_COMPUTE = 22':f'NTHREAD_COMPUTE = {preprocess_params["NTHREAD_COMPUTE"]}',
                    f'NTHREAD_WRITE = 4':f'NTHREAD_WRITE = {preprocess_params["NTHREAD_WRITE"]}',
                    # PROCESSING EXTENT AND RESOLUTION
                    f'X_TILE_RANGE = 0 0':f'X_TILE_RANGE = {x_c} {x_c}',
                    f'Y_TILE_RANGE = 0 0':f'Y_TILE_RANGE = {y_c} {y_c}',
                    f'BLOCK_SIZE = 0':f'BLOCK_SIZE = {preprocess_params["BLOCK_SIZE"]}',
                    f'FEATURE_NODATA = -9999':f'FEATURE_NODATA = -9999',
                    f'FEATURE_EXCLUDE = FALSE':f'FEATURE_EXCLUDE = FALSE',
                    # SAMPLING
                    f'FILE_POINTS = NULL':f'FILE_POINTS = {aoi.replace(".shp",".csv")}',
                    f'FILE_SAMPLE = NULL':f'FILE_SAMPLE = {temp_folder}/{project_name}/FORCE/{basename}/tiles_tss/features_X00{x_c}_Y00{y_c}.txt',
                    f'FILE_RESPONSE = NULL':f'FILE_RESPONSE = {temp_folder}/{project_name}/FORCE/{basename}/tiles_tss/response_X00{x_c}_Y00{y_c}.txt',
                    f'FILE_COORDINATES = NULL':f'FILE_COORDINATES = {temp_folder}/{project_name}/FORCE/{basename}/tiles_tss/coordinates_X00{x_c}_Y00{y_c}.txt',
                    f'PROJECTED = FALSE':f'PROJECTED = TRUE',
                }

                # Replace parameters in the file
                replace_parameters(f"{temp_folder}/{project_name}/FORCE/{basename}/sample_X00{x_c}_Y00{y_c}.prm", replacements)

                # Use the function
                tif_file = glob.glob(f"{temp_folder}/{project_name}/FORCE/{basename}/tiles_tss/X00{x_c}_Y00{y_c}")[0]
                replace_parameters_feature(f"{temp_folder}/{project_name}/FORCE/{basename}/sample_X00{x_c}_Y00{y_c}.prm", tif_file, preprocess_params["feature_order"])

                cmd = f"sudo docker run -it -v {local_dir} -v {force_dir} davidfrantz/force " \
                      f"force-higher-level {temp_folder}/{project_name}/FORCE/{basename}/sample_X00{x_c}_Y00{y_c}.prm"

                if preprocess_params["hold"] == True:
                    subprocess.run(['xterm', '-hold', '-e', cmd])
                else:
                    subprocess.run(['xterm', '-e', cmd])

                subprocess.run(['sudo', 'chmod', '-R', '777', f"{temp_folder}/{project_name}/FORCE/{basename}"])

    endzeit = time.time()
    print("FORCE-Processing beendet nach "+str((endzeit-startzeit)/60)+" Minuten")