# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:50:22 2023

@author: Admin
"""
import geopandas as gpd
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import rasterio
import numpy as np


def shape_to_forcecsv(shape,outputcsv,drop_lst,response):

    file = gpd.read_file(shape)
    
    # Drop the CID column
    file = file.drop(drop_lst, axis=1)
    
    # Create new columns for X and Y coordinates
    file['X'] = file.geometry.x
    file['Y'] = file.geometry.y
    
    # Reorder the columns
    file = file[['X', 'Y', response, 'geometry']]
    # Drop the CID column
    file = file.drop('geometry', axis=1)
    # Export the file as CSV without index and header
    file.to_csv(outputcsv, index=False, header=False, sep=' ')





def forcesample_tocsv(feature_path,response_path,bands,split_train):
    
    features = pd.read_csv(feature_path,sep=" ", header = None)
    response = pd.read_csv(response_path, header = None)
    features = features.replace(-9999, np.nan)
    ##interpolate for every band of every point
    band_length = features.shape[1] // bands
    for i in range(bands):
        start = i * band_length
        end = start + band_length
        band = features.iloc[:,start:end]
        band_interpolated = band.interpolate(axis=1, limit_direction='both')
        features.iloc[:,start:end] = band_interpolated
        #print(band)
    
    # Add row ID as a new column
    features['row_id'] = features.index
    
    # Concatenate response and interpolated_features DataFrames
    result = pd.concat([response, features['row_id'], features.drop(columns=['row_id'])], axis=1)
    result.columns = range(result.shape[1])
    result = result.loc[result[0] != -9999]
    
    result.columns = result.columns.astype(str)
    
    
    # Define the train percentage
    train_perc = split_train
    
    
    train_result, test_result = train_test_split(result, train_size=train_perc, random_state=42)
    
    train_result = train_result.dropna()
    test_result = test_result.dropna()
    
    train_output = feature_path.replace(".txt","_train.csv")
    test_output = feature_path.replace(".txt","_test.csv")
    
    train_result.to_csv(train_output, index=False, header=False, sep=",")
    test_result.to_csv(test_output, index=False, header=False, sep=",")
    






def stack_raster(rasters,output):
    # Open each raster and store the data in a list
    data = []
    for file in rasters:
        with rasterio.open(file) as src:
            data.append(src.read())
    
    # Stack the data
    stacked_data = np.concatenate(data, axis=0)
    
        
    # Get the metadata of one of the input rasters and update the count to match the stacked data
    with rasterio.open(rasters[0]) as src:
        meta = src.meta.copy()
    meta.update(count=stacked_data.shape[0])
    
    # Write the stacked data to a new raster file
    with rasterio.open(output, "w", **meta) as dest:
        dest.write(stacked_data)
    print(f"stacking finished: {output}")