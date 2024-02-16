# -*- coding: utf-8 -*-
"""
Created on Mon May  1 15:44:03 2023

@author: Admin
"""
import glob
import pandas as pd

coords_lst = glob.glob(r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\samples\*\coordinates*")

response_lst = glob.glob(r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\samples\*\response*")

features_lst = glob.glob(r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\samples\*\features*")

coords_out = r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\samples\coordinates.txt"

response_out = r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\samples\response.txt"

features_out = r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\samples\features.txt"



# Merge txt files for features
df_features = pd.concat([pd.read_csv(f, sep=' ', header=None) for f in features_lst])
df_features.to_csv(features_out, sep=' ', header=False, index=False)

# Merge txt files for response
df_response = pd.concat([pd.read_csv(f, sep=' ', header=None) for f in response_lst])
df_response.to_csv(response_out, sep=' ', header=False, index=False)

# Merge txt files for coords
df_coords = pd.concat([pd.read_csv(f, sep=' ', header=None) for f in coords_lst])
df_coords.to_csv(coords_out, sep=' ', header=False, index=False)