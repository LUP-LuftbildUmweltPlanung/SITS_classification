# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:34:02 2023

@author: Admin
"""
import geopandas as gpd
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


def validate_main(csv_ref, csv_test, response_name, aoi_name, algorithm_name, strat_validation):

    dataframe = gpd.GeoDataFrame()
    csv_test = pd.read_csv(csv_test)
    csv_ref = pd.read_csv(csv_ref)

    dataframe['pred'] = csv_test
    dataframe['ref'] = csv_ref

    # Drop rows with missing values in Potsdam
    dataframe.dropna(inplace=True)
    # Define a function to calculate R-squared and RMSE
    def calc_r_squared_and_rmse_sklearn(x, y):
        rsq = r2_score(y, x)
        rmse = np.sqrt(mean_squared_error(y, x))
        return rsq, rmse

    # Calculate R-squared and RMSE for Potsdam
    rsq_p, rmse_p = calc_r_squared_and_rmse_sklearn(dataframe['pred'], dataframe['ref'])
    print(f'Validation Metrics - {aoi_name}{algorithm_name}{response_name}:')
    print(f'R-squared: {rsq_p:.2f}, RMSE: {rmse_p:.2f}\n')


    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.scatter(dataframe['pred'], dataframe['ref'], s=0.02, c='green')
    plt.xlabel('Prediction', weight='bold')
    plt.ylabel('Reference', weight='bold')

    title_font = {'family': 'serif', 'weight': 'bold', 'size': 14}
    subtitle_font = {'family': 'serif', 'weight': 'normal', 'size': 12}  # Smaller and not bold

    plt.title(f'Distribution Prediction/Reference\n ', fontdict=title_font, loc='center')

    # Adjust the vertical position of the subtitle depending on your plot size.
    subtitle_position = 1.02
    plt.text(0.5, subtitle_position,
             f'{aoi_name}{algorithm_name}{response_name}',
             horizontalalignment='center',
             transform=plt.gca().transAxes,
             **subtitle_font)

    max_value = max(dataframe['pred'])
    # Define the position and dimensions of the rectangle
    left, width = max_value * 0, max_value * 0.35
    bottom, height = max_value * 0.8, max_value * 0.15
    right = left + width
    top = bottom + height

    # Add a rectangle with grey color and 50% transparency
    rect = Rectangle((left, bottom), width, height, facecolor='black', alpha=0.6)
    plt.gca().add_patch(rect)

    # Calculate middle points
    mid_horizontal = left + width / 2
    mid_vertical = bottom + height / 2

    # Add metrics as text centered in the middle of the rectangle
    # The ha and va parameters are used to align the text horizontally and vertically
    plt.text(mid_horizontal, mid_vertical + height * 1 / 4, f'R-squared: {rsq_p:.2f}', color='white', ha='center',
             va='center')
    plt.text(mid_horizontal, mid_vertical - height * 1 / 4, f'RMSE: {rmse_p:.2f}', color='white', ha='center',
             va='center')

    plt.show()

    if strat_validation == True:
        ref_lt5_p = dataframe[dataframe['ref'] < 10]
        ref_5to20_p = dataframe[(dataframe['ref'] >= 10) & (dataframe['ref'] < 20)]
        ref_gt20_p = dataframe[dataframe['ref'] > 20]

        # Calculate R-squared and RMSE for Berlin/Magdeburg using TempCNN predictions
        rsq_lt5_bm_temp, rmse_lt5_bm_temp = calc_r_squared_and_rmse_sklearn(ref_lt5_p['pred'], ref_lt5_p['ref'])
        rsq_5to20_bm_temp, rmse_5to20_bm_temp = calc_r_squared_and_rmse_sklearn(ref_5to20_p['pred'], ref_5to20_p['ref'])
        rsq_gt20_bm_temp, rmse_gt20_bm_temp = calc_r_squared_and_rmse_sklearn(ref_gt20_p['pred'], ref_gt20_p['ref'])



        print(f'Validation Metrics for different Heights - {aoi_name}{algorithm_name}{response_name}:')
        print(f'<10 m: RMSE: {rmse_lt5_bm_temp:.2f}')
        print(f'10-20m: RMSE: {rmse_5to20_bm_temp:.2f}')
        print(f'>20m: RMSE: {rmse_gt20_bm_temp:.2f}\n')
