#!/usr/bin/python

import os, sys
import argparse
import random
import glob
#from osgeo import gdal, osr, ogr
#from osgeo.gdalconst import *
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from PIL import Image, ImageOps

import keras
from keras.models import Model, load_model

from tensorflow_legacy.sits.readingsits import *
from tensorflow_legacy.outputfiles.save import *
from tensorflow_legacy.deeplearning.architecture_features import *
from scipy import interpolate
#-----------------------------------------------------------------------

def stack_raster(tiles):

	#raster_glob_path = glob.glob(f"{tiles}/*.tif")
	# Define a custom sort key
	def custom_sort(filename):
		# Ordering based on the pattern in your filenames
		order = ["BLU", "GRN", "RED", "NIR", "SW1", "SW2", "RE1", "RE2", "RE3", "BNR"]
		for idx, band in enumerate(order):
			if band in filename:
				return idx
		return len(order)

	# Sort the files based on the custom key
	rasters = sorted(tiles, key=custom_sort)
	print(rasters)
	# Now, rasters list will contain the files in the desired order.

	# Open each raster and store the data in a list
	data = []
	for file in rasters:
		with rasterio.open(file) as src:
			data.append(src.read())

	with rasterio.open(rasters[0]) as src:
		meta = src.meta.copy()
		out_raster_crs = src.crs
		nodata = src.nodatavals[0]
		x, y = src.shape
	# Stack the data
	stacked_data = np.concatenate(data, axis=0)

	print(f"stacking finished: {tiles}")
	return stacked_data, meta, nodata, out_raster_crs, x, y
def main(model_path, tiles, proba, feature, extrapolate, regression, n_channels, sizex, sizey):
	glob_tiles = glob.glob(tiles)
	for tile in glob_tiles:
		tiles = glob.glob(f"{tile}/*TSI.tif")
		result_file = f"{tile}/predicted.tif"
		print(f"stacking: {tile}")
		stacked_raster, meta, nodata, out_raster_crs, xref, yref = stack_raster(tiles)
		print(f"classifying: {tile}")
		#-- Get the number of classes
		n_classes = getNoClasses(model_path)
		#-- Read min max values
		minMaxVal_file = '.'.join(model_path.split('.')[0:-1])
		minMaxVal_file = minMaxVal_file + '_minMax.txt'
		if os.path.exists(minMaxVal_file):
			min_per, max_per = read_minMaxVal(minMaxVal_file)
		else:
			assert False, "ERR: min-max values needs to be stored during training"
		#-- Downloading

		if regression == True:
			dtype = 'float32'
			meta['dtype'] = dtype
			meta['count'] = 1
		elif proba == False:
			dtype = 'uint8'
			meta['dtype'] = 'uint8'
			meta['count'] = 1
			meta['nodata'] = 255
		else:
			dtype = 'float32'
			meta['dtype'] = dtype
			meta['count'] = n_classes



		with rasterio.open(result_file, 'w', **meta) as dst:
			# Create a blank raster filled with NoData or zeros, etc.
			# You can fill it with nodata or some placeholder value
			dst.write(np.full((yref, xref), nodata, dtype=dtype), 1)
		#---- Loading the model
		model = load_model(model_path)

		size_areaX = sizex  # decrease the values if the tiff data cannot be in the memory, e.g. size_areaX = 10980, r =50 (get tiff BlockSize information for a nice setting)
		size_areaY = sizey
		x_vec = list(range(int(xref/size_areaX)))
		x_vec = [xref*size_areaX for xref in x_vec]
		y_vec = list(range(int(yref/size_areaY)))
		y_vec = [yref*size_areaY for yref in y_vec]
		x_vec.append(xref)
		y_vec.append(yref)
		count = 0
		for x in range(len(x_vec)-1):
			for y in range(len(y_vec)-1):
				count += 1
				xy_top_left = (x_vec[x],y_vec[y])
				xy_bottom_right = (x_vec[x+1],y_vec[y+1])
				#---- now loading associated data
				xoff = xy_top_left[0]
				yoff = xy_top_left[1]
				xsize = xy_bottom_right[0]-xy_top_left[0]
				ysize = xy_bottom_right[1]-xy_top_left[1]
				X_test = stacked_raster[:,yoff:yoff + ysize, xoff:xoff + xsize]
				X_test = X_test.astype(np.float32)
				X_test[X_test == nodata] = np.nan
				if np.all(np.isnan(X_test)):
					continue
				#---- reshape the cube in a column vector
				X_test = X_test.transpose((1, 2, 0))
				sX = X_test.shape[0]
				sY = X_test.shape[1]
				X_test = X_test.reshape(X_test.shape[0]*X_test.shape[1],X_test.shape[2])
				if extrapolate ==True:
					x_indices = np.tile(np.arange(X_test.shape[1] // n_channels), n_channels)
					for i in range(X_test.shape[0]):
						y_values = X_test[i]
						for j in range(n_channels):
							start_idx = j * (X_test.shape[1] // n_channels)
							end_idx = start_idx + (X_test.shape[1] // n_channels)

							nan_mask = np.isnan(y_values[start_idx:end_idx])
							non_nan_mask = ~nan_mask

							if non_nan_mask.any():
								y_interp = np.interp(x_indices[start_idx:end_idx],
													 x_indices[start_idx:end_idx][non_nan_mask],
													 y_values[start_idx:end_idx][non_nan_mask],
													 left=y_values[start_idx:end_idx][non_nan_mask][0],
													 right=y_values[start_idx:end_idx][non_nan_mask][-1])
								y_values[start_idx:end_idx][nan_mask] = y_interp[nan_mask]

						X_test[i] = y_values

				# X_interp now contains the interpolated data
				#---- pre-processing the data
				X_test = addingfeat_reshape_data(X_test, feature, n_channels)
				X_test = normalizingData(X_test, min_per, max_per)
				#---- saving the information
				p_img = model.predict(X_test)

				# Inside your loop (after predictions):
				with rasterio.open(result_file, 'r+') as dst:
					window = Window(col_off=xoff, row_off=yoff, width=xsize, height=ysize)
					if regression == True:
						y_test = p_img
						pred_array = y_test.reshape(sX, sY)
						dst.write(pred_array, 1, window=window)
					elif proba == False:
						y_test = p_img.argmax(axis=1)
						pred_array = y_test.reshape(sX, sY)
						dst.write(pred_array.astype('uint8'), 1, window=window)
					else:
						for b in range(n_classes):
							dst.write(p_img[:, :, b], b + 1, window=window)






#-----------------------------------------------------------------------
if __name__ == "__main__":
	try:
		if len(sys.argv) == 1:
			prog = os.path.basename(sys.argv[0])
			print('      '+sys.argv[0]+' [options]')
			print("     Help: ", prog, " --help")
			print("       or: ", prog, " -h")
			print("example 1: python %s --model_path path/to/model --test_file path/to/test.csv --result_file path/to/results/result.csv --proba" %sys.argv[0])
			sys.exit(-1)
		else:
			parser = argparse.ArgumentParser(description='Running deep learning architectures on SITS datasets')
			parser.add_argument('--model_path', dest='model_path',
								help='path to the trained model',
								default=None)
			parser.add_argument('--test_file', dest='test_file',
								help='file to classify (csv/tif)',
								default="csv")
			parser.add_argument('--result_file', dest='result_file',
								help='path where to store the output file (same extension than test_file)',
								default=None)
			parser.add_argument('--proba', dest='proba',
								help='if True probabilities, rather than class, are stored',
								default=False, action="store_true")
			parser.add_argument('--feat', dest='feature',
								help='used feature vector',
								default="SB")
			args = parser.parse_args()
			main(args.model_path, args.test_file, args.result_file, args.proba, args.feature)
			print("0")
	except(RuntimeError):
		print >> sys.stderr
		sys.exit(1)

#EOF
