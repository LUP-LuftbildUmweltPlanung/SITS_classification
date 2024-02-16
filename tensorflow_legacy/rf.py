
from sits.readingsits import *
import time
import joblib
from osgeo import gdal, osr, ogr

def train_rf(sits_path_train, regression, res_path, feature, n_channels):
    if regression==False:
        from sklearn.ensemble import RandomForestClassifier
    else:
        from sklearn.ensemble import RandomForestRegressor
    # -- Creating output path if does not exist
    if not os.path.exists(res_path):
        os.makedirs(res_path)


    train_str = sits_path_train

    # ---- Get filenames
    train_file = train_str + '.csv'
    print("train_file: ", train_file)


    # ---- output files
    res_path = res_path + '/rf'+'/'
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    model_file = res_path+"test.joblib"
    out_model_file = model_file
    # ---- Downloading
    X_train, polygon_ids_train, y_train = readSITSData(train_file)

    if not regression:
        n_classes_train = len(np.unique(y_train))
        if (n_classes_test != n_classes_train):
            print("WARNING: different number of classes in train and test")
        n_classes = max(n_classes_train, n_classes_test)

        y_train_one_hot = to_categorical(y_train, n_classes)
    else:
        y_train_one_hot = y_train
    # print(X_train.shape)
    # print(type(X_train))
    # print(X_train)
    # ---- Adding the features and reshaping the data if necessary
    X_train = addingfeat_reshape_data(X_train, feature, n_channels)
    # print(X_train.shape)
    # ---- Normalizing the data per band
    minMaxVal_file = '.'.join(out_model_file.split('.')[0:-1])
    minMaxVal_file = minMaxVal_file + '_minMax.txt'
    if not os.path.exists(minMaxVal_file):
        min_per, max_per = computingMinMax(X_train)
        save_minMaxVal(minMaxVal_file, min_per, max_per)
    else:
        min_per, max_per = read_minMaxVal(minMaxVal_file)
    X_train = normalizingData(X_train, min_per, max_per)


    if regression ==False:
        rf = RandomForestClassifier(n_estimators=400, max_features='sqrt',
                                    max_depth=25, min_samples_split=2, oob_score=True, n_jobs=-1, verbose=1)
    else:
        rf = RandomForestRegressor(n_estimators=400, max_features='sqrt',
                                    max_depth=25, min_samples_split=2, oob_score=True, n_jobs=-1, verbose=1)


    X_train, polygon_ids_train, y_train = readSITSData(train_file)
    # -- train a rf classifier
    start_train_time = time.time()
    rf.fit(X_train, y_train)
    #res_mat[2] = round(time.time() - start_train_time, 2)
    #print('Training time (s): ', res_mat[2, 0])

    # -- save the model
    joblib.dump(rf, model_file)
    print("Writing the model over")

def predict_rf(model_path, test_file, result_file, proba, feature, extrapolate, regression, n_channels, sizex, sizey):

    # -- Checking the extension
    assert result_file.split('.')[-1] == test_file.split('.')[-1], "ERR: requires similar extension"
    file_type = result_file.split('.')[-1]

    # -- Get the number of classes
    #n_classes = getNoClasses(model_path)

    # -- Read min max values
    minMaxVal_file = '.'.join(model_path.split('.')[0:-1])
    minMaxVal_file = minMaxVal_file + '_minMax.txt'
    if os.path.exists(minMaxVal_file):
        min_per, max_per = read_minMaxVal(minMaxVal_file)
    else:
        assert False, "ERR: min-max values needs to be stored during training"
    # -- Downloading
    if file_type == "csv":
        X_test, polygon_ids_test, y_test = readSITSData(test_file)
        X_test = addingfeat_reshape_data(X_test, feature, n_channels)
        X_test = normalizingData(X_test, min_per, max_per)
    elif file_type == "tif":
        # ---- Get image info about gps coordinates for origin plus size pixels
        image = gdal.Open(test_file, gdal.GA_ReadOnly)  # , NUM_THREADS=8
        geotransform = image.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        spacingX = geotransform[1]
        spacingY = geotransform[5]
        r, c = image.RasterYSize, image.RasterXSize
        out_raster_SRS = osr.SpatialReference()
        out_raster_SRS.ImportFromWkt(image.GetProjectionRef())

        # -- Set up the characteristics of the output image
        driver = gdal.GetDriverByName('GTiff')
        if regression == True:
            out_map_raster = driver.Create(result_file, c, r, 1, gdal.GDT_Float32)
        else:
            out_map_raster = driver.Create(result_file, c, r, 1, gdal.GDT_Byte)
        out_map_raster.SetGeoTransform([originX, spacingX, 0, originY, 0, spacingY])
        out_map_raster.SetProjection(out_raster_SRS.ExportToWkt())
        out_map_band = out_map_raster.GetRasterBand(1)

        if proba == True:
            result_conf_file = '.'.join(result_file.split('.')[0:-1]) + 'conf_map.tif'
            out_confmap_raster = driver.Create(result_conf_file, c, r, n_classes, gdal.GDT_Float32)
            out_confmap_raster.SetGeoTransform([originX, spacingX, 0, originY, 0, spacingY])
            out_confmap_raster.SetProjection(out_raster_SRS.ExportToWkt())

    # ---- Loading the model
    rf = joblib.load(model_path)


    def gps_2_image_xy(x, y):
        return (x - originX) / spacingX, (y - originY) / spacingY

    def gps_2_image_p(point):
        return gps_2_image_xy(point[0], point[1])

    # size_areaX = c # decrease the values if the tiff data cannot be in the memory, e.g. size_areaX = 10980, r =50 (get tiff BlockSize information for a nice setting)
    # size_areaY = r
    size_areaX = sizex  # decrease the values if the tiff data cannot be in the memory, e.g. size_areaX = 10980, r =50 (get tiff BlockSize information for a nice setting)
    size_areaY = sizey
    x_vec = list(range(int(c / size_areaX)))
    x_vec = [x * size_areaX for x in x_vec]
    y_vec = list(range(int(r / size_areaY)))
    y_vec = [y * size_areaY for y in y_vec]
    x_vec.append(c)
    y_vec.append(r)
    count = 0
    for x in range(len(x_vec) - 1):
        for y in range(len(y_vec) - 1):
            count += 1
            # print(f'{count} / {(len(x_vec)-1)*(len(y_vec)-1)} processed')
            xy_top_left = (x_vec[x], y_vec[y])
            xy_bottom_right = (x_vec[x + 1], y_vec[y + 1])
            # ---- now loading associated data
            xoff = xy_top_left[0]
            yoff = xy_top_left[1]
            xsize = xy_bottom_right[0] - xy_top_left[0]
            ysize = xy_bottom_right[1] - xy_top_left[1]
            X_test = image.ReadAsArray(xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize).astype(
                float)  # , gdal.GDT_Float32
            nodata = image.GetRasterBand(1).GetNoDataValue()
            # print(X_test.shape)
            X_test[X_test == nodata] = np.nan
            if np.all(np.isnan(X_test)):
                continue

            # print(type(X_test))
            # print(X_test)
            # ---- reshape the cube in a column vector
            X_test = X_test.transpose((1, 2, 0))
            sX = X_test.shape[0]
            sY = X_test.shape[1]
            X_test = X_test.reshape(X_test.shape[0] * X_test.shape[1], X_test.shape[2])

            if extrapolate == True:
                # x_indices = np.tile(np.arange(X_test.shape[1] // n_channels), n_channels)
                # for i in range(X_test.shape[0]):
                # 	y_values = X_test[i]
                # 	nan_mask = np.isnan(y_values)
                # 	non_nan_mask = ~nan_mask
                # 	if non_nan_mask.any():
                # 		y_interp = np.interp(x_indices, x_indices[non_nan_mask], y_values[non_nan_mask],
                # 							 left=y_values[non_nan_mask][0], right=y_values[non_nan_mask][-1])
                # 		y_values[nan_mask] = y_interp[nan_mask]
                # 		X_test[i] = y_values

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
            X_test[np.isnan(X_test)] = -9999
            # X_interp now contains the interpolated data
            # ---- pre-processing the data
            #X_test = addingfeat_reshape_data(X_test, feature, n_channels)

            #X_test = normalizingData(X_test, min_per, max_per)
            # -- prediction
            p_img = rf.predict(X_test)

            if regression == False:
                y_test = p_img.argmax(axis=1)
            y_test = p_img
            pred_array = y_test.reshape(sX, sY)

            out_map_band.WriteArray(pred_array, xoff=xoff, yoff=yoff)
            out_map_band.FlushCache()
            if proba == True:
                confpred_array = p_img.reshape(sX, sY, n_classes)
                for b in range(n_classes):
                    out_confmap_band = out_confmap_raster.GetRasterBand(b + 1)
                    out_confmap_band.WriteArray(confpred_array[:, :, b], xoff=xoff, yoff=yoff)
                out_confmap_band.FlushCache()


