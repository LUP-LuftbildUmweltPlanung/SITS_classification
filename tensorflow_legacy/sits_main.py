from run_archi import main as train_main
from write_output import main as write_main
from utils import *
import time
import glob
#from osgeo import gdal
start = time.time()
##########################################
##########PREPROCESS######################
##########################################
# Delay execution for 5 hours (in seconds)
#time.sleep(5 * 60 * 60)
##################refpoints shape to csv for force sampling ## drop_lst like ['CID','test'],

shape = r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\sbs\TrainingPoints_nDOM_for_VegetationHeight\bearbeitung\merged_raw_lower0height_2018_3035_90pct.shp"
outputcsv = r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\sbs\TrainingPoints_nDOM_for_VegetationHeight\bearbeitung\merged_raw_lower0height_2018_3035_90pct_csv.csv"
drop_lst = ['herkunft','datum']
response = 'hoehe'
#shape_to_forcecsv(shape,outputcsv,drop_lst,response)


###############force txts to csv EXTRAPOLATION included

response_path = r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\samples\response.txt" # has to be .txt
feature_path = r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\samples\features.txt" # has to be .txt
bands = 10
split_train = 0.9

#forcesample_tocsv(feature_path,response_path,bands,split_train)


##################stack band stacks
raster1 = r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\results_arnsberg\X0057_Y0047\2018-2019_001-365_HL_TSA_SEN2L_BLU_TSI.tif"
raster2 = r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\results_arnsberg\X0057_Y0047\2018-2019_001-365_HL_TSA_SEN2L_GRN_TSI.tif"
raster3 = r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\results_arnsberg\X0057_Y0047\2018-2019_001-365_HL_TSA_SEN2L_RED_TSI.tif"
raster4 = r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\results_arnsberg\X0057_Y0047\2018-2019_001-365_HL_TSA_SEN2L_NIR_TSI.tif"
raster5 = r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\results_arnsberg\X0057_Y0047\2018-2019_001-365_HL_TSA_SEN2L_SW1_TSI.tif"
raster6 = r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\results_arnsberg\X0057_Y0047\2018-2019_001-365_HL_TSA_SEN2L_SW2_TSI.tif"
raster7 = r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\results_arnsberg\X0057_Y0047\2018-2019_001-365_HL_TSA_SEN2L_RE1_TSI.tif"
raster8 = r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\results_arnsberg\X0057_Y0047\2018-2019_001-365_HL_TSA_SEN2L_RE2_TSI.tif"
raster9 = r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\results_arnsberg\X0057_Y0047\2018-2019_001-365_HL_TSA_SEN2L_RE3_TSI.tif"
raster10 = r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\results_arnsberg\X0057_Y0047\2018-2019_001-365_HL_TSA_SEN2L_BNR_TSI.tif"
rasters = [raster1, raster2, raster3, raster4,raster5,raster6, raster7, raster8, raster9, raster10]

#rasters = glob.glob(r"")

output = r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\results_arnsberg\X0057_Y0047\stack_2019.tif"
#stack_raster(rasters,output)


##########################################
##########TRAIN######################
##########################################
import tensorflow as tf
print("Available GPUs:", tf.config.list_physical_devices("GPU"))
#train csv without extension
sits_path_train = r"/uge_mount/FORCE/new_struc/process/result/uge_class/features_vgh_train"

#test_csv without extension
sits_path_test = r"/uge_mount/FORCE/new_struc/process/result/uge_class/features_vgh_test"

# regression task? otherwise classification
regression = True

n_epochs = 20
batch_size = 32
# path for results
res_path = r"/uge_mount/FORCE/new_struc/scripts/tempcnn/example"
# ---- Parameters to set
n_channels = 10  # -- NIR, R, G, ...
val_rate = 0.2

## class_label equal to classes, for regression just one
#class_label = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12']
#class_label = ['c0', 'c1']
class_label = ['c0']


feature = "SB_cont"
archi = "complexity" #"complexity" --> tempcnn (2 best), "rnn" --> gru (3 best)
noarchi = 6
norun = 0


train_main(sits_path_train, sits_path_test,regression, res_path, feature, archi, noarchi, norun, n_epochs, batch_size, class_label, n_channels, val_rate)


##########################################
##########PREDICTION###################### WITH POSSIBILITY TO EXTRAPOLATE
##########################################

model_path = r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\samples\results\Archi2\bestmodel-SB_cont-features_train-noarchi2-norun0.h5"
#test_file = r"E:\++++Promotion\SitsClassification\data\test\full_bands_100kpoints\Berlin_Mageburg\X0065_Y0044\stack.tif"
#result_file = r"E:\++++Promotion\SitsClassification\data\test\full_bands_100kpoints\Berlin_Mageburg\resultneu4.tif"

lst = glob.glob(r"E:\++++Promotion\SitsClassification\data\veg_height_notUrban\results_arnsberg\X*\stack*.tif")

for test_file in lst:
    print(f"classifying: {test_file}")
    result_file = test_file.replace(".tif","_tempcnn.tif")
    proba = False
    feature = "SB_cont"
    extrapolate = True
    regression = True
    n_channels = 10
    sizex = 100
    sizey = 100



    #write_main(model_path, test_file, result_file, proba, feature, extrapolate, regression, n_channels, sizex, sizey)


end = time.time()
print(end - start)


##########################################
##########GDAL WARP MERGE######################
##########################################

# files_to_mosaic = glob.glob(r'E:\++++Promotion\SitsClassification\data\veg_height_notUrban\results_arnsberg\X*\out\stack_2022_tempcnn.tif')
# print(files_to_mosaic)
# #files_to_mosaic = ["a.tif", "b.tif"] # However many you want.
# g = gdal.Warp(r'E:\++++Promotion\SitsClassification\data\veg_height_notUrban\results_arnsberg\2022_tempcnn.tif', files_to_mosaic, format="GTiff",
#               creationOptions=["COMPRESS=LZW", "TILED=YES","BIGTIFF=YES"]) # if you want
# g = None # Close file and flush to disk



# ###########################################
# ########RANDOM FOREST #####################
# ###########################################
# from rf import train_rf, predict_rf
#
#
# sits_path_train = r"E:\++++Promotion\SitsClassification\data\test\full_bands_100kpoints\features_train"
# regression = True
# res_path = "results"
# n_channels = 10
# feature = "SB_cont"
#
# train_rf(sits_path_train, regression, res_path, feature, n_channels)
#
# model_path = r"E:\++++Promotion\SitsClassification\temporalCNN-master\results\rf\test.joblib"
# #test_file = r"E:\++++Promotion\SitsClassification\data\test\full_bands_100kpoints\Berlin_Mageburg\X0065_Y0044\stack.tif"
# #result_file = r"E:\++++Promotion\SitsClassification\data\test\full_bands_100kpoints\Berlin_Mageburg\resultneu4.tif"
#
# lst = glob.glob(r"E:\++++Promotion\SitsClassification\data\test\full_bands_100kpoints\Potsdam\X*\stack.tif")
#
# for test_file in lst:
#     print(f"classifying: {test_file}")
#     result_file = test_file.replace("stack.tif","stack_class_rf.tif")
#     proba = False
#     feature = "SB_cont"
#     extrapolate = True
#     regression = True
#     n_channels = 10
#     sizex = 100
#     sizey = 100
#
#     predict_rf(model_path, test_file, result_file, proba, feature, extrapolate, regression, n_channels, sizex, sizey)

