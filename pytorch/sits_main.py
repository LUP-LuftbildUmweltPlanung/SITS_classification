import os
import time
import sys
sys.path.append("..")
from train import train_init
from predict import predict_init
from predict import predict_csv

args_train = {
    'batchsize': 256,  # batch size
    'epochs': 60,  # number of training epochs
    'workers': 10,  # number of CPU workers to load the next batch
    'data_root': '/uge_mount/FORCE/new_struc/process/result/_SITSrefdata/uge_class_gv_5dint_1year_nodatafilled_testDOY/sepfiles/test/', # folder with CSV or cached NPY folder
    'store': '/uge_mount/FORCE/new_struc/process/result/_SITSrefdata/test/',  # store run logger results
    'valid_every_n_epochs': 5,  # skip some valid epochs for faster overall training
    'checkpoint_every_n_epochs': 5,  # save checkpoints during training
    'seed': 0,  # seed for batching and weight initialization
    'ref_split': 0.8, # split ratio for training, other part is validation
    'model': "transformer",  # "tempcnn","rnn","msresnet","transformer"
    'tune': False, #Hyperparameter Tune?
    'study_name':"tempcnn_rmse_tcd",
    'response': "regression_relu",  # "classification", "regression_relu" Use ReLU for 0 to infinity output, "regression_sigmoid" Use sigmoid for 0 to 1 output
    'classes_lst': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    'order': ["BLU", "GRN", "RED", "NIR", "SW1", "SW2", "RE1", "RE2", "RE3", "BNR"],
    'normalizing_factor': 1e-4,
    'years': 3, #time series years for doy max sequence length
}

args_predict = {
    'model_path' : '/uge_mount/FORCE/new_struc/process/result/_SITSModels/phd_vgh_tss_1years_EQUAL/transformer/model_e10.pth',
    #'reference_folder': '/uge_mount/FORCE/new_struc/process/result/_SITSrefdata/phd_vgh_tss_1years_EQUAL',
    'reference_folder' : None,
    'aois' : '/uge_mount/FORCE/new_struc/process/temp/leipzig_prechrist_gv_newapproach/FORCE/aoi_2018.shp/tiles_tss/X0067_Y0047/', # aois can be path or list. Path for Force Tile folder or list for process structure and shapefiles must be correct
    #'aois': None,
    'chunksize' : 10000,
}



if __name__ == '__main__':
    start_t = time.time()

    ###TRAIN
    train_init(args_train)

    ###PREDICT
    predict_init(args_predict, None)


    end_t = time.time()
    print(f"SITS process took {(end_t-start_t)/60} minutes")



