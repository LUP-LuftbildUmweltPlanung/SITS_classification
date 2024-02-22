import os
import time
import sys
sys.path.append("..")
from train import train_init
from predict import predict_init
from pytorch.utils.hw_monitor import disk_info



hw_args = {
    'disks_to_monitor': ['nvme0n1p2','nvme0n1p4','nvme0n1p5'],
    'hw_logs_dir': args['store'] + '/hw_monitor',
    # The following are used in case of train and predict, but not for tune! For tuning, the name is generated from the study name.
    'hw_init_logs_file_name': 'hw_monitor_init.csv',
    'hw_train_logs_file_name': 'hw_monitor_train.csv',
    'hw_predict_logs_file': 'hw_monitor_predict.csv',
}


args_train = {
    'batchsize': 256,  # batch size
    'epochs': 60,  # number of training epochs
    'workers': 10,  # number of CPU workers to load the next batch
    'data_root': '/uge_mount/FORCE/new_struc/process/result/_SITSrefdata/uge_class_training_tcd/sepfiles/train/', # folder with CSV or cached NPY folder
    'store': '/uge_mount/FORCE/new_struc/process/result/_SITSrefdata/uge_class_training_tcd/',  # store run logger results
    'valid_every_n_epochs': 5,  # skip some valid epochs for faster overall training
    'checkpoint_every_n_epochs': 5,  # save checkpoints during training
    'seed': 0,  # seed for batching and weight initialization
    'ref_split': 0.8, # split ratio for training, other part is validation
    'model': "tempcnn",  # "tempcnn","rnn","msresnet","transformer"
    'tune': False, #Hyperparameter Tune?
    'study_name':"tempcnn_rmse_tcd",
    'response': "regression",  # "regression", "classification"
    'classes_lst': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    'order': ["BLU", "GRN", "RED", "NIR", "SW1", "SW2", "RE1", "RE2", "RE3", "BNR"],
    'normalizing_factor': 1e-4,
    'padding_value': -1,
    'hw_monitor': hw_args
}

args_predict = {
    'model_path' : '/uge_mount/FORCE/new_struc/process/result/_SITSrefdata/gv_30ep_full_tempcnn_5dint_RELU_nodatafilled/transformer/model_e4.pth',
    'folder_path' : '/uge_mount/FORCE/new_struc/process/temp/leipzig_prechrist_gv_newapproach/FORCE/aoi_2018.shp/tiles_tss/X0067_Y0047/',
    'chunksize' : 10000,
    'hw_monitor': hw_args
}




if __name__ == '__main__':

    # get disk info
    disk_info()

    start_t = time.time()

    ###TRAIN
    train_init(args_train)

    ###PREDICT
    predict_init(args_predict)

    end_t = time.time()
    print(f"SITS process took {(end_t-start_t)/60} minutes")



