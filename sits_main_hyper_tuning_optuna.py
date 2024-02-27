import argparse

import sys
from pathlib import Path
#import time

from train_optuna import train, prepare_dataset
import optuna

from hw_monitor import HWMonitor, disk_info


args = {
    'batchsize': 256,  # batch size
    'epochs': 1,#150,  # number of training epochs
    'workers': 10,  # number of CPU workers to load the next batch
    #'data_root': '/uge_mount/data_test/',
    'data_root': '../tmp_data/',
    #'store': '/uge_mount/results/',  # store run logger results
    'store': '../tmp_data/results/',  # store run logger results
    'valid_every_n_epochs': 1,  # skip some valid epochs for faster overall training
    'checkpoint_every_n_epochs': 2,  # save checkpoints during training
    'seed': 0,  # seed for batching and weight initialization
    'partition': 25,  # partition of whole reference data
    'ref_on': "reference", # folder name within 'data_root' for reference data
    'ref_split': 0.8, # split ratio for training, other part is validation
    'model': "transformer",  # "tempcnn","rnn","msresnet","transformer"
    'response': "regression",  # "regression", "classification"
    'classes_lst': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
}


hw_args = {
    'disks_to_monitor': ['nvme0n1p2','nvme0n1p4','nvme0n1p5'],
    'hw_logs_dir': args['store'] + '/hw_monitor',
    # The following are used in case of train and predict, but not for tune! For tuning, the name is generated from the study name.
    'hw_init_logs_file_name': 'hw_monitor_init.csv',
    'hw_train_logs_file_name': 'hw_monitor_train.csv',
    'hw_predict_logs_file': 'hw_monitor_predict.csv',
}


def hyperparameter_config(model):
    assert model in ["tempcnn", "transformer", "rnn", "msresnet"]
    if model == "tempcnn":
        return {
            "model": "tempcnn",
            "kernel_size": 5,
            "hidden_dims": 64,
            "dropout": 0.5,
            "weight_decay": 1e-6,
            "learning_rate": 0.001
        }
    elif model == "transformer":
        return {
            "model": "transformer",
            "hidden_dims": 128,
            "n_heads": 4,
            "n_layers": 3,
            "learning_rate": 0.00255410,
            "dropout": 0,
            "weight_decay": 0.000213,
            "warmup": 1000
        }
    elif model == "rnn":
        return {
            "model": "rnn",
            "num_layers": 4,
            "hidden_dims": 32,
            "learning_rate": 0.010489,
            "dropout": 0.710883,
            "weight_decay": 0.000371,
            "bidirectional": True
        }
    elif model == "msresnet":
        return {
            "model": "msresnet",
            "hidden_dims": 32,
            "weight_decay": 0.000059,
            "learning_rate": 0.000657
        }
    else:
        raise ValueError("Invalid model")




if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Run the training or hyperparameter tuning.")

    parser.add_argument("--tune", action=argparse.BooleanOptionalAction, default=False,
                        help="To tune hyperparameters.")

    p_args = parser.parse_args()
    tune = p_args.tune

    #print(tune)

    # get disk info
    disk_info()

    # create hw_monitor output dir if it doesn't exist
    Path(hw_args['hw_logs_dir']).mkdir(parents=True, exist_ok=True)

    # create hw log file name for initial reading from the disk
    if tune:
        #print("tuning")
        storage_path = args['data_root']+'optuna/storage'
        print(storage_path)

        storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(storage_path))
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(),storage=storage)

        hw_init_logs_file = hw_args['hw_logs_dir'] + '/hw_monitor_init_' +study.study_name+'.csv'
        hw_tune_logs_file = hw_args['hw_logs_dir'] + '/hw_monitor_tune_' +study.study_name+'.csv'

    else:
        hw_init_logs_file = hw_args['hw_logs_dir'] + '/' + hw_args['hw_init_logs_file_name']
        hw_train_logs_file = hw_args['hw_logs_dir'] + '/' + hw_args['hw_train_logs_file_name']
        hw_predict_logs_file = hw_args['hw_logs_dir'] + '/' + hw_args['hw_predict_logs_file']

    new_args = hyperparameter_config(args['model'])
    args.update(new_args)

    # Instantiate monitor with a 0.1-second delay between updates
    hwmon_i = HWMonitor(0.1,hw_init_logs_file,hw_args['disks_to_monitor'])
    # start monitoring
    hwmon_i.start()
    ref_dataset = prepare_dataset(args)
    # stop monitoring
    hwmon_i.stop()


    #sys.exit()


    if tune:

        print("tuning")
        #storage_path = args['data_root']+'optuna/storage'
        #print(storage_path)

        #storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(storage_path))
        #study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(),storage=storage)

        # Instantiate monitor with a 1-second delay between updates
        hwmon = HWMonitor(1,hw_tune_logs_file,hw_args['disks_to_monitor'])
        # start monitoring
        hwmon.start()

        study.optimize(lambda trial: train(trial, args, ref_dataset, hwmon), n_trials=100)

        print(f"Best value: {study.best_value} (params: {study.best_params})")

    else:
        print('training')

        # Instantiate monitor with a 1-second delay between updates
        hwmon = HWMonitor(1,hw_train_logs_file,hw_args['disks_to_monitor'])
        # start monitoring
        hwmon.start()

        train(None,args,ref_dataset)

    # stop monitoring
    hwmon.stop()