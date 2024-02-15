import argparse

import sys

from train_optuna import train, prepare_dataset
import optuna

from hw_monitor import HWMonitor


args = {
    'batchsize': 256,  # batch size
    'epochs': 5,#150,  # number of training epochs
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

    new_args = hyperparameter_config(args['model'])
    args.update(new_args)
    #print("args:")
    #print(args)

    ref_dataset = prepare_dataset(args)

    if tune:

        print("tuning")
        storage_path = args['data_root']+'optuna/storage'
        print(storage_path)
        storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(storage_path))
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(),storage=storage)

        # Instantiate monitor with a 1-second delay between updates
        hw_logs_path = args['store']+study.study_name+'.csv'
        hwmon = HWMonitor(2,hw_logs_path)
        # start monitoring
        hwmon.start()

        study.optimize(lambda trial: train(trial, args, ref_dataset), n_trials=100)
        print(f"Best value: {study.best_value} (params: {study.best_params})")

        hwmon.stop()

    else:
        print('training')
        train(None,args,ref_dataset)
