#import argparse

from train_optuna import train
import optuna


args = {
    'batchsize': 256,  # batch size
    'epochs': 30,#150,  # number of training epochs
    'workers': 10,  # number of CPU workers to load the next batch
    #'data_root': '/uge_mount/data_test/',
    'data_root': '../fast_data/',
    #'store': '/uge_mount/results/',  # store run logger results
    'store': '../fast_data/results/',  # store run logger results
    'valid_every_n_epochs': 1,  # skip some valid epochs for faster overall training
    'checkpoint_every_n_epochs': 2,  # save checkpoints during training
    'seed': 0,  # seed for batching and weight initialization
    'valid_on': "valid",
    'train_on': "train",
    'model': "transformer",  # "tempcnn","rnn","msresnet","transformer"
    'response': "regression",  # "regression", "classification"
    'classes_lst': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
}


def old_hyperparameter_config(model):#,weight_decay,dropout,learning_rate):
    assert model in ["tempcnn", "transformer", "rnn", "msresnet"]
    if model == "tempcnn":
        return {
            "model": "tempcnn",
            "kernel_size": 5,
            "hidden_dims": 64,
            "dropout": 0.5,#dropout,#
            "weight_decay": 1e-6,#weight_decay,#
            "learning_rate": 0.001#learning_rate,#
        }
    elif model == "transformer":
        return {
            "model": "transformer",
            "hidden_dims": 128,
            "n_heads": 4,
            "n_layers": 3,
            "learning_rate": 0.00255410,#learning_rate,#
            "dropout": 0,#dropout,#
            "weight_decay": 0.000213,#weight_decay,#
            "warmup": 1000
        }
    elif model == "rnn":
        return {
            "model": "rnn",
            "num_layers": 4,
            "hidden_dims": 32,
            "learning_rate": 0.010489,#learning_rate,#
            "dropout": 0.710883,#dropout,#
            "weight_decay": 0.000371,#weight_decay,#
            "bidirectional": True
        }
    elif model == "msresnet":
        return {
            "model": "msresnet",
            "hidden_dims": 32,
            "weight_decay": 0.000059,#weight_decay,#
            "learning_rate": 0.000657#learning_rate,#
        }
    else:
        raise ValueError("Invalid model")




if __name__ == '__main__':


    new_args = old_hyperparameter_config(args['model'])
    args.update(new_args)
    #args['batchsize'] = comm_args.batchsize
    print("args:")
    print(args)

    storage_path = args['data_root']+'optuna/storage'
    print(storage_path)
    storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(storage_path))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(),storage=storage)
    study.optimize(lambda trial: train(trial, args), n_trials=30)

    print(f"Best value: {study.best_value} (params: {study.best_params})")
