# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:30:26 2023

@author: benjaminstoeckigt
"""


import sys
sys.path.append("./models")

import numpy as np
import torch
import random

from pathlib import Path

from pytorch.models.TransformerEncoder import TransformerEncoder
from pytorch.models.multi_scale_resnet import MSResNet
from pytorch.models.TempCNN import TempCNN
from pytorch.models.rnn import RNN
from pytorch.utils.Dataset import Dataset
from pytorch.utils.trainer import Trainer
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from pytorch.utils.logger import Logger
from pytorch.utils.scheduled_optimizer import ScheduledOptim
from pytorch.utils.hw_monitor import HWMonitor, disk_info, squeeze_hw_info
import torch.optim as optim
import os, json
import shutil
from config_hyperparameter import hyperparameter_config, hyperparameter_tune
import optuna
from torch.nn.utils.rnn import pad_sequence
from pytorch.utils.augmentation import time_warp, plot, apply_scaling, apply_augmentation

def train_init(args_train, preprocess_params):

    args_train["time_range"] = preprocess_params["time_range"] # relevant for relative yearls doy seperation for augmentations
    args_train["workers"] = 10  # number of CPU workers to load the next batch

    args_train["data_root"] = f'{preprocess_params["process_folder"]}/results/_SITSrefdata/{preprocess_params["project_name"]}/sepfiles/train/' # folder with CSV or cached NPY folder
    args_train["store"] = f'{preprocess_params["process_folder"]}/results/_SITSModels/{preprocess_params["project_name"]}/'  # Store Model Data Path

    args_train["thermal_time"] = preprocess_params["thermal_time"]
    # create hw_monitor output dir if it doesn't exist
    Path(args_train['store'] + '/' + args_train['model'] + '/hw_monitor').mkdir(parents=True, exist_ok=True)
    args_train["sdb1"] = ["sdb1"]

    hw_train_logs_file = args_train['store'] + '/' + args_train['model'] + '/hw_monitor/hw_monitor_train.csv'
    # Instantiate monitor with a 1-second delay between updates
    hwmon = HWMonitor(1,hw_train_logs_file,args_train["sdb1"])
    hwmon.start()
    hwmon.start_averaging()

    if args_train['tune'] == True:
        print("hyperparameter tuning ...")
        os.makedirs(args_train['store'] + args_train['model'] + '/optuna', exist_ok=True)
        storage_path = args_train['store'] + args_train['model'] + '/optuna/storage'
        print(storage_path)
        storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(storage_path))
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.CmaEsSampler(),pruner=optuna.pruners.MedianPruner(), storage=storage,
                                    study_name=args_train['study_name'])
        study.optimize(lambda trial: train(trial, args_train), n_trials=100)
        print(f"Best value: {study.best_value} (params: {study.best_params})")
    else:
        train(None, args_train,)

    hwmon.stop_averaging()
    avgs = hwmon.get_averages()
    squeezed = squeeze_hw_info(avgs)
    mean_data = {key: round(value, 1) for key, value in squeezed.items() if "mean" in key}
    print(f"Mean Values Hardware Monitoring (Training Model):\n{mean_data}\n##############################")

    hwmon.stop()

def train(trial,args_train):

    if args_train['seed'] is not None:
        print("setting random seed for cuda, numpy and random to " + str(args_train['seed']))
        os.environ['PYTHONHASHSEED'] = str(args_train['seed'])
        random.seed(args_train['seed'])
        torch.manual_seed(args_train['seed'])
        torch.cuda.manual_seed(args_train['seed'])
        np.random.seed(args_train['seed'])
        #torch.random.manual_seed(args_train['seed'])

    hw_init_logs_file = args_train['store'] + '/' + args_train['model'] + '/hw_monitor/hw_monitor_init.csv'
    # Instantiate monitor with a 0.drive_name1-second delay between updates
    hwmon_i = HWMonitor(0.1, hw_init_logs_file, args_train["sdb1"])
    hwmon_i.start()
    hwmon_i.start_averaging()

    # add the splitting part here
    if args_train["tune"] == True:
        new_args_tune = hyperparameter_tune(trial, args_train['model'])
        args_train.update(new_args_tune)
        ref_dataset = prepare_dataset(args_train)
    else:
        new_args = hyperparameter_config(args_train['model'])
        args_train.update(new_args)
        ref_dataset = prepare_dataset(args_train)

        os.makedirs(os.path.join(args_train['store'], args_train['model']), exist_ok=True)
        try:
            shutil.copy(f'{Path(args_train["data_root"]).parent.parent}/preprocess_settings.json',f'{os.path.join(args_train["store"], args_train["model"])}/preprocess_settings.json')
        except:
            print("Couldnt Copy preprocess_settings.json")
        hyperparmeter_path = os.path.join(args_train['store'], args_train['model'], "hyperparameters.json")
        with open(hyperparmeter_path, 'w') as file:
            json.dump(args_train, file, indent=4)


    hwmon_i.stop_averaging()
    avgs = hwmon_i.get_averages()
    squeezed = squeeze_hw_info(avgs)
    mean_data = {key: round(value, 1) for key, value in squeezed.items() if "mean" in key}
    print(f"##################\nMean Values Hardware Monitoring (Preparing Data):\n{mean_data}\n##################")
    hwmon_i.stop()

    #selected_size = int((args['partition'] / 100.0) * len(ref_dataset))
    selected_size = int(args_train['partition']*len(ref_dataset)/100.0)
    print("selected_size="+str(selected_size))

    remaining_size = len(ref_dataset) - selected_size
    selected_dataset, _ = torch.utils.data.random_split(ref_dataset, [selected_size, remaining_size])
    print(f"Selected {args_train['partition']}% of the dataset: {len(selected_dataset)} samples from a total of {len(ref_dataset)} samples.")

    ref_split = args_train['ref_split']
    #train_size = int(args['ref_split'] * len(selected_dataset))
    train_size = int(ref_split * len(selected_dataset))
    valid_size = len(selected_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(selected_dataset, [train_size, valid_size])

    p = args_train['augmentation']
    plotting = args_train['augmentation_plot']
    time_range = args_train['time_range'][1]

    include_thermal = args_train['thermal_time'] is not None

    traindataloader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=RandomSampler(train_dataset),
                                                  batch_size=args_train['batchsize'], num_workers=args_train['workers'],
                                                  collate_fn=lambda batch: collate_fn(batch, p, plotting, time_range, include_thermal))

    validdataloader = torch.utils.data.DataLoader(dataset=valid_dataset, sampler=SequentialSampler(valid_dataset),
                                                  batch_size=args_train['batchsize'], num_workers=args_train['workers'],
                                                  collate_fn=lambda batch: collate_fn_notransform(batch, include_thermal, p=0, plotting=None))



    print(f"Training Sample Size: {len(traindataloader.dataset)}")
    print(f"Validation Sample Size: {len(validdataloader.dataset)}")

    if args_train['model'] in ["transformer"]:
        args_train['seqlength'] = args_train['max_seq_length']
    elif args_train['model'] in ["rnn", "msresnet","tempcnn"]:
        args_train['seqlength'] = traindataloader.dataset.dataset.dataset.sequencelength
    # OPTUNA: this is the build_model_custom(trial)
    #model = getModel(args)
    args_train['nclasses'] = traindataloader.dataset.dataset.dataset.nclasses
    args_train['input_dims'] = traindataloader.dataset.dataset.dataset.ndims
    print(f"Exemplary Sequence Length: {traindataloader.dataset.dataset.dataset.sequencelength}")
    print(f"Maximum DOY Sequence Length: {args_train['seqlength']}")
    print(f"Input Dims: {args_train['input_dims']}")
    print(f"Prediction Classes: {len(args_train['classes_lst'])}")
    print(f"Data Augmentation: {p * 100} % Training Data will be augmented (Single, Double or Triple (30/30/30) of Annual Scaling / DOY Day Shifting / Zero Out")
    if include_thermal:
        print(f"Applying Transformer Model with Thermal Positional Encoding!\n-> GDD Path:{args_train['thermal_time']}")
    else:
        print("Applying Transformer Model with Calendar Positional Encoding!")

    model = getModel(args_train)

    store = os.path.join(args_train['store'],args_train['model'])

    logger = Logger(columns=["accuracy"], modes=["train", "valid"], rootpath=store)

    if args_train['model'] in ["transformer"]:
        optimizer = ScheduledOptim(
            optim.Adam(
                filter(lambda x: x.requires_grad, model.parameters()),
                betas=(0.9, 0.98), eps=1e-09, weight_decay=args_train['weight_decay']),
            model.d_model, args_train['warmup'])
    elif args_train['model'] in ["rnn", "msresnet","tempcnn"]:
        optimizer = optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args_train['weight_decay'], lr=args_train['learning_rate'])
    else:
        raise ValueError(args_train['model'] + "no valid model. either 'rnn', 'msresnet', 'transformer', 'tempcnn'")

    config = dict(
        epochs=args_train['epochs'],
        learning_rate=args_train['learning_rate'],
        store=store,
        checkpoint_every_n_epochs=args_train['checkpoint_every_n_epochs'],
        valid_every_n_epochs=args_train['valid_every_n_epochs'],
        logger=logger,
        optimizer=optimizer,
        response=args_train['response'],
        norm_factor_response=args_train['norm_factor_response']
    )
    trainer = Trainer(trial,model,traindataloader,validdataloader,**config)
    logger = trainer.fit()

    validation_metrics = logger.get_data()[logger.get_data()['mode'] == 'valid']
    if config['response'] == 'classification':
        return validation_metrics['accuracy'].max()
    else:
        return validation_metrics['rmse'].min()
    pass

def getModel(args):

    if args['model'] == "rnn":
        model = RNN(input_dim=args['input_dims'], nclasses=args['nclasses'], hidden_dims=args["hidden_dims"],
                              num_rnn_layers=args["num_layers"], dropout=args["dropout"], bidirectional=True, response = args['response'])
    if args['model'] == "msresnet":
        model = MSResNet(input_channel=args['input_dims'], layers=[1, 1, 1, 1], num_classes=args['nclasses'], hidden_dims=args["hidden_dims"], response = args['response'])

    if args['model'] == "tempcnn":
        model = TempCNN(input_dim=args['input_dims'], nclasses=args['nclasses'], sequence_length=args['seqlength'], hidden_dims = args["hidden_dims"], dropout=args["dropout"], kernel_size = args['kernel_size'], response = args['response'])

    elif args['model'] == "transformer":
        len_max_seq = args['seqlength']
        d_inner = args["hidden_dims"]*4
        model = TransformerEncoder(in_channels=args['input_dims'], len_max_seq=len_max_seq,
            d_word_vec=args["hidden_dims"], d_model=args["hidden_dims"], d_inner=d_inner,
            n_layers=args["n_layers"], n_head=args['n_heads'], d_k=args["hidden_dims"]//args['n_heads'], d_v=args["hidden_dims"]//args['n_heads'],
            dropout=args["dropout"], nclasses=args['nclasses'], response = args['response'])

    if torch.cuda.is_available():
        model = model.cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("initialized {} model ({} parameters)".format(args['model'], pytorch_total_params))

    return model

def prepare_dataset(args):
    assert args['response'] in ["regression_sigmoid", "regression", "regression_relu", "classification"]

    if args['response'].startswith("regression"):
        args['classes_lst'] = [0]
    #ImbalancedDatasetSampler

    ref_dataset = Dataset(root=args['data_root'], classes=args['classes_lst'], seed=args['seed'], response=args['response'],
                          norm=args['norm_factor_features'], norm_response = args['norm_factor_response'], thermal = args["thermal_time"])

    return ref_dataset

def collate_fn(batch, p, plotting, time_range, include_thermal):

    X_batch, y_batch, doy_batch, thermal_batch = zip(*batch)
    # Apply augmentation with probability p to each item in the batch
    thermal_batch_augmented = []
    X_batch_augmented = []
    doy_batch_augmented = []

    # Check if thermal_batch is None, if so, create a list of None values with the same length as X_batch
    if thermal_batch is None:
        thermal_batch = [None] * len(X_batch)

    for X, doy, thermal in zip(X_batch, doy_batch, thermal_batch):
        X_aug, doy_aug, thermal_aug = apply_augmentation(X, doy, thermal, p, plotting, time_range)
        if include_thermal:
            thermal_batch_augmented.append(thermal_aug)
        X_batch_augmented.append(X_aug)
        doy_batch_augmented.append(doy_aug)


    X_padded = pad_sequence(X_batch_augmented, batch_first=True, padding_value=0)
    doy_padded = pad_sequence(doy_batch_augmented, batch_first=True, padding_value=0)
    y_padded = torch.stack(y_batch)

    if include_thermal:
        thermal_padded = pad_sequence(thermal_batch_augmented, batch_first=True, padding_value=0)
        return X_padded, y_padded, doy_padded, thermal_padded
    else:
        return X_padded, y_padded, doy_padded, None

def collate_fn_notransform(batch, include_thermal, p, plotting):

    X_batch, y_batch, doy_batch, thermal_batch = zip(*batch)

    X_padded = pad_sequence(X_batch, batch_first=True, padding_value=0)
    doy_padded = pad_sequence(doy_batch, batch_first=True, padding_value=0)
    y_padded = torch.stack(y_batch)

    if include_thermal:
        thermal_padded = pad_sequence(thermal_batch, batch_first=True, padding_value=0)
        return X_padded, y_padded, doy_padded, thermal_padded
    else:
        return X_padded, y_padded, doy_padded, None


