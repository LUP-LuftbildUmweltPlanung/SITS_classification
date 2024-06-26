import sys
sys.path.append("./models")

import numpy as np
import torch

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
def prepare_dataset(args):
    assert args['response'] in ["regression_sigmoid", "regression_relu", "classification"]

    if args['response'].startswith("regression"):
        args['classes_lst'] = [0]
    #ImbalancedDatasetSampler

    print("setting random seed to "+str(args['seed']))
    np.random.seed(args['seed'])
    if args['seed'] is not None:
        torch.random.manual_seed(args['seed'])

    ref_dataset = Dataset(root=args['data_root'], classes=args['classes_lst'], seed=args['seed'], response=args['response'], norm=args['norm_factor_features'], bands=args['order'], norm_response = args['norm_factor_response'])

    return ref_dataset

def collate_fn(batch, p, plotting):
    X_batch, y_batch, doy_batch = zip(*batch)

    # Apply augmentation with probability p to each item in the batch
    X_batch_augmented = []
    doy_batch_augmented = []
    for X, doy in zip(X_batch, doy_batch):
        X_aug, doy_aug = apply_augmentation(X, doy, p, plotting)
        X_batch_augmented.append(X_aug)
        doy_batch_augmented.append(doy_aug)

    X_padded = pad_sequence(X_batch_augmented, batch_first=True, padding_value=0)
    doy_padded = pad_sequence(doy_batch_augmented, batch_first=True, padding_value=0)
    y_padded = torch.stack(y_batch)

    return X_padded, y_padded, doy_padded

def train(trial,args_train,ref_dataset):
    # add the splitting part here
    if args_train["tune"]==True:
        new_args_tune = hyperparameter_tune(trial,args_train['model'])
        args_train.update(new_args_tune)
    else:
        new_args = hyperparameter_config(args_train['model'])
        args_train.update(new_args)

        os.makedirs(os.path.join(args_train['store'], args_train['model']), exist_ok=True)
        try:
            shutil.copy(f'{Path(args_train["data_root"]).parent.parent}/preprocess_settings.json',f'{os.path.join(args_train["store"], args_train["model"])}/preprocess_settings.json')
        except:
            print("Couldnt Copy preprocess_settings.json")
        hyperparmeter_path = os.path.join(args_train['store'], args_train['model'], "hyperparameters.json")
        with open(hyperparmeter_path, 'w') as file:
            json.dump(args_train, file, indent=4)

    #selected_size = int((args['partition'] / 100.0) * len(ref_dataset))
    torch.manual_seed(args_train['seed'])
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

    traindataloader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=RandomSampler(train_dataset),
                                                  batch_size=args_train['batchsize'], num_workers=args_train['workers'],
                                                  collate_fn=lambda batch: collate_fn(batch, p, plotting))

    validdataloader = torch.utils.data.DataLoader(dataset=valid_dataset, sampler=SequentialSampler(valid_dataset),
                                                  batch_size=args_train['batchsize'], num_workers=args_train['workers'],
                                                  collate_fn=lambda batch: collate_fn(batch, p=0, plotting=None))


    print(f"Training Sample Size: {len(traindataloader.dataset)}")
    print(f"Validation Sample Size: {len(validdataloader.dataset)}")

    if args_train['model'] in ["transformer"]:
        args_train['seqlength'] = 366 * args_train['years'] #2562#
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
        response = args_train['response']
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


def train_init(args_train, preprocess_params, path_params):
    args_train["workers"] = 10  # number of CPU workers to load the next batch
    args_train["data_root"] = f'{path_params["proc_folder"]}/_SITSrefdata/{preprocess_params["project_name"]}/sepfiles/train/' # folder with CSV or cached NPY folder
    args_train["store"] = f'{path_params["proc_folder"]}/_SITSModels/{preprocess_params["project_name"]}/'  # Store Model Data Path
    # create hw_monitor output dir if it doesn't exist
    Path(args_train['store'] + '/' + args_train['model'] + '/hw_monitor').mkdir(parents=True, exist_ok=True)
    drive_name = ["sdb1"]

    if args_train['tune'] == True:
        print("tuning")

        hw_init_logs_file = args_train['store'] + '/' + args_train['model'] + '/hw_monitor/hw_monitor_init_' +args_train['study_name']+'.csv'
        hw_tune_logs_file = args_train['store'] + '/' + args_train['model'] + '/hw_monitor/hw_monitor_tune_' +args_train['study_name']+'.csv'

        # Instantiate monitor with a 0.1-second delay between updates
        hwmon_i = HWMonitor(0.1,hw_init_logs_file,drive_name)
        hwmon_i.start()
        ref_dataset = prepare_dataset(args_train)
        hwmon_i.stop()

        os.makedirs(args_train['store'] + 'optuna', exist_ok=True)
        storage_path = args_train['store'] + '/optuna/storage'
        print(storage_path)
        storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(storage_path))
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.CmaEsSampler(),
                                    pruner=optuna.pruners.MedianPruner(), storage=storage, study_name=args_train['study_name'])

        # Instantiate monitor with a 1-second delay between updates
        hwmon = HWMonitor(1,hw_tune_logs_file,drive_name)
        hwmon.start()
        study.optimize(lambda trial: train(trial, args_train, ref_dataset), n_trials=100)
        hwmon.stop()

        print(f"Best value: {study.best_value} (params: {study.best_params})")

    else:
        hw_init_logs_file = args_train['store'] + '/' + args_train['model'] + '/hw_monitor/hw_monitor_init.csv'
        hw_train_logs_file = args_train['store'] + '/' + args_train['model'] + '/hw_monitor/hw_monitor_train.csv'

        # Instantiate monitor with a 0.drive_name1-second delay between updates
        hwmon_i = HWMonitor(0.1,hw_init_logs_file,drive_name)
        hwmon_i.start()
        hwmon_i.start_averaging()

        ref_dataset = prepare_dataset(args_train)

        hwmon_i.stop_averaging()
        avgs = hwmon_i.get_averages()
        squeezed = squeeze_hw_info(avgs)
        mean_data = {key: round(value, 1) for key, value in squeezed.items() if "mean" in key}
        print(f"##################\nMean Values Hardware Monitoring (Preparing Data):\n{mean_data}\n##################")
        hwmon_i.stop()



        # Instantiate monitor with a 1-second delay between updates
        hwmon = HWMonitor(1,hw_train_logs_file,drive_name)
        hwmon.start()
        hwmon.start_averaging()

        train(None, args_train, ref_dataset)

        hwmon.stop_averaging()
        avgs = hwmon.get_averages()
        squeezed = squeeze_hw_info(avgs)
        mean_data = {key: round(value, 1) for key, value in squeezed.items() if "mean" in key}
        print(f"Mean Values Hardware Monitoring (Training Model):\n{mean_data}\n##############################")

        hwmon.stop()



