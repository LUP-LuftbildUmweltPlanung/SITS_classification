import sys
sys.path.append("./models")

from pathlib import Path

import numpy as np
import torch

from pytorch.models.TransformerEncoder import TransformerEncoder
from pytorch.models.multi_scale_resnet import MSResNet
from pytorch.models.TempCNN import TempCNN
from pytorch.models.rnn import RNN
from pytorch.utils.Dataset import Dataset
from pytorch.utils.trainer import Trainer
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from pytorch.utils.logger import Logger
from pytorch.utils.hw_monitor import HWMonitor, disk_info
from pytorch.utils.scheduled_optimizer import ScheduledOptim
import torch.optim as optim
import os, json
from hyperparameter_config import hyperparameter_config, hyperparameter_tune
import optuna

def prepare_dataset(args):
    assert args['response'] in ["regression_sigmoid", "regression_relu", "classification"]

    if args['response'].startswith("regression"):
        args['classes_lst'] = [0]
    #ImbalancedDatasetSampler

    print("setting random seed to "+str(args['seed']))
    np.random.seed(args['seed'])
    if args['seed'] is not None:
        torch.random.manual_seed(args['seed'])

    ref_dataset = Dataset(root=args['data_root'], classes=args['classes_lst'], seed=args['seed'], response=args['response'], padding=args['padding_value'], norm=args['normalizing_factor'], bands=args['order'])

    return ref_dataset

#def train(trial,args_train,ref_dataset):
def train(trial,args_train,ref_dataset,hwm):
    # add the splitting part here
    if args_train["tune"]==True:
        new_args_tune = hyperparameter_tune(trial,args_train['model'])
        args_train.update(new_args_tune)
    else:
        new_args = hyperparameter_config(args_train['model'])
        args_train.update(new_args)

        os.makedirs(os.path.join(args_train['store'], args_train['model']), exist_ok=True)
        hyperparmeter_path = os.path.join(args_train['store'], args_train['model'], "hyperparameters.json")
        with open(hyperparmeter_path, 'w') as file:
            json.dump(args_train, file, indent=4)

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

    traindataloader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=RandomSampler(train_dataset),
                                                  batch_size=args_train['batchsize'], num_workers=args_train['workers'])

    validdataloader = torch.utils.data.DataLoader(dataset=valid_dataset, sampler=SequentialSampler(valid_dataset),
                                                 batch_size=args_train['batchsize'], num_workers=args_train['workers'])
    print(f"Training Sample Size: {len(traindataloader.dataset)}")
    print(f"Validation Sample Size: {len(validdataloader.dataset)}")

    args_train['nclasses'] = traindataloader.dataset.dataset.dataset.nclasses
    args_train['seqlength'] = traindataloader.dataset.dataset.dataset.sequencelength
    args_train['input_dims'] = traindataloader.dataset.dataset.dataset.ndims
    print(f"Sequence Length: {args_train['seqlength']}")
    print(f"Input Dims: {args_train['input_dims']}")
    print(f"Prediction Classes: {args_train['nclasses']}")

    # OPTUNA: this is the build_model_custom(trial)
    #model = getModel(args)
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
        response = args_train['response'],
        hwmonitor = hwm
    )

    trainer = Trainer(trial,model,traindataloader,validdataloader,**config)
    hwm.start_averaging()
    logger = trainer.fit()
    hwm.stop_averaging()
    avgs = hwm.get_averages()
    print(avgs)

    # stores all stored values in the rootpath of the logger
    logger.save()

    #pth = store+"/npy/confusion_matrix_{epoch}.npy".format(epoch = args[epochs)
    #parse_run(store, args['classmapping'], outdir=store)
    #pass
    if config['response'] == 'classification':
        return logger.get_data().iloc[-1]['accuracy']
    else:
        return logger.get_data().iloc[-1]['rmse']


    pass

def getModel(args):

    if args['model'] == "rnn":
        model = RNN(input_dim=args['input_dims'], nclasses=args['nclasses'], hidden_dims=args["hidden_dims"],
                              num_rnn_layers=args["n_layers"], dropout=args["dropout"], bidirectional=True, response = args['response'])
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


def train_init(args_train):

    hw_args = args_train['hw_monitor']

    # create hw_monitor output dir if it doesn't exist
    Path(hw_args['hw_logs_dir']).mkdir(parents=True, exist_ok=True)

    if args_train['tune'] == True:
        print("tuning")

        hw_init_logs_file = hw_args['hw_logs_dir'] + '/hw_monitor_init_' +args_train['study_name']+'.csv'
        hw_tune_logs_file = hw_args['hw_logs_dir'] + '/hw_monitor_tune_' +args_train['study_name']+'.csv'

        # Instantiate monitor with a 0.1-second delay between updates
        hwmon_i = HWMonitor(0.1,hw_init_logs_file,hw_args['disks_to_monitor'])
        # start monitoring
        hwmon_i.start()
        ref_dataset = prepare_dataset(args_train)
        # stop monitoring
        hwmon_i.stop()

        os.makedirs(args_train['data_root'] + 'optuna', exist_ok=True)
        storage_path = args_train['data_root'] + 'optuna/storage'
        print(storage_path)
        storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(storage_path))
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.CmaEsSampler(),
                                    pruner=optuna.pruners.MedianPruner(), storage=storage, study_name=args_train['study_name'])

        # Instantiate monitor with a 1-second delay between updates
        hwmon = HWMonitor(1,hw_tune_logs_file,hw_args['disks_to_monitor'])
        # start monitoring
        hwmon.start()
        #study.optimize(lambda trial: train(trial, args_train, ref_dataset), n_trials=100)
        study.optimize(lambda trial: train(trial, args_train, ref_dataset, hwmon), n_trials=100)
        # stop monitoring
        hwmon.stop()
        print(f"Best value: {study.best_value} (params: {study.best_params})")

    else:
        print("training")

        hw_init_logs_file = hw_args['hw_logs_dir'] + '/' + hw_args['hw_init_logs_file_name']
        hw_train_logs_file = hw_args['hw_logs_dir'] + '/' + hw_args['hw_train_logs_file_name']

        # Instantiate monitor with a 0.1-second delay between updates
        hwmon_i = HWMonitor(0.1,hw_init_logs_file,hw_args['disks_to_monitor'])
        # start monitoring
        hwmon_i.start()
        ref_dataset = prepare_dataset(args_train)
        # stop monitoring
        hwmon_i.stop()

        # Instantiate monitor with a 1-second delay between updates
        hwmon = HWMonitor(1,hw_train_logs_file,hw_args['disks_to_monitor'])
        # start monitoring
        hwmon.start()
        train(None, args_train, ref_dataset)
        # stop monitoring
        hwmon.stop()