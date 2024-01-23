import sys
sys.path.append("./models")

import numpy as np
import torch
#import torch.nn as nn


from models.TransformerEncoder import TransformerEncoder
from models.multi_scale_resnet import MSResNet
from models.TempCNN import TempCNN
from models.rnn import RNN
from utils.Dataset import Dataset
from utils.trainer import Trainer
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from utils.logger import Logger
from utils.scheduled_optimizer import ScheduledOptim
import torch.optim as optim
import os

#from torchtnt.framework.callbacks import SystemResourcesMonitor
#from torchtnt.utils.loggers.logger import MetricLogger



def prepare_dataset(args):

    assert args['response'] in ["regression", "classification"]
    assert 0 <= args['partition'] <= 100, "Partition must be between 0 and 100"

    if args['response'] == "regression":
        args['classes_lst'] = [0]
    #ImbalancedDatasetSampler

    print("setting random seed to "+str(args['seed']))
    np.random.seed(args['seed'])
    if args['seed'] is not None:
        torch.random.manual_seed(args['seed'])

    ref_dataset = Dataset(root=args['data_root'], partition=args['ref_on'],
                          classes=args['classes_lst'], seed=args['seed'], response=args['response'])

    return ref_dataset



# OPTUNA: make this one into objective(trial)
#def train(args):
def train(trial,args,ref_dataset):

    # add the splitting part here

    #selected_size = int((args['partition'] / 100.0) * len(ref_dataset))
    selected_size = trial.suggest_int("partition", 25, 50, 25)# (name, low, high, step)
    selected_size = selected_size*len(ref_dataset)/100.0

    remaining_size = len(ref_dataset) - selected_size
    selected_dataset, _ = torch.utils.data.random_split(ref_dataset, [selected_size, remaining_size])
    print(f"Selected {args['partition']}% of the dataset: {len(selected_dataset)} samples from a total of {len(ref_dataset)} samples.")

    ref_split = trial.suggest_int("ref_split", 0.8, 0.9, 0.1)# (name, low, high, step)
    #train_size = int(args['ref_split'] * len(selected_dataset))
    train_size = int(ref_split * len(selected_dataset))
    valid_size = len(selected_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(selected_dataset, [train_size, valid_size])

    traindataloader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=RandomSampler(train_dataset),
                                                  batch_size=args['batchsize'], num_workers=args['workers'])

    validdataloader = torch.utils.data.DataLoader(dataset=valid_dataset, sampler=SequentialSampler(valid_dataset),
                                                 batch_size=args['batchsize'], num_workers=args['workers'])
    print(f"Training Sample Size: {len(traindataloader.dataset)}")
    print(f"Validation Sample Size: {len(validdataloader.dataset)}")

    # continue with the original part of training

    args['nclasses'] = traindataloader.dataset.dataset.dataset.nclasses
    args['seqlength'] = traindataloader.dataset.dataset.dataset.sequencelength
    args['input_dims'] = traindataloader.dataset.dataset.dataset.ndims
    print(f"Sequence Length: {args['seqlength']}")
    print(f"Input Dims: {args['input_dims']}")
    print(f"Prediction Classes: {args['nclasses']}")

    # OPTUNA: this is the build_model_custom(trial)
    #model = getModel(args)
    model = getModel(trial,args)

    store = os.path.join(args['store'],args['model'])

    logger = Logger(columns=["accuracy"], modes=["train", "valid"], rootpath=store)

    learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    weight_decay=trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)


    if args['model'] in ["transformer"]:
        optimizer = ScheduledOptim(
            optim.Adam(
                filter(lambda x: x.requires_grad, model.parameters()),
                betas=(0.9, 0.98), eps=1e-09, weight_decay=args['weight_decay']),
            model.d_model, args['warmup'])
    elif args['model'] in ["rnn", "msresnet","tempcnn"]:
        optimizer = optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            #betas=(0.9, 0.999), eps=1e-08, weight_decay=args['weight_decay'], lr=args['learning_rate'])
            betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, lr=learning_rate)
    else:
        raise ValueError(args['model'] + "no valid model. either 'rnn', 'msresnet', 'transformer', 'tempcnn'")

    config = dict(
        epochs=args['epochs'],
        #learning_rate=args['learning_rate'],
        learning_rate=learning_rate,
        store=store,
        checkpoint_every_n_epochs=args['checkpoint_every_n_epochs'],
        test_every_n_epochs=args['valid_every_n_epochs'],
        logger=logger,
        optimizer=optimizer,
        response = args['response']
    )


    #sysresmonitor_callback = SystemResourcesMonitor(loggers=MetricLogger)

    # OPTUNA: here we need train_and_evaluate(params, model, trial)
    #trainer = Trainer(model,traindataloader,validdataloader,**config)
    trainer = Trainer(trial,model,traindataloader,validdataloader,**config)
    logger = trainer.fit()
    #logger = trainer.fit(callbacks=[sysresmonitor_callback])

    # stores all stored values in the rootpath of the logger
    logger.save()

    #print("logger:")
    #print(logger.get_data())
    #print(type(logger.get_data()))
    #print("last result:")
    #print(logger.get_data().iloc[-1]['rmse'])

    #pth = store+"/npy/confusion_matrix_{epoch}.npy".format(epoch = args[epochs)
    #parse_run(store, args['classmapping'], outdir=store)

    #pass
    if config['response'] == 'classification':
        return logger.get_data().iloc[-1]['accuracy']
    else:
        return logger.get_data().iloc[-1]['rmse']

# OPTUNA: this should be build_model_custom
#def getModel(args):
def getModel(trial,args):

    hidden_dims = trial.suggest_int("hidden_dims", 128, 512, 128)# (name, low, high, step)

    mdl = trial.suggest_categorical("model", ["transformer", "rnn", "msresnet"])

    #if args['model'] == "rnn":
    if mdl == "rnn":
        dropout=trial.suggest_float("dropout", 0, 0.9, step=0.2)
        n_layers = trial.suggest_int("n_layers", 3, 6)
        model = RNN(input_dim=args['input_dims'], nclasses=args['nclasses'], hidden_dims=hidden_dims,
                              #num_rnn_layers=args['num_layers'],
                              num_rnn_layers=n_layers,
                              dropout=dropout, bidirectional=True, response = args['response'])

    #if args['model'] == "msresnet":
    if mdl == "msresnet":
        model = MSResNet(input_channel=args['input_dims'], layers=[1, 1, 1, 1], num_classes=args['nclasses'], hidden_dims=hidden_dims, response = args['response'])

    #if args['model'] == "tempcnn":
    if mdl == "tempcnn":
        dropout=trial.suggest_float("dropout", 0, 0.9, step=0.2)
        model = TempCNN(input_dim=args['input_dims'], nclasses=args['nclasses'], sequence_length=args['seqlength'], hidden_dims=hidden_dims,
                        dropout=dropout, kernel_size=args['kernel_size'], response = args['response'])

    #elif args['model'] == "transformer":
    elif mdl == "transformer":

        #hidden_dims = args['hidden_dims'] # 256
        n_heads = args['n_heads'] # 8
        #n_layers = args['n_layers'] # 6
        n_layers = trial.suggest_int("n_layers", 3, 6)
        len_max_seq = args['seqlength']
        dropout=trial.suggest_float("dropout", 0, 0.9, step=0.2)
        d_inner = hidden_dims*4
        model = TransformerEncoder(in_channels=args['input_dims'], len_max_seq=len_max_seq,
            d_word_vec=hidden_dims, d_model=hidden_dims, d_inner=d_inner,
            n_layers=n_layers, n_head=n_heads, d_k=hidden_dims//n_heads, d_v=hidden_dims//n_heads,
            dropout=dropout, nclasses=args['nclasses'], response = args['response'])

    if torch.cuda.is_available():
        model = model.cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("initialized {} model ({} parameters)".format(args['model'], pytorch_total_params))

    return model
