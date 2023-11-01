import sys
sys.path.append("./models")

import numpy as np
import torch

from models.TransformerEncoder import TransformerEncoder
from models.multi_scale_resnet import MSResNet
from models.TempCNN import TempCNN
from models.rnn import RNN
from datasets.BavarianCrops_Dataset import BavarianCropsDataset
from utils.trainer import Trainer
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from utils.texparser import parse_run
from utils.logger import Logger
from utils.scheduled_optimizer import ScheduledOptim
import torch.optim as optim
import os


def prepare_dataset(args):


    #ImbalancedDatasetSampler

    valid_dataset_list = BavarianCropsDataset(root=args['data_root'], partition=args['valid_on'],
                                                classes=args['classes_lst'], seed=args['seed'], response=args['response'])


    train_dataset_list = BavarianCropsDataset(root=args['data_root'], partition=args['train_on'],
                                                classes=args['classes_lst'], seed=args['seed'], response=args['response'])


    print("setting random seed to "+str(args['seed']))
    np.random.seed(args['seed'])
    if args['seed'] is not None:
        torch.random.manual_seed(args['seed'])

    traindataloader = torch.utils.data.DataLoader(dataset=train_dataset_list, sampler=RandomSampler(train_dataset_list),
                                                  batch_size=args['batchsize'], num_workers=args['workers'])

    validdataloader = torch.utils.data.DataLoader(dataset=valid_dataset_list, sampler=SequentialSampler(valid_dataset_list),
                                                 batch_size=args['batchsize'], num_workers=args['workers'])

    return traindataloader, validdataloader

def train(args):
    assert args['response'] in ["regression", "classification"]
    if args['response'] == "regression":
        args['classes_lst'] = [0]

    traindataloader, validdataloader = prepare_dataset(args)

    args['nclasses'] = traindataloader.dataset.nclasses
    args['seqlength'] = traindataloader.dataset.sequencelength
    args['input_dims'] = traindataloader.dataset.ndims

    print(f"sequence_length is: {args['seqlength']}")
    model = getModel(args)

    store = os.path.join(args['store'],args['model'])

    logger = Logger(columns=["accuracy"], modes=["train", "valid"], rootpath=store)


    if args['model'] in ["transformer"]:
        optimizer = ScheduledOptim(
            optim.Adam(
                filter(lambda x: x.requires_grad, model.parameters()),
                betas=(0.9, 0.98), eps=1e-09, weight_decay=args['weight_decay']),
            model.d_model, args['warmup'])
    elif args['model'] in ["rnn", "msresnet","tempcnn"]:
        optimizer = optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args['weight_decay'], lr=args['learning_rate'])
    else:
        raise ValueError(args['model'] + "no valid model. either 'rnn', 'msresnet', 'transformer', 'tempcnn'")

    config = dict(
        epochs=args['epochs'],
        learning_rate=args['learning_rate'],
        store=store,
        checkpoint_every_n_epochs=args['checkpoint_every_n_epochs'],
        test_every_n_epochs=args['valid_every_n_epochs'],
        logger=logger,
        optimizer=optimizer,
        response = args['response']
    )

    trainer = Trainer(model,traindataloader,validdataloader,**config)
    logger = trainer.fit()

    # stores all stored values in the rootpath of the logger
    logger.save()

    #pth = store+"/npy/confusion_matrix_{epoch}.npy".format(epoch = args[epochs)
    #parse_run(store, args['classmapping'], outdir=store)


    pass

def getModel(args):

    if args['model'] == "rnn":
        model = RNN(input_dim=args['input_dims'], nclasses=args['nclasses'], hidden_dims=args['hidden_dims'],
                              num_rnn_layers=args['num_layers'], dropout=args['dropout'], bidirectional=True, response = args['response'])

    if args['model'] == "msresnet":
        model = MSResNet(input_channel=args['input_dims'], layers=[1, 1, 1, 1], num_classes=args['nclasses'], hidden_dims=args['hidden_dims'], response = args['response'])

    if args['model'] == "tempcnn":
        model = TempCNN(input_dim=args['input_dims'], nclasses=args['nclasses'], sequence_length=args['seqlength'], hidden_dims=args['hidden_dims'], kernel_size=args['kernel_size'], response = args['response'])

    elif args['model'] == "transformer":

        hidden_dims = args['hidden_dims'] # 256
        n_heads = args['n_heads'] # 8
        n_layers = args['n_layers'] # 6
        len_max_seq = args['seqlength']
        dropout = args['dropout']
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
