from train import train

args = {
    'batchsize': 256,  # batch size
    'epochs': 150,  # number of training epochs
    'workers': 10,  # number of CPU workers to load the next batch
    'data_root': '/uge_mount/data_test/',
    'store': '/uge_mount/results/',  # store run logger results
    'valid_every_n_epochs': 1,  # skip some valid epochs for faster overall training
    'checkpoint_every_n_epochs': 2,  # save checkpoints during training
    'seed': 0,  # seed for batching and weight initialization
    'valid_on': "valid",
    'train_on': "train",
    'model': "transformer",  # "tempcnn","rnn","msresnet","transformer"
    'response': "regression",  # "regression", "classification"
    'classes_lst': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
}


def old_hyperparameter_config(model):
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


new_args = old_hyperparameter_config(args['model'])
args.update(new_args)

if __name__ == '__main__':
    train(args)