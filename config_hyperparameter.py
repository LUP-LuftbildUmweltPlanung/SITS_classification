def hyperparameter_config(model):
    assert model in ["tempcnn", "transformer", "rnn", "msresnet"]
    if model == "tempcnn":
        return {
            "model": "tempcnn",
            "batchsize": 512,  # batch size
            "kernel_size": 3,
            "hidden_dims": 128,
            "dropout": 0,
            "weight_decay": 0.00012,
            "learning_rate": 0.00018,
            'partition': 100,  # partition of whole reference data
        }
    elif model == "transformer":
        return {
            "model": "transformer",
            "batchsize": 256,  # batch size
            "hidden_dims": 128,
            "n_heads": 5,
            "n_layers": 4,
            "learning_rate": 0.0010913,
            "dropout": 0,
            "weight_decay": 0.000306,
            "warmup": 1000,
            'partition': 100,  # partition of whole reference data
        }
    elif model == "rnn":
        return {
            "model": "rnn",
            "batchsize": 512,  # batch size
            "num_layers": 4,
            "hidden_dims": 32,
            "learning_rate": 0.010489,
            "dropout": 0,
            "weight_decay": 0.000371,
            "bidirectional": True,
            'partition': 100,  # partition of whole reference data
        }
    elif model == "msresnet":
        return {
            "model": "msresnet",
            "batchsize": 512,  # batch size
            "hidden_dims": 32,
            "weight_decay": 0.000059,
            "learning_rate": 0.000657,
            'partition': 100,  # partition of whole reference data
        }
    else:
        raise ValueError("Invalid model")


def hyperparameter_tune(trial,model):
        assert model in ["tempcnn", "transformer", "rnn", "msresnet"]
        if model == "tempcnn":
            return {
                "model": "tempcnn",
                "batchsize": 2 ** trial.suggest_int("hidden_dims", 7, 9),  # (name, low, high, step)
                "kernel_size": trial.suggest_int("kernel_size", 3, 7, step=2),
                "hidden_dims": 2 ** trial.suggest_int("hidden_dims", 5, 8),# (name, low, high, step)
                "dropout": trial.suggest_float("dropout", 0, 1),
                "learning_rate": trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                "weight_decay": trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
                "partition": trial.suggest_int("partition", 20, 100, step=10),  # (name, low, high, step)
            }
        elif model == "transformer":
            return {
                "model": "transformer",
                "batchsize": 2 ** trial.suggest_int("batchsize", 7, 9),  # (name, low, high, step)
                "hidden_dims": 2 ** trial.suggest_int("hidden_dims", 6, 8),# (name, low, high, step)
                "n_heads": trial.suggest_int("n_heads", 3, 6),
                "n_layers": trial.suggest_int("n_layers", 2, 5),
                "dropout": 0,
                "learning_rate": trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                "weight_decay": trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
                "warmup": 1000,
                "partition": 100# trial.suggest_int("partition", 20, 100, step=10),  # (name, low, high, step)
            }
        elif model == "rnn":
            return {
                "model": "rnn",
                "batchsize": 2 ** trial.suggest_int("hidden_dims", 7, 9),  # (name, low, high, step)
                "dropout": trial.suggest_float("dropout", 0, 0.9, step=0.3),
                "n_layers": trial.suggest_int("n_layers", 3, 6),
                "hidden_dims": trial.suggest_int("hidden_dims", 64, 512, step=64),# (name, low, high, step)
                "learning_rate": trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                "weight_decay": trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                "bidirectional": True,
                "partition": trial.suggest_int("partition", 20, 100, step=20),  # (name, low, high, step)
            }
        elif model == "msresnet":
            return {
                "model": "msresnet",
                "batchsize": 2 ** trial.suggest_int("hidden_dims", 7, 9),  # (name, low, high, step)
                "hidden_dims": trial.suggest_int("hidden_dims", 64, 512, step=64),# (name, low, high, step)
                "learning_rate": trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                "weight_decay": trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                "partition": trial.suggest_int("partition", 20, 100, step=20),  # (name, low, high, step)
            }
        else:
            raise ValueError("Invalid model")

