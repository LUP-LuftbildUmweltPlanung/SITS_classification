import wandb


# setting sweep parameters

sweep_config = {
    'program': 'sits_main.py',
    'command': ['python3','${program}','${args}'],
    'method': 'random'
    }

metric = {
    'name': 'loss',
    'goal': 'minimize'
    }

sweep_config['metric'] = metric


parameters_dict = {
    #'optimizer': {
    #    'values': ['adam', 'sgd']
    #    },
    'weight_decay': {
        'values': [0.0003, 0.0004]
        },
    'dropout': {
          'values': [0.1, 0.2]
        },
    'learning_rate': {
          'values': [0.01, 0.02]
        },
    'batchsize': {
            'values': [128,256]
        },
    }

sweep_config['parameters'] = parameters_dict


sweep_id = wandb.sweep(sweep_config, project="test_sits_main")
wandb.agent(sweep_id, count=5)