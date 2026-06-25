# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:30:26 2023

@author: benjaminstoeckigt
"""


import sys
sys.path.append("./models")

import hashlib
import numpy as np
import torch
import random
import time
from pathlib import Path
import platform
import socket
import subprocess
import glob

from pytorch.models.TransformerEncoder import TransformerEncoder
from pytorch.models.multi_scale_resnet import MSResNet
from pytorch.models.TempCNN import TempCNN
from pytorch.models.rnn import RNN
from pytorch.utils.Dataset import Dataset
from pytorch.utils.trainer import Trainer, get_underlying_dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from pytorch.utils.logger import Logger
from pytorch.utils.mlflow_logger import MLflowLogger
from pytorch.utils.scheduled_optimizer import ScheduledOptim
from pytorch.utils.hw_monitor import HWMonitor, disk_info, squeeze_hw_info
import torch.optim as optim
import os, json
import shutil
from config_hyperparameter import hyperparameter_config, hyperparameter_tune
import optuna
from optuna.trial import TrialState
from torch.nn.utils.rnn import pad_sequence
from pytorch.utils.augmentation import time_warp, plot, apply_scaling, apply_augmentation

SENSITIVE_KEY_PARTS = (
    "password",
    "secret",
    "token",
    "key",
    "credential",
    "access_key",
    "private",
)

def sanitize_response_normalization(args):
    if args.get('response') == 'classification' and args.get('norm_factor_response') is not None:
        print("Ignoring norm_factor_response for classification targets.")
        args['norm_factor_response'] = None
    return args

def load_mlflow_server_config(args_train):
    if not args_train.get("use_mlflow", False):
        return

    try:
        import mlflow_config as mlflow_server_config
    except ImportError:
        return

    tracking_uri = getattr(mlflow_server_config, "MLFLOW_TRACKING_URI", None)
    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if tracking_uri:
        args_train["mlflow_tracking_uri"] = tracking_uri

    config_experiment = getattr(mlflow_server_config, "MLFLOW_EXPERIMENT_NAME", None)
    if config_experiment:
        args_train["mlflow_experiment"] = config_experiment

    config_run_name = getattr(mlflow_server_config, "MLFLOW_RUN_NAME", None)
    if config_run_name and not args_train.get("mlflow_run_name"):
        args_train["mlflow_run_name"] = config_run_name

    print(f"Using MLflow tracking URI from mlflow_config.py: {args_train.get('mlflow_tracking_uri')}")


def to_serializable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def is_sensitive_key(key):
    key_str = str(key).lower()
    return any(part in key_str for part in SENSITIVE_KEY_PARTS)


def sanitize_for_logging(value, parent_key=None):
    if parent_key is not None and is_sensitive_key(parent_key):
        return "***REDACTED***"
    if isinstance(value, dict):
        return {
            str(key): sanitize_for_logging(val, parent_key=key)
            for key, val in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_logging(item, parent_key=parent_key) for item in value]
    return to_serializable(value)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as file:
        json.dump(sanitize_for_logging(payload), file, indent=4, sort_keys=True)


def summarize_numeric_row(row):
    summary = {}
    for key, value in row.items():
        if key in ["epoch", "iteration", "mode"]:
            continue
        scalar = None
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            try:
                scalar = value.item()
            except (TypeError, ValueError):
                scalar = None
        if scalar is None:
            try:
                arr = np.array(value)
                if arr.size == 1:
                    scalar = arr.reshape(-1)[0]
            except Exception:
                scalar = None
        if scalar is None:
            continue
        if isinstance(scalar, np.generic):
            scalar = scalar.item()
        if isinstance(scalar, (float, np.floating)) and (np.isnan(scalar) or np.isinf(scalar)):
            continue
        if isinstance(scalar, (int, float, np.integer, np.floating)):
            summary[key] = float(scalar)
    return summary


def split_has_samples(root_path, include_thermal=False):
    csv_files = glob.glob(os.path.join(root_path, "csv", "*.csv"))
    if len(csv_files) > 0:
        return True

    cache_root = os.path.join(root_path, "npy")
    required_files = ["y.npy", "ndims.npy", "sequencelengths.npy", "ids.npy", "X.pkl", "doy.pkl"]
    if include_thermal:
        required_files.append("thermal_time.pkl")
    return all(os.path.exists(os.path.join(cache_root, filename)) for filename in required_files)


def random_train_valid_split(train_dataset, partition, split_ratio):
    total_size = len(train_dataset)
    if total_size < 2:
        raise ValueError(f"Need at least 2 training samples for random train/val split, got {total_size}.")

    selected_size = int(partition * total_size / 100.0)
    selected_size = max(2, min(total_size, selected_size))
    print("selected_size=" + str(selected_size))

    remaining_size = total_size - selected_size
    if remaining_size > 0:
        selected_dataset, _ = torch.utils.data.random_split(train_dataset, [selected_size, remaining_size])
    else:
        selected_dataset = train_dataset

    print(
        f"Selected {partition}% of the dataset: {len(selected_dataset)} samples "
        f"from a total of {total_size} samples."
    )

    train_size = int(split_ratio * len(selected_dataset))
    train_size = max(1, min(len(selected_dataset) - 1, train_size))
    valid_size = len(selected_dataset) - train_size
    train_subset, valid_subset = torch.utils.data.random_split(selected_dataset, [train_size, valid_size])
    return train_subset, valid_subset


def get_git_metadata(project_root):
    metadata = {
        "commit": None,
        "branch": None,
        "is_dirty": None,
    }
    try:
        metadata["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
        ).strip()
        metadata["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=project_root,
            text=True,
        ).strip()
        metadata["is_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                text=True,
            ).strip()
        )
    except Exception:
        pass
    return metadata


def get_environment_metadata(project_root):
    environment = {
        "timestamp_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cwd": os.getcwd(),
        "project_root": str(project_root),
        "git": get_git_metadata(project_root),
    }
    if torch.cuda.is_available():
        environment["cuda_device_name"] = torch.cuda.get_device_name(0)
    return environment


def get_model_config(args):
    model_config = {
        "model": args["model"],
        "response": args["response"],
        "nclasses": args["nclasses"],
        "input_dims": args["input_dims"],
        "seqlength": args["seqlength"],
        "hidden_dims": args["hidden_dims"],
        "dropout": args["dropout"],
    }
    if args["model"] == "rnn":
        model_config["num_layers"] = args["num_layers"]
    elif args["model"] == "tempcnn":
        model_config["kernel_size"] = args["kernel_size"]
    elif args["model"] == "transformer":
        model_config["n_layers"] = args["n_layers"]
        model_config["n_heads"] = args["n_heads"]
    return model_config


def get_preprocess_signature(preprocess_params):
    return {
        "time_range": preprocess_params.get("time_range"),
        "feature_order": preprocess_params.get("feature_order"),
        "interpolation": preprocess_params.get("Interpolation"),
        "interpolate_mode": preprocess_params.get("INTERPOLATE"),
        "int_day": preprocess_params.get("INT_DAY"),
        "thermal_time": preprocess_params.get("thermal_time"),
        "start_doy_month": preprocess_params.get("start_doy_month"),
        "split_method": preprocess_params.get("split_method"),
        "split_ratio": preprocess_params.get("split_ratio"),
        "column_name": preprocess_params.get("column_name"),
    }


def get_mlflow_logging_context(args_train):
    return {
        "use_mlflow": args_train.get("use_mlflow", False),
        "tracking_uri": args_train.get("mlflow_tracking_uri"),
        "experiment_name": args_train.get("mlflow_experiment"),
        "run_name": args_train.get("mlflow_run_name"),
    }


def get_source_artifact_paths(args_train):
    project_root = Path(__file__).resolve().parents[1]
    default_entry_script_path = str((project_root / "class_main_2_train.py").resolve())
    source_paths = [
        args_train.get("entry_script_path", default_entry_script_path),
        str(Path(__file__).resolve()),
        str((project_root / "pytorch" / "utils" / "trainer.py").resolve()),
        str((project_root / "pytorch" / "utils" / "mlflow_logger.py").resolve()),
        str((project_root / "config_hyperparameter.py").resolve()),
        str((project_root / "requirements.txt").resolve()),
    ]

    unique_paths = []
    seen = set()
    for path in source_paths:
        if path and path not in seen and os.path.exists(path):
            unique_paths.append(path)
            seen.add(path)
    return unique_paths


def build_artifact_manifest(store, entries):
    manifest_entries = []
    for entry in entries:
        local_path = entry.get("path")
        if not local_path or not os.path.exists(local_path):
            continue
        file_hash = hashlib.sha256()
        with open(local_path, "rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                file_hash.update(chunk)
        manifest_entries.append(
            {
                "category": entry.get("category"),
                "artifact_path": entry.get("artifact_path"),
                "basename": os.path.basename(local_path),
                "local_path": local_path,
                "size_bytes": os.path.getsize(local_path),
                "sha256": file_hash.hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "artifact_count": len(manifest_entries),
        "artifacts": manifest_entries,
    }
    manifest_path = os.path.join(store, "artifact_manifest.json")
    write_json(manifest_path, manifest)
    return manifest_path


def prepare_run_artifacts(args_train, preprocess_params, trial=None):
    project_root = Path(__file__).resolve().parents[1]
    store = os.path.join(args_train['store'], args_train['model'])
    os.makedirs(store, exist_ok=True)

    hyperparameter_keys = set(hyperparameter_config(args_train['model']).keys())
    resolved_hyperparameters = {
        key: args_train.get(key)
        for key in sorted(hyperparameter_keys)
        if key in args_train
    }
    training_parameters = {
        key: args_train.get(key)
        for key in sorted(args_train.keys())
        if key not in hyperparameter_keys and key != "preprocess_params"
    }

    preprocess_source_path = f'{Path(args_train["data_root"]).parent.parent}/preprocess_settings.json'
    preprocess_settings_path = os.path.join(store, "preprocess_settings.json")
    if os.path.exists(preprocess_source_path):
        try:
            shutil.copy(preprocess_source_path, preprocess_settings_path)
        except Exception:
            print("Couldnt Copy preprocess_settings.json")
            write_json(preprocess_settings_path, preprocess_params)
    elif not os.path.exists(preprocess_settings_path):
        write_json(preprocess_settings_path, preprocess_params)

    preprocess_runtime_path = os.path.join(store, "preprocess_runtime_params.json")
    hyperparameters_path = os.path.join(store, "hyperparameters.json")
    training_parameters_path = os.path.join(store, "training_parameters.json")
    resolved_hyperparameters_path = os.path.join(store, "resolved_hyperparameters.json")
    run_context_path = os.path.join(store, "run_context.json")

    optuna_context = None
    if trial is not None:
        optuna_context = {
            "trial_number": trial.number,
            "params": trial.params,
            "distributions": {key: str(value) for key, value in trial.distributions.items()},
        }

    write_json(preprocess_runtime_path, preprocess_params)
    write_json(
        hyperparameters_path,
        {key: value for key, value in args_train.items() if key != "preprocess_params"},
    )
    write_json(training_parameters_path, training_parameters)
    write_json(resolved_hyperparameters_path, resolved_hyperparameters)
    write_json(
        run_context_path,
        {
            "training_parameters": training_parameters,
            "resolved_hyperparameters": resolved_hyperparameters,
            "preprocess_params": preprocess_params,
            "optuna_trial": optuna_context,
            "environment": get_environment_metadata(project_root),
        },
    )

    optuna_trial_path = None
    if optuna_context is not None:
        optuna_trial_path = os.path.join(store, f"optuna_trial_{trial.number:03d}.json")
        write_json(optuna_trial_path, optuna_context)

    return {
        "store": store,
        "preprocess_settings_path": preprocess_settings_path,
        "preprocess_runtime_path": preprocess_runtime_path,
        "hyperparameters_path": hyperparameters_path,
        "training_parameters_path": training_parameters_path,
        "resolved_hyperparameters_path": resolved_hyperparameters_path,
        "run_context_path": run_context_path,
        "optuna_trial_path": optuna_trial_path,
        "resolved_hyperparameters": resolved_hyperparameters,
        "training_parameters": training_parameters,
        "source_paths": get_source_artifact_paths(args_train),
        "model_config": get_model_config(args_train),
        "preprocess_signature": get_preprocess_signature(preprocess_params),
        "mlflow_logging": get_mlflow_logging_context(args_train),
    }

def train_init(args_train, preprocess_params):

    args_train["preprocess_params"] = preprocess_params
    args_train["time_range"] = preprocess_params["time_range"] # relevant for relative yearls doy seperation for augmentations
    args_train["workers"] = 10  # number of CPU workers to load the next batch
    args_train["project_name"] = preprocess_params["project_name"]
    args_train.setdefault("use_mlflow", False)
    args_train.setdefault("mlflow_tracking_uri", None)
    args_train.setdefault("mlflow_experiment", preprocess_params["project_name"])
    args_train.setdefault("mlflow_run_name", None)
    args_train.setdefault("mlflow_nested_runs", False)
    load_mlflow_server_config(args_train)
    if args_train.get("tune") and args_train.get("final_training"):
        print(
            "WARNING: 'tune=True' requires validation metrics, "
            "but 'final_training=True' disables validation. "
            "Setting 'final_training=False' for hyperparameter tuning."
        )
        args_train["final_training"] = False

    args_train["data_root"] = f'{preprocess_params["process_folder"]}/results/_SITSrefdata/{preprocess_params["project_name"]}/sepfiles/train/' # folder with CSV or cached NPY folder
    args_train["data_root_val"] = f'{preprocess_params["process_folder"]}/results/_SITSrefdata/{preprocess_params["project_name"]}/sepfiles/val/'  # folder with CSV or cached NPY folder
    args_train["store"] = f'{preprocess_params["process_folder"]}/results/_SITSModels/{preprocess_params["project_name"]}/'  # Store Model Data Path

    args_train["thermal_time"] = preprocess_params["thermal_time"]
    # create hw_monitor output dir if it doesn't exist
    Path(args_train['store'] + '/' + args_train['model'] + '/hw_monitor').mkdir(parents=True, exist_ok=True)
    args_train["sdb1"] = ["sdb1"]
    args_train["split_method"] = preprocess_params["split_method"]
    args_train["split_ratio"] = preprocess_params["split_ratio"]

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
        if args_train['response'] == 'classification':
            direction = "maximize"
        else:
            direction = "minimize"
        study_logger = MLflowLogger(
            enabled=args_train.get("use_mlflow", False),
            tracking_uri=args_train.get("mlflow_tracking_uri"),
            experiment_name=args_train.get("mlflow_experiment"),
            run_name=args_train.get("mlflow_run_name") or f"study_{args_train['study_name']}_{time.strftime('%Y%m%d_%H%M%S')}",
            tags={
                "project_name": args_train.get("project_name"),
                "model": args_train.get("model"),
                "response": args_train.get("response"),
                "split_method": args_train.get("split_method"),
                "study_name": args_train.get("study_name"),
                "run_type": "optuna_study",
            },
        )
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(),pruner=optuna.pruners.MedianPruner(), storage=storage,
                                    study_name=args_train['study_name'])
        try:
            study_logger.start_run(run_name=f"study_{args_train['study_name']}_{time.strftime('%Y%m%d_%H%M%S')}")
            study_logger.log_params({key: value for key, value in args_train.items() if key != "preprocess_params"})
            study_logger.log_params({"optuna_storage": storage_path})
            args_train["mlflow_nested_runs"] = args_train.get("use_mlflow", False)
            study.optimize(lambda trial: train(trial, args_train), n_trials=50)

            completed_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
            state_counts = {}
            for trial_item in study.trials:
                state_key = trial_item.state.name.lower()
                state_counts[state_key] = state_counts.get(state_key, 0) + 1

            if not completed_trials:
                raise RuntimeError(
                    "No Optuna trials completed successfully. "
                    "Check trial errors above and verify the objective returns a numeric value."
                )

            print(f"Best value: {study.best_value} (params: {study.best_params})")
            study_logger.log_metrics({"best_value": study.best_value}, step=study.best_trial.number)
            study_logger.log_params({"best_trial_number": study.best_trial.number})
            study_logger.log_params({f"best_{key}": value for key, value in study.best_params.items()})
            best_trial_run_id = study.best_trial.user_attrs.get("mlflow_run_id")
            if best_trial_run_id:
                study_logger.log_params({"best_trial_run_id": best_trial_run_id})
                study_logger.log_params({"best_trial_model_uri": f"runs:/{best_trial_run_id}/best_model"})
            study_summary_path = os.path.join(args_train['store'], args_train['model'], 'optuna', 'study_summary.json')
            write_json(
                study_summary_path,
                {
                    "study_name": study.study_name,
                    "direction": direction,
                    "best_value": study.best_value,
                    "best_trial_number": study.best_trial.number,
                    "best_params": study.best_params,
                    "best_trial_run_id": best_trial_run_id,
                    "best_trial_model_uri": f"runs:/{best_trial_run_id}/best_model" if best_trial_run_id else None,
                    "n_trials": len(study.trials),
                    "state_counts": state_counts,
                },
            )
            study_logger.log_artifact(study_summary_path, artifact_path="optuna")
            if os.path.exists(storage_path):
                study_logger.log_artifact(storage_path, artifact_path="optuna")
            try:
                study_trials_path = os.path.join(args_train['store'], args_train['model'], 'optuna', 'study_trials.csv')
                study.trials_dataframe().to_csv(study_trials_path, index=False)
                study_logger.log_artifact(study_trials_path, artifact_path="optuna")
            except Exception as exc:
                print(f"Could not export Optuna trials dataframe: {exc}")
            study_logger.end_run(status="FINISHED")
        except Exception:
            study_logger.end_run(status="FAILED")
            raise
        finally:
            args_train["mlflow_nested_runs"] = False
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
        sanitize_response_normalization(args_train)
    else:
        new_args = hyperparameter_config(args_train['model'])
        args_train.update(new_args)
        sanitize_response_normalization(args_train)

    time.sleep(3)
    hwmon_i.stop_averaging()
    avgs = hwmon_i.get_averages()
    squeezed = squeeze_hw_info(avgs)
    mean_data = {key: round(value, 1) for key, value in squeezed.items() if "mean" in key}
    print(f"##################\nMean Values Hardware Monitoring (Preparing Data):\n{mean_data}\n##################")
    hwmon_i.stop()

    # load dataset
    train_dataset = prepare_dataset(args_train)

    # validation dataset is not used for final model training"
    if not args_train["final_training"] and args_train["split_method"] == "user_defined":
        if split_has_samples(args_train["data_root_val"], include_thermal=args_train['thermal_time'] is not None):
            valid_dataset = prepare_dataset(args_train, split="val")
            args_train["validation_source"] = "user_defined"
        else:
            print(
                "WARNING: split_method='user_defined' but no validation samples were found in "
                f"{args_train['data_root_val']}. Falling back to random train/val split from training data."
            )
            train_dataset, valid_dataset = random_train_valid_split(
                train_dataset,
                partition=args_train['partition'],
                split_ratio=args_train['split_ratio'],
            )
            args_train["validation_source"] = "random_fallback_from_train"
    elif not args_train["final_training"] and args_train["split_method"] in ["random", "random_test", "no_split"]:
        train_dataset, valid_dataset = random_train_valid_split(
            train_dataset,
            partition=args_train['partition'],
            split_ratio=args_train['split_ratio'],
        )
        args_train["validation_source"] = "random"
    else:
        valid_dataset = None
        args_train["validation_source"] = "none"

    p = args_train['augmentation']
    plotting = args_train['augmentation_plot']
    time_range = args_train['time_range'][1]

    include_thermal = args_train['thermal_time'] is not None

    traindataloader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=RandomSampler(train_dataset),
                                                  batch_size=args_train['batchsize'], num_workers=args_train['workers'],
                                                  collate_fn=lambda batch: collate_fn(batch, p, plotting, time_range, include_thermal))

    if args_train["final_training"] == True:
        validdataloader = None
    else:
        validdataloader = torch.utils.data.DataLoader(dataset=valid_dataset, sampler=SequentialSampler(valid_dataset),
                                                      batch_size=args_train['batchsize'], num_workers=args_train['workers'],
                                                      collate_fn=lambda batch: collate_fn_notransform(batch, include_thermal, p=0, plotting=None))

    print(f"Training Sample Size: {len(traindataloader.dataset)}")
    if validdataloader is not None:
        print(f"Validation Sample Size: {len(validdataloader.dataset)}")
    args_train["train_sample_size"] = len(traindataloader.dataset)
    args_train["valid_sample_size"] = len(validdataloader.dataset) if validdataloader is not None else 0
    args_train["has_validation_data"] = validdataloader is not None

    base_dataset = get_underlying_dataset(traindataloader)
    if args_train['model'] in ["transformer"]:
        args_train['seqlength'] = args_train['max_seq_length']
    elif args_train['model'] in ["rnn", "msresnet","tempcnn"]:
        args_train['seqlength'] = base_dataset.sequencelength
    # OPTUNA: this is the build_model_custom(trial)
    #model = getModel(args)
    #args_train['nclasses'] = traindataloader.dataset.dataset.dataset.nclasses
    #args_train['input_dims'] = traindataloader.dataset.dataset.dataset.ndims
    args_train['nclasses'] = base_dataset.nclasses
    args_train['input_dims'] = base_dataset.ndims
    args_train["best_model_selection_metric"] = (
        "valid_mean_f1"
        if validdataloader is not None and args_train['response'] == 'classification' and args_train['validation_metric'] == 'f1'
        else "valid_accuracy"
        if validdataloader is not None and args_train['response'] == 'classification'
        else "valid_rmse"
        if validdataloader is not None
        else "final_epoch_model"
    )
    run_artifacts = prepare_run_artifacts(args_train, args_train.get("preprocess_params", {}), trial=trial)
    #print(f"Exemplary Sequence Length: {base_dataset.sequencelength}")
    print(f"Maximum DOY Sequence Length: {args_train['seqlength']}")
    print(f"Input Dims: {args_train['input_dims']}")
    print(f"Prediction Classes: {len(args_train['classes_lst'])}")
    print(f"Data Augmentation: {p * 100} % Training Data will be augmented (Single or Double (50/50) of DOY Day Shifting / Zero Out")
    if include_thermal:
        print(f"Applying Transformer Model with Thermal Positional Encoding!\n-> GDD Path:{args_train['thermal_time']}")
    else:
        print("Applying Transformer Model with Calendar Positional Encoding!")

    model = getModel(args_train)

    store = run_artifacts["store"]

    logger = Logger(columns=["accuracy", "mean_f1"], modes=["train", "valid"], rootpath=store)
    if trial is not None:
        run_name = f"{args_train.get('study_name', args_train.get('project_name', args_train['model']))}_trial_{trial.number:03d}"
    else:
        run_name = args_train.get("mlflow_run_name") or f"{args_train.get('project_name', args_train['model'])}_{args_train['model']}_{time.strftime('%Y%m%d_%H%M%S')}"
    mlflow_logger = MLflowLogger(
        enabled=args_train.get("use_mlflow", False),
        tracking_uri=args_train.get("mlflow_tracking_uri"),
        experiment_name=args_train.get("mlflow_experiment"),
        run_name=run_name,
        tags={
            "project_name": args_train.get("project_name"),
            "model": args_train.get("model"),
            "response": args_train.get("response"),
            "split_method": args_train.get("split_method"),
            "study_name": args_train.get("study_name"),
            "run_type": "optuna_trial" if trial is not None else "training",
            "trial_number": trial.number if trial is not None else None,
        },
        nested=args_train.get("mlflow_nested_runs", False),
    )
    mlflow_logger.start_run(run_name=run_name)
    if trial is not None and mlflow_logger.run_id is not None:
        try:
            trial.set_user_attr("mlflow_run_id", mlflow_logger.run_id)
        except Exception:
            pass
    mlflow_logger.log_params({key: value for key, value in args_train.items() if key != "preprocess_params"})
    mlflow_logger.log_params({
        "model_store": store,
        "train_sample_size": len(traindataloader.dataset),
        "valid_sample_size": len(validdataloader.dataset) if validdataloader is not None else 0,
        "has_validation_data": validdataloader is not None,
        "input_dims": args_train['input_dims'],
        "nclasses": args_train['nclasses'],
        "seqlength": args_train['seqlength'],
        "best_model_selection_metric": args_train["best_model_selection_metric"],
    })
    mlflow_logger.log_params({f"hp_{key}": value for key, value in run_artifacts["resolved_hyperparameters"].items()})
    if trial is not None:
        mlflow_logger.log_params({f"optuna_param_{key}": value for key, value in trial.params.items()})
        mlflow_logger.log_params({f"optuna_distribution_{key}": str(value) for key, value in trial.distributions.items()})

    mlflow_logger.log_artifact(run_artifacts["preprocess_settings_path"], artifact_path="config")
    mlflow_logger.log_artifact(run_artifacts["preprocess_runtime_path"], artifact_path="config")
    mlflow_logger.log_artifact(run_artifacts["hyperparameters_path"], artifact_path="config")
    mlflow_logger.log_artifact(run_artifacts["training_parameters_path"], artifact_path="config")
    mlflow_logger.log_artifact(run_artifacts["resolved_hyperparameters_path"], artifact_path="config")
    mlflow_logger.log_artifact(run_artifacts["run_context_path"], artifact_path="config")
    if run_artifacts["optuna_trial_path"] is not None:
        mlflow_logger.log_artifact(run_artifacts["optuna_trial_path"], artifact_path="optuna")
    for source_path in run_artifacts["source_paths"]:
        mlflow_logger.log_artifact(source_path, artifact_path="source")

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
        norm_factor_response=args_train['norm_factor_response'],
        use_class_weights=args_train['use_class_weights'],
        validation_metric=args_train['validation_metric'],
        checkpoint_metadata={
            "model_config": run_artifacts["model_config"],
            "preprocess_signature": run_artifacts["preprocess_signature"],
            "mlflow_logging": run_artifacts["mlflow_logging"],
        },
    )

    trainer = Trainer(trial,model,traindataloader,validdataloader, mlflow_logger=mlflow_logger, **config)

    try:
        logger = trainer.fit()
        log_df = logger.get_data()
        final_log_path = os.path.join(store, "log.csv")
        log_df.to_csv(final_log_path)
        mlflow_logger.log_artifact(final_log_path, artifact_path="logs")
        final_model_path = os.path.join(store, "final_model.pth")
        trainer.snapshot(final_model_path)
        mlflow_logger.log_artifact(final_model_path, artifact_path="models")
        mlflow_logger.log_pytorch_model(trainer.model, artifact_path="final_model")

        best_model_summary = {
            "selection_strategy": "validation_metric" if trainer.best_model_path is not None else "final_epoch_no_validation",
            "selection_metric": trainer.best_metric_name,
            "selection_mode": trainer.best_metric_mode,
            "best_epoch": trainer.best_epoch,
            "best_metric_value": trainer.best_metric_value,
            "best_stats": trainer.best_stats,
        }
        if trainer.best_model_path is not None and os.path.exists(trainer.best_model_path):
            mlflow_logger.log_artifact(trainer.best_model_path, artifact_path="models")
            trainer.model.load(trainer.best_model_path)
            mlflow_logger.log_pytorch_model(trainer.model, artifact_path="best_model")
            if trainer.best_metric_value is not None and trainer.best_metric_name is not None:
                mlflow_logger.log_metrics(
                    {
                        f"best_valid_{trainer.best_metric_name}": trainer.best_metric_value,
                        "best_epoch": trainer.best_epoch,
                    }
                )
        else:
            best_model_summary["selection_metric"] = None
            best_model_summary["selection_mode"] = None
            best_model_summary["best_epoch"] = trainer.epoch
            best_model_summary["best_metric_value"] = None

        best_model_summary_path = os.path.join(store, "best_model_summary.json")
        write_json(best_model_summary_path, best_model_summary)
        mlflow_logger.log_artifact(best_model_summary_path, artifact_path="models")

        train_rows = log_df[log_df['mode'] == 'train']
        valid_rows = log_df[log_df['mode'] == 'valid']
        if not train_rows.empty:
            final_train_row = train_rows.sort_values("epoch").iloc[-1]
            final_train_metrics = summarize_numeric_row(final_train_row)
            if final_train_metrics:
                mlflow_logger.log_metrics(
                    final_train_metrics,
                    step=int(final_train_row.get("epoch", 0)),
                    prefix="final_train",
                )
        if not valid_rows.empty:
            final_valid_row = valid_rows.sort_values("epoch").iloc[-1]
            final_valid_metrics = summarize_numeric_row(final_valid_row)
            if final_valid_metrics:
                mlflow_logger.log_metrics(
                    final_valid_metrics,
                    step=int(final_valid_row.get("epoch", 0)),
                    prefix="final_valid",
                )

            monitored_metric = trainer.best_metric_name
            best_valid_row = None
            if monitored_metric and monitored_metric in valid_rows.columns:
                metric_series = valid_rows[monitored_metric].dropna()
                if not metric_series.empty:
                    best_index = metric_series.idxmax() if trainer.best_metric_mode == "max" else metric_series.idxmin()
                    best_valid_row = valid_rows.loc[best_index]
            if best_valid_row is not None:
                best_valid_metrics = summarize_numeric_row(best_valid_row)
                if best_valid_metrics:
                    mlflow_logger.log_metrics(
                        best_valid_metrics,
                        step=int(best_valid_row.get("epoch", 0)),
                        prefix="best_valid",
                    )

        metrics_summary = {
            "final_train": summarize_numeric_row(train_rows.sort_values("epoch").iloc[-1]) if not train_rows.empty else None,
            "final_valid": summarize_numeric_row(valid_rows.sort_values("epoch").iloc[-1]) if not valid_rows.empty else None,
            "best_model_summary": best_model_summary,
        }
        metrics_summary_path = os.path.join(store, "metrics_summary.json")
        write_json(metrics_summary_path, metrics_summary)
        mlflow_logger.log_artifact(metrics_summary_path, artifact_path="logs")

        if trial is not None:
            trial_artifact_path = f"optuna/trials/trial_{trial.number:03d}"
            mlflow_logger.log_artifact(final_log_path, artifact_path=f"{trial_artifact_path}/logs")
            mlflow_logger.log_artifact(final_model_path, artifact_path=f"{trial_artifact_path}/models")
            mlflow_logger.log_artifact(best_model_summary_path, artifact_path=f"{trial_artifact_path}/models")
            mlflow_logger.log_artifact(metrics_summary_path, artifact_path=f"{trial_artifact_path}/logs")
            if trainer.best_model_path is not None and os.path.exists(trainer.best_model_path):
                mlflow_logger.log_artifact(trainer.best_model_path, artifact_path=f"{trial_artifact_path}/models")

        manifest_entries = [
            {"path": run_artifacts["preprocess_settings_path"], "artifact_path": "config", "category": "config"},
            {"path": run_artifacts["preprocess_runtime_path"], "artifact_path": "config", "category": "config"},
            {"path": run_artifacts["hyperparameters_path"], "artifact_path": "config", "category": "config"},
            {"path": run_artifacts["training_parameters_path"], "artifact_path": "config", "category": "config"},
            {"path": run_artifacts["resolved_hyperparameters_path"], "artifact_path": "config", "category": "config"},
            {"path": run_artifacts["run_context_path"], "artifact_path": "config", "category": "config"},
            {"path": final_log_path, "artifact_path": "logs", "category": "metrics"},
            {"path": final_model_path, "artifact_path": "models", "category": "model"},
            {"path": best_model_summary_path, "artifact_path": "models", "category": "model"},
            {"path": metrics_summary_path, "artifact_path": "logs", "category": "metrics"},
        ]
        if run_artifacts["optuna_trial_path"] is not None:
            manifest_entries.append(
                {"path": run_artifacts["optuna_trial_path"], "artifact_path": "optuna", "category": "optuna"}
            )
        if trainer.best_model_path is not None and os.path.exists(trainer.best_model_path):
            manifest_entries.append(
                {"path": trainer.best_model_path, "artifact_path": "models", "category": "model"}
            )
        for source_path in run_artifacts["source_paths"]:
            manifest_entries.append(
                {"path": source_path, "artifact_path": "source", "category": "source"}
            )
        artifact_manifest_path = build_artifact_manifest(store, manifest_entries)
        mlflow_logger.log_artifact(artifact_manifest_path, artifact_path="meta")

        mlflow_logger.end_run(status="FINISHED")
    except Exception:
        mlflow_logger.end_run(status="FAILED")
        raise

    all_metrics = logger.get_data()
    metric_name = None
    metric_mode = None
    if config['response'] == 'classification':
        metric_name = 'mean_f1' if args_train['validation_metric'] == 'f1' else 'accuracy'
        metric_mode = 'max'
    else:
        metric_name = 'rmse'
        metric_mode = 'min'

    objective_value = None
    objective_source = None
    preferred_modes = ['valid', 'train'] if not args_train["final_training"] else ['train']
    if trial is not None and 'train' not in preferred_modes:
        preferred_modes.append('train')

    for mode in preferred_modes:
        mode_metrics = all_metrics[all_metrics['mode'] == mode]
        if mode_metrics.empty or metric_name not in mode_metrics.columns:
            continue
        metric_series = mode_metrics[metric_name].dropna()
        if metric_series.empty:
            continue
        if metric_mode == 'max':
            objective_value = float(metric_series.max())
        else:
            objective_value = float(metric_series.min())
        objective_source = mode
        break

    if trial is not None:
        if objective_value is None:
            raise ValueError(
                f"Could not derive Optuna objective metric '{metric_name}' from logged metrics."
            )
        try:
            trial.set_user_attr("objective_metric", metric_name)
            trial.set_user_attr("objective_mode", metric_mode)
            trial.set_user_attr("objective_source", objective_source)
            trial.set_user_attr("objective_value", float(objective_value))
        except Exception:
            pass
        if objective_source != 'valid':
            print(
                f"Optuna objective fallback: using {objective_source} '{metric_name}' "
                "because validation metrics were unavailable."
            )
        return objective_value

    if not args_train["final_training"] and objective_value is not None:
        return objective_value

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

def prepare_dataset(args, split=None):
    assert args['response'] in ["regression_sigmoid", "regression", "regression_relu", "classification"]

    if args['response'].startswith("regression"):
        args['classes_lst'] = [0]
    #ImbalancedDatasetSampler

    if split is None:
        ref_dataset = Dataset(root=args['data_root'], classes=args['classes_lst'], seed=args['seed'], response=args['response'],
                              norm=args['norm_factor_features'], norm_response=args['norm_factor_response'], thermal=args["thermal_time"])
    else:
        ref_dataset = Dataset(root=args['data_root_val'], classes=args['classes_lst'], seed=args['seed'], response=args['response'],
                              norm=args['norm_factor_features'], norm_response=args['norm_factor_response'], thermal=args["thermal_time"])

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
