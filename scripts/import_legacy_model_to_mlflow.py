#!/usr/bin/env python3

'''
cd /rvt_mount/SITS_classification
python3 scripts/import_legacy_model_to_mlflow.py \
  --checkpoint /path/to/old_model.pth \
  --config-dir /path/to/old_model_folder \
  --experiment legacy_models
'''

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pytorch.train import (
    build_artifact_manifest,
    get_environment_metadata,
    get_mlflow_logging_context,
    load_mlflow_server_config,
    sanitize_for_logging,
    write_json,
)
from pytorch.train import getModel
from pytorch.utils.legacy_compat import install_legacy_pickle_compat
from pytorch.utils.mlflow_logger import MLflowLogger


def load_json_if_exists(path):
    if not path or not os.path.exists(path):
        return None
    import json
    with open(path, "r") as file_obj:
        return json.load(file_obj)


def derive_model_config(saved_state, config_dir, override_config_path=None):
    model_config = {}

    if saved_state.get("model_config"):
        model_config.update(saved_state["model_config"])

    override_config = load_json_if_exists(override_config_path)
    if isinstance(override_config, dict):
        model_config.update(override_config.get("model_config", override_config))

    hyperparameters = load_json_if_exists(os.path.join(config_dir, "hyperparameters.json"))
    if isinstance(hyperparameters, dict):
        for key in [
            "model",
            "response",
            "hidden_dims",
            "dropout",
            "num_layers",
            "kernel_size",
            "n_layers",
            "n_heads",
            "nclasses",
            "input_dims",
            "seqlength",
        ]:
            if key in hyperparameters and model_config.get(key) is None:
                model_config[key] = hyperparameters[key]

    if saved_state.get("nclasses") is not None:
        model_config["nclasses"] = saved_state["nclasses"]
    if saved_state.get("ndims") is not None:
        model_config["input_dims"] = saved_state["ndims"]
    if saved_state.get("sequencelength") is not None:
        model_config["seqlength"] = saved_state["sequencelength"]

    required_keys = ["model", "response", "nclasses", "input_dims", "seqlength", "hidden_dims", "dropout"]
    missing_keys = [key for key in required_keys if model_config.get(key) is None]
    if missing_keys:
        raise ValueError(
            "Legacy checkpoint is missing required model config keys: "
            + ", ".join(missing_keys)
            + ". Provide --model-config with a JSON file containing them."
        )

    model_type = model_config["model"]
    if model_type == "rnn" and model_config.get("num_layers") is None:
        raise ValueError("RNN imports require 'num_layers' in the model config.")
    if model_type == "tempcnn" and model_config.get("kernel_size") is None:
        raise ValueError("TempCNN imports require 'kernel_size' in the model config.")
    if model_type == "transformer":
        for key in ["n_layers", "n_heads"]:
            if model_config.get(key) is None:
                raise ValueError(f"Transformer imports require '{key}' in the model config.")

    return model_config


def main():
    parser = argparse.ArgumentParser(description="Import a legacy SITS checkpoint into MLflow.")
    parser.add_argument("--checkpoint", required=True, help="Path to a legacy .pth checkpoint.")
    parser.add_argument("--config-dir", default=None, help="Directory containing legacy JSON config artifacts.")
    parser.add_argument("--model-config", default=None, help="Optional JSON file with model_config overrides.")
    parser.add_argument("--experiment", default=None, help="MLflow experiment name.")
    parser.add_argument("--run-name", default=None, help="MLflow run name.")
    parser.add_argument("--tracking-uri", default=None, help="Optional MLflow tracking URI override.")
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    config_dir = os.path.abspath(args.config_dir or os.path.dirname(checkpoint_path))
    install_legacy_pickle_compat()
    saved_state = torch.load(checkpoint_path, map_location="cpu")
    model_config = derive_model_config(saved_state, config_dir, override_config_path=args.model_config)

    mlflow_args = {
        "use_mlflow": True,
        "mlflow_tracking_uri": args.tracking_uri,
        "mlflow_experiment": args.experiment,
        "mlflow_run_name": args.run_name,
    }
    load_mlflow_server_config(mlflow_args)
    if args.tracking_uri:
        mlflow_args["mlflow_tracking_uri"] = args.tracking_uri
    if args.experiment:
        mlflow_args["mlflow_experiment"] = args.experiment
    if args.run_name:
        mlflow_args["mlflow_run_name"] = args.run_name

    model = getModel(model_config)
    model.load_state_dict(saved_state["model_state"])
    model.eval()

    run_name = mlflow_args.get("mlflow_run_name") or f"legacy_import_{Path(checkpoint_path).stem}"
    mlflow_logger = MLflowLogger(
        enabled=True,
        tracking_uri=mlflow_args.get("mlflow_tracking_uri"),
        experiment_name=mlflow_args.get("mlflow_experiment"),
        run_name=run_name,
        tags={
            "run_type": "legacy_model_import",
            "model": model_config.get("model"),
            "response": model_config.get("response"),
        },
    )

    artifact_dir = os.path.join(config_dir, "legacy_import_artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    import_summary = {
        "checkpoint_path": checkpoint_path,
        "config_dir": config_dir,
        "checkpoint_schema_version": saved_state.get("checkpoint_schema_version"),
        "checkpoint_keys": sorted(saved_state.keys()),
        "model_config": model_config,
        "preprocess_signature": saved_state.get("preprocess_signature"),
        "mlflow_logging": get_mlflow_logging_context(mlflow_args),
        "environment": get_environment_metadata(PROJECT_ROOT),
    }
    import_summary_path = os.path.join(artifact_dir, "legacy_import_summary.json")
    write_json(import_summary_path, import_summary)

    config_artifacts = []
    for filename in [
        "preprocess_settings.json",
        "preprocess_runtime_params.json",
        "hyperparameters.json",
        "training_parameters.json",
        "resolved_hyperparameters.json",
        "run_context.json",
    ]:
        file_path = os.path.join(config_dir, filename)
        if os.path.exists(file_path):
            config_artifacts.append(file_path)

    try:
        mlflow_logger.start_run(run_name=run_name)
        mlflow_logger.log_params({f"model_{key}": value for key, value in sanitize_for_logging(model_config).items()})
        mlflow_logger.log_params(
            {
                "import_checkpoint_path": checkpoint_path,
                "import_config_dir": config_dir,
                "checkpoint_schema_version": saved_state.get("checkpoint_schema_version"),
            }
        )
        mlflow_logger.log_artifact(checkpoint_path, artifact_path="models")
        mlflow_logger.log_pytorch_model(model, artifact_path="legacy_model")
        mlflow_logger.log_artifact(import_summary_path, artifact_path="meta")
        for config_path in config_artifacts:
            mlflow_logger.log_artifact(config_path, artifact_path="config")

        manifest_entries = [
            {"path": checkpoint_path, "artifact_path": "models", "category": "model"},
            {"path": import_summary_path, "artifact_path": "meta", "category": "meta"},
        ]
        for config_path in config_artifacts:
            manifest_entries.append({"path": config_path, "artifact_path": "config", "category": "config"})
        manifest_path = build_artifact_manifest(artifact_dir, manifest_entries)
        mlflow_logger.log_artifact(manifest_path, artifact_path="meta")
        mlflow_logger.end_run(status="FINISHED")
    except Exception:
        mlflow_logger.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    main()
