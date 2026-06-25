import json
import os
import numpy as np


class MLflowLogger:

    def __init__(self, enabled=False, tracking_uri=None, experiment_name=None, run_name=None, tags=None, nested=False):
        self.enabled = enabled
        self.active = False
        self.mlflow = None
        self.client = None
        self.run_id = None
        self.experiment_id = None
        self.run_name = run_name
        self.tags = tags or {}
        self.nested = nested

        if not self.enabled:
            return

        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                "MLflow logging is enabled, but the 'mlflow' package is not installed."
            ) from exc

        self.mlflow = mlflow
        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)
        self.client = self.mlflow.tracking.MlflowClient()
        if experiment_name:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment and experiment.lifecycle_stage == "deleted":
                self.client.restore_experiment(experiment.experiment_id)
                experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                self.client.create_experiment(experiment_name)
            self.mlflow.set_experiment(experiment_name)
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is not None:
                self.experiment_id = experiment.experiment_id

    def _to_scalar(self, value):
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            try:
                return value.item()
            except (TypeError, ValueError):
                pass

        arr = np.array(value)
        if arr.size != 1:
            return None

        scalar = arr.reshape(-1)[0]
        if isinstance(scalar, np.generic):
            scalar = scalar.item()
        return scalar

    def _to_param_value(self, value):
        scalar = self._to_scalar(value)
        if scalar is not None:
            return scalar
        return json.dumps(value)

    def start_run(self, run_name=None, tags=None, nested=None):
        if not self.enabled or self.active:
            return

        run_name = run_name or self.run_name
        run_tags = self.tags.copy()
        if tags:
            run_tags.update(tags)
        if nested is None:
            nested = self.nested

        run = self.mlflow.start_run(run_name=run_name, nested=nested)
        if run is not None:
            self.run_id = run.info.run_id
        if run_tags:
            self.mlflow.set_tags(run_tags)
        self.active = True

    def log_params(self, params):
        if not self.enabled or not self.active:
            return

        cleaned_params = {}
        for key, value in params.items():
            if value is None:
                continue
            cleaned_params[key] = self._to_param_value(value)

        if cleaned_params:
            try:
                self.mlflow.log_params(cleaned_params)
            except Exception as exc:
                print(f"MLflow parameter logging failed: {exc}")

    def log_metrics(self, metrics, step=None, prefix=None):
        if not self.enabled or not self.active:
            return

        cleaned_metrics = {}
        for key, value in metrics.items():
            scalar = self._to_scalar(value)
            if scalar is None:
                continue
            if isinstance(scalar, (float, np.floating)) and (np.isnan(scalar) or np.isinf(scalar)):
                continue
            metric_key = f"{prefix}_{key}" if prefix else key
            cleaned_metrics[metric_key] = float(scalar)

        if cleaned_metrics:
            try:
                self.mlflow.log_metrics(cleaned_metrics, step=step)
            except Exception as exc:
                print(f"MLflow metric logging failed: {exc}")

    def log_artifact(self, local_path, artifact_path=None):
        if not self.enabled or not self.active or not local_path:
            return
        if os.path.exists(local_path):
            try:
                self.mlflow.log_artifact(local_path, artifact_path=artifact_path)
            except Exception as exc:
                print(f"MLflow artifact logging failed for '{local_path}': {exc}")

    def log_artifacts(self, local_dir, artifact_path=None):
        if not self.enabled or not self.active or not local_dir:
            return
        if os.path.isdir(local_dir):
            try:
                self.mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
            except Exception as exc:
                print(f"MLflow artifacts logging failed for '{local_dir}': {exc}")

    def log_dict(self, payload, artifact_file):
        if not self.enabled or not self.active or not artifact_file:
            return
        try:
            self.mlflow.log_dict(payload, artifact_file)
        except Exception as exc:
            print(f"MLflow dict logging failed for '{artifact_file}': {exc}")

    def log_text(self, text, artifact_file):
        if not self.enabled or not self.active or not artifact_file:
            return
        try:
            self.mlflow.log_text(text, artifact_file)
        except Exception as exc:
            print(f"MLflow text logging failed for '{artifact_file}': {exc}")

    def set_tags(self, tags):
        if not self.enabled or not self.active or not tags:
            return
        cleaned_tags = {key: str(value) for key, value in tags.items() if value is not None}
        if cleaned_tags:
            try:
                self.mlflow.set_tags(cleaned_tags)
            except Exception as exc:
                print(f"MLflow tag logging failed: {exc}")

    def log_pytorch_model(self, model, artifact_path):
        if not self.enabled or not self.active or model is None or not artifact_path:
            return False
        model_name = str(artifact_path)
        for invalid_char in ['/', ':', '.', '%', '"', "'"]:
            model_name = model_name.replace(invalid_char, "_")
        model_name = model_name.strip("_") or "model"
        try:
            try:
                # MLflow 3.x prefers `name`; older versions only accept `artifact_path`.
                self.mlflow.pytorch.log_model(model, name=model_name)
            except TypeError:
                self.mlflow.pytorch.log_model(model, artifact_path=model_name)
            return True
        except Exception as exc:
            print(f"MLflow PyTorch model logging failed for '{model_name}': {exc}")
            return False

    def end_run(self, status="FINISHED"):
        if not self.enabled or not self.active:
            return
        try:
            self.mlflow.end_run(status=status)
        finally:
            self.run_id = None
        self.active = False
