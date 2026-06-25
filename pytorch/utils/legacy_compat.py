import sys
import types


def install_legacy_pickle_compat():
    """
    Provide module aliases needed to unpickle older pandas objects embedded in
    historic PyTorch checkpoints.
    """
    try:
        import pandas as pd
    except Exception:
        return

    module_name = "pandas.core.indexes.numeric"
    if module_name in sys.modules:
        return

    numeric_module = types.ModuleType(module_name)
    try:
        numeric_module.Int64Index = pd.Index
    except Exception:
        pass
    try:
        numeric_module.UInt64Index = pd.Index
    except Exception:
        pass
    try:
        numeric_module.Float64Index = pd.Index
    except Exception:
        pass

    sys.modules[module_name] = numeric_module
