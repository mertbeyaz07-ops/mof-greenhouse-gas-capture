# -*- coding: utf-8 -*-
"""Shared settings and data loading for the TPOT models.

The Excel files in ml_data/ use descriptive column names with units. The trained pipelines in models/ were fitted
on pandas DataFrames with the short input names below, so the data are renamed to these names before fitting or
predicting. Column order matters and is fixed here.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ML_DATA = ROOT / "ml_data"
MODELS = ROOT / "models"

RANDOM_STATE = 42
TRAIN_SIZE, TEST_SIZE = 0.80, 0.20
TPOT_SETTINGS = dict(generations=10, population_size=50, cv_folds=5)

# Excel column name -> model input name (in the order used for training)
_BASE = {
    "C (%)": "C_%", "H (%)": "H_%", "N (%)": "N_%", "O (%)": "O_%", "Halogen (%)": "Halogen_%",
    "Metal (%)": "Metal_%", "Nonmetal (%)": "Nonmetal_%", "Metalloid (%)": "Metalloid_%",
    "PLD (Å)": "PLD (A)", "LCD (Å)": "LCD (A)", "Porosity": "Porosity", "Pore volume (cm3/g)": "Pore Volume (cm3/g)",
}
SINGLE = {
    "file": ML_DATA / "ML_dataset_single_N2O.xlsx",
    "features": {**_BASE, "KH_N2O (mol/kg/Pa)": "KH", "Qst0_N2O (kJ/mol)": "Qst0_kJmol"},
    "targets": {"N2O_uptake_1bar (mol/kg)": "N2O_uptake_1bar_molkg"},
}
MIXTURE = {
    "file": ML_DATA / "ML_dataset_mixture.xlsx",
    "features": {**_BASE, "KH_N2O (mol/kg/Pa)": "KH_N2O", "Qst0_N2O (kJ/mol)": "Qst0_N2O",
                 "Qst0_CO2 (kJ/mol)": "Qst0_CO2", "Qst0_N2 (kJ/mol)": "Qst0_N2"},
    "targets": {"WC_VSA (mol/kg)": "WC_VSA", "logS_VSA": "logS_VSA", "logAPS_VSA": "logAPS_VSA",
                "WC_PSA (mol/kg)": "WC_PSA", "logS_PSA": "logS_PSA", "logAPS_PSA": "logAPS_PSA"},
}


def _rename(df, spec):
    return df.rename(columns={**spec["features"], **spec["targets"]})


def feature_names(spec):
    return list(spec["features"].values())


def load_core(spec):
    """CoRE MOF table (model input names) and its train / test parts in the order used for training."""
    from sklearn.model_selection import train_test_split
    core = _rename(pd.read_excel(spec["file"], sheet_name="CoRE_MOF"), spec)
    num = feature_names(spec) + list(spec["targets"].values())
    core[num] = core[num].astype(float)
    tr_idx, te_idx = train_test_split(np.arange(len(core)), train_size=TRAIN_SIZE, test_size=TEST_SIZE,
                                      random_state=RANDOM_STATE, shuffle=True)
    # the split above reproduces the 'Set' column of the Excel file
    assert (core["Set"].iloc[tr_idx] == "train").all() and (core["Set"].iloc[te_idx] == "test").all()
    tr = core.iloc[tr_idx].reset_index(drop=True)
    te = core.iloc[te_idx].reset_index(drop=True)
    return core, tr, te


def load_qmof(spec):
    q = _rename(pd.read_excel(spec["file"], sheet_name="QMOF_unseen"), spec)
    num = [c for c in feature_names(spec) + list(spec["targets"].values()) if c in q.columns]
    q[num] = q[num].astype(float)
    return q


def to_model_input(df, spec):
    """Accepts either Excel names or model input names; returns the feature matrix in the training order."""
    df = _rename(df, spec)
    feats = feature_names(spec)
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise KeyError(f"Missing descriptor columns: {missing}")
    return df[feats].apply(pd.to_numeric, errors="raise").astype(float)


def metrics(y, p):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {"R2": float(r2_score(y, p)), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p)), "n": int(len(y))}


def make_tpot(n_jobs):
    """TPOT 1.x regressor with the settings used in the study."""
    from sklearn.model_selection import KFold
    from tpot import TPOTRegressor
    cv = KFold(n_splits=TPOT_SETTINGS["cv_folds"], shuffle=True, random_state=RANDOM_STATE)
    return TPOTRegressor(generations=TPOT_SETTINGS["generations"], population_size=TPOT_SETTINGS["population_size"],
                         cv=cv, random_state=RANDOM_STATE, n_jobs=n_jobs, scorers=["neg_mean_squared_error"],
                         scorers_weights=[1], max_time_mins=None, verbose=2)


def versions():
    import platform

    import sklearn
    import tpot
    return {"tpot": tpot.__version__, "scikit-learn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "python": platform.python_version()}
