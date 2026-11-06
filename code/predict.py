# -*- coding: utf-8 -*-
"""Predict with the trained pipelines of the study for any set of MOFs.

The input table (.xlsx or .csv) needs a 'MOF' column (or 'QMOF ID') and the descriptor columns of the model, with the
names used in ml_data/ (e.g. 'PLD (Å)', 'KH_N2O (mol/kg/Pa)'); the model input names of common.py are accepted too.
Single model: 14 descriptors. Mixture models: 16 descriptors (the 14 above + Qst0 of CO2 and N2).

Usage : python predict.py <input.xlsx|csv> <output.csv> [--model single|mixture] [--sheet QMOF_unseen]
Example (reproduces the QMOF predictions of the study):
        python predict.py ../ml_data/ML_dataset_mixture.xlsx qmof_mixture_predictions.csv --model mixture --sheet QMOF_unseen
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from common import MIXTURE, MODELS, SINGLE, to_model_input


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--model", choices=["single", "mixture"], default="mixture")
    ap.add_argument("--sheet", default=0, help="sheet name for Excel input (default: first sheet)")
    ap.add_argument("--models_dir", default=str(MODELS))
    a = ap.parse_args()

    p = Path(a.input)
    df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p, sheet_name=a.sheet)
    name = next(c for c in ["MOF", "QMOF ID", "MOF_Name"] if c in df.columns)
    out = pd.DataFrame({"MOF": df[name]})
    md = Path(a.models_dir)
    if a.model == "single":
        X = to_model_input(df, SINGLE)
        out["pred_N2O_uptake_1bar (mol/kg)"] = joblib.load(md / "single_N2O_uptake" / "pipeline.joblib").predict(X)
    else:
        X = to_model_input(df, MIXTURE)
        for t in MIXTURE["targets"].values():
            out[f"pred_{t}"] = joblib.load(md / "mixture" / t / "pipeline.joblib").predict(X)
        for proc in ["VSA", "PSA"]:
            out[f"rank_logAPS_{proc}"] = out[f"pred_logAPS_{proc}"].rank(ascending=False, method="first").astype(int)
    out.to_csv(a.output, index=False)
    print(f"{len(out)} structures -> {a.output}")


if __name__ == "__main__":
    main()
