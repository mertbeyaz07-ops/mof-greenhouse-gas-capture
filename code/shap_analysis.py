# -*- coding: utf-8 -*-
"""Permutation SHAP analysis of the trained pipelines (settings of the study).

Explainer : shap.explainers.Permutation on pipeline.predict
Background: training set, shap.maskers.Independent(max_samples = 100)
Explained : all CoRE MOFs (row order of the ML Excel file), max_evals = 500,
            split into 48 chunks; chunk k uses seed 42 + k (chunks run in parallel)
Output    : <out_dir>/<model>/shap_values.csv and shap_importance.csv (mean |SHAP|)

Usage : python shap_analysis.py single
        python shap_analysis.py mixture [--targets logAPS_VSA logAPS_PSA]
        --models_dir points to a folder with the same layout as models/ (default: models/ of this repository)
"""
import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import MIXTURE, MODELS, ROOT, SINGLE, feature_names, load_core

warnings.filterwarnings("ignore")
BACKGROUND, MAX_EVALS, SEED, N_CHUNKS = 100, 500, 42, 48


def _chunk(pipe_path, X, bg, seed):
    import warnings
    warnings.filterwarnings("ignore")
    import joblib
    import shap
    pipe = joblib.load(pipe_path)
    masker = shap.maskers.Independent(bg, max_samples=BACKGROUND)
    expl = shap.explainers.Permutation(pipe.predict, masker, seed=seed)
    return expl(X, max_evals=MAX_EVALS, silent=True).values


def run(pipe_path, spec, n_jobs, out_dir):
    feats = feature_names(spec)
    core, _, _ = load_core(spec)
    bg = core.loc[core["Set"] == "train", feats]
    idx = np.array_split(np.arange(len(core)), N_CHUNKS)
    parts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_chunk)(str(pipe_path), core[feats].iloc[ix], bg, SEED + k) for k, ix in enumerate(idx))
    sv = np.vstack(parts)
    tdir = Path(out_dir)
    tdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sv, columns=feats).assign(MOF=core["MOF"].values).to_csv(tdir / "shap_values.csv", index=False)
    imp = pd.Series(np.abs(sv).mean(0), index=feats).sort_values(ascending=False).rename("mean_abs_SHAP")
    imp.to_csv(tdir / "shap_importance.csv")
    print(f"{tdir.name}:\n{imp.round(4).to_string()}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["single", "mixture"])
    ap.add_argument("--targets", nargs="+", default=list(MIXTURE["targets"].values()))
    ap.add_argument("--models_dir", default=str(MODELS))
    ap.add_argument("--out_dir", default=str(ROOT / "results" / "shap"))
    ap.add_argument("--n_jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    md = Path(a.models_dir)
    if a.kind == "single":
        run(md / "single_N2O_uptake" / "pipeline.joblib", SINGLE, a.n_jobs, Path(a.out_dir) / "single_N2O_uptake")
    else:
        for t in a.targets:
            run(md / "mixture" / t / "pipeline.joblib", MIXTURE, a.n_jobs, Path(a.out_dir) / t)


if __name__ == "__main__":
    main()
