# -*- coding: utf-8 -*-
"""Train the TPOT model for single-component N2O uptake (1 bar, 298 K).

Data   : ml_data/ML_dataset_single_N2O.xlsx (CoRE MOFs for training/testing, QMOFs as an unseen set)
Split  : 80/20 train/test, random_state = 42 (reproduces the 'Set' column)
TPOT   : 10 generations, population 50, 5-fold shuffled KFold CV, negative MSE, random_state = 42
Output : <out_dir>/pipeline.joblib, pipeline.txt, predictions.csv, metrics.json, run_info.json

Usage  : python train_single_N2O.py [--out_dir results/single_N2O_uptake] [--n_jobs 4]
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import ROOT, SINGLE, TPOT_SETTINGS, feature_names, load_core, load_qmof, make_tpot, metrics, versions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT / "results" / "single_N2O_uptake"))
    ap.add_argument("--n_jobs", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    feats = feature_names(SINGLE)
    target = list(SINGLE["targets"].values())[0]
    core, tr, te = load_core(SINGLE)
    qmof = load_qmof(SINGLE)
    print(f"Train {len(tr)} | Test {len(te)} | QMOF (unseen) {len(qmof)} | {len(feats)} descriptors")

    t0 = time.time()
    est = make_tpot(a.n_jobs)
    est.fit(tr[feats], tr[target])
    pipe = est.fitted_pipeline_
    minutes = (time.time() - t0) / 60
    joblib.dump(pipe, out / "pipeline.joblib")
    (out / "pipeline.txt").write_text(str(pipe), encoding="utf-8")

    res, rows = {}, []
    for label, d in [("train", tr), ("test", te), ("QMOF", qmof)]:
        p = pipe.predict(d[feats])
        res[label] = metrics(d[target].values, p)
        rows.append(pd.DataFrame({"MOF": d["MOF"], "set": label, "simulated": d[target].values, "predicted": p}))
        print(f"{label:6s} R2 {res[label]['R2']:.3f} | RMSE {res[label]['RMSE']:.3f} | MAE {res[label]['MAE']:.3f}")
    pd.concat(rows).to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "run_info.json").write_text(json.dumps({
        "features": feats, "target": target, **TPOT_SETTINGS, "random_state": 42, "n_jobs": a.n_jobs,
        "minutes": round(minutes, 1), "versions": versions()}, indent=2), encoding="utf-8")
    print(f"Done in {minutes:.1f} min -> {out}")


if __name__ == "__main__":
    main()
