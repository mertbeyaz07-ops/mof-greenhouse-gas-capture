# -*- coding: utf-8 -*-
"""Train the six TPOT models for N2O+CO2 capture from the N2O/CO2/N2 (30/30/40) mixture.

Targets: WC_VSA, WC_PSA (N2O+CO2 working capacity, mol/kg), logS_VSA, logS_PSA (log10 selectivity),
         logAPS_VSA, logAPS_PSA (log10 adsorbent performance score)
Data   : ml_data/ML_dataset_mixture.xlsx (1685 CoRE MOFs for training/testing, 2187 unseen QMOFs)
Split  : 80/20 train/test, random_state = 42 (reproduces the 'Set' column)
TPOT   : 10 generations, population 50, 5-fold shuffled KFold CV, negative MSE, random_state = 42
Output : <out_dir>/<target>/pipeline.joblib, pipeline.txt, predictions_core.csv, metrics.json
         <out_dir>/QMOF_predictions.csv (all trained targets + logAPS ranks)

Usage  : python train_mixture.py [--targets logAPS_VSA logAPS_PSA] [--out_dir results/mixture] [--n_jobs 4]
         Each target is independent; targets can be trained in separate processes.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import MIXTURE, ROOT, TPOT_SETTINGS, feature_names, load_core, load_qmof, make_tpot, metrics, versions

TARGETS = list(MIXTURE["targets"].values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=TARGETS, choices=TARGETS)
    ap.add_argument("--out_dir", default=str(ROOT / "results" / "mixture"))
    ap.add_argument("--n_jobs", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    feats = feature_names(MIXTURE)
    core, tr, te = load_core(MIXTURE)
    qmof = load_qmof(MIXTURE)
    print(f"Train {len(tr)} | Test {len(te)} | QMOF (unseen) {len(qmof)} | {len(feats)} descriptors")

    qpred = pd.DataFrame({"MOF": qmof["MOF"]})
    for t in a.targets:
        tdir = out / t
        tdir.mkdir(exist_ok=True)
        print(f"\n=== {t} ({time.strftime('%H:%M:%S')})")
        t0 = time.time()
        est = make_tpot(a.n_jobs)
        est.fit(tr[feats], tr[t])
        pipe = est.fitted_pipeline_
        minutes = (time.time() - t0) / 60
        joblib.dump(pipe, tdir / "pipeline.joblib")
        (tdir / "pipeline.txt").write_text(str(pipe), encoding="utf-8")
        res, rows = {}, []
        for label, d in [("train", tr), ("test", te)]:
            p = pipe.predict(d[feats])
            res[label] = metrics(d[t].values, p)
            rows.append(pd.DataFrame({"MOF": d["MOF"], "set": label, "simulated": d[t].values, "predicted": p}))
        pd.concat(rows).to_csv(tdir / "predictions_core.csv", index=False)
        (tdir / "metrics.json").write_text(json.dumps({**res, "minutes": round(minutes, 1)}, indent=2), encoding="utf-8")
        qpred[t] = pipe.predict(qmof[feats])
        print(f"{t}: train R2 {res['train']['R2']:.3f} | test R2 {res['test']['R2']:.3f}, RMSE {res['test']['RMSE']:.3f}, "
              f"MAE {res['test']['MAE']:.3f} ({minutes:.1f} min)")

    for proc in ["VSA", "PSA"]:
        if f"logAPS_{proc}" in qpred:
            qpred[f"rank_logAPS_{proc}"] = qpred[f"logAPS_{proc}"].rank(ascending=False, method="first").astype(int)
    qpred.to_csv(out / "QMOF_predictions.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({
        "features": feats, "targets": a.targets, **TPOT_SETTINGS, "random_state": 42, "n_jobs": a.n_jobs,
        "n_train": len(tr), "n_test": len(te), "n_qmof": len(qmof), "versions": versions()}, indent=2), encoding="utf-8")
    print(f"\nDone -> {out}")


if __name__ == "__main__":
    main()
