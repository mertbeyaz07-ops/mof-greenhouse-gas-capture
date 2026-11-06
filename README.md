# Metal-Organic Frameworks for Greenhouse Gas Capture: Machine Learning-Guided Screening for Simultaneous CO2 and N2O Separation from N2

Data and machine-learning models for the high-throughput screening of CoRE MOFs and QMOFs for N2O capture and
simultaneous N2O + CO2 capture from an N2O/CO2/N2 mixture (Beyaz, Aksu, Keskin).

## Contents

```
data/
  CoREMOF_single_N2O_1976.xlsx          single-component N2O GCMC (1 bar, 298 K), descriptors, ML predictions and metrics
  CoREMOF_mixture_N2O_CO2_N2_1685.xlsx  N2O/CO2/N2 (30/30/40) mixture GCMC at 0.1, 1, 10 bar; WC, S, APS (VSA, PSA);
                                        ML predictions, metrics and SHAP importance
  QMOF_single_N2O_2188.xlsx             single-component N2O GCMC and descriptors of the QMOFs
  QMOF_ML_predictions.xlsx              ML predictions for the unseen QMOFs (single and mixture models), logAPS ranks,
                                        GCMC validation of the top 5 QMOFs per process
ml_data/
  ML_dataset_single_N2O.xlsx            exact inputs of the single-uptake model: 14 descriptors + target
  ML_dataset_mixture.xlsx               exact inputs of the six mixture models: 16 descriptors + 6 targets
models/
  single_N2O_uptake/pipeline.joblib     trained TPOT pipeline used in the paper (+ pipeline.txt)
  mixture/<target>/pipeline.joblib      WC_VSA, logS_VSA, logAPS_VSA, WC_PSA, logS_PSA, logAPS_PSA
code/
  common.py                             descriptor lists, column mapping, data loading, TPOT settings
  train_single_N2O.py                   trains the single-uptake model
  train_mixture.py                      trains the mixture models
  shap_analysis.py                      permutation SHAP analysis
  predict.py                            predictions with the trained pipelines for any MOF table
```

Every Excel file includes a README sheet; descriptor and target column headers include units where applicable. The ML files contain only the
descriptors that the models use, in the order used for training.

## Models

| Model | Descriptors | Target | Training data |
|---|---|---|---|
| Single | C, H, N, O, halogen, metal, nonmetal, metalloid (%), PLD, LCD, porosity, pore volume, KH_N2O, Qst0_N2O (14) | N2O uptake at 1 bar, 298 K | 1976 CoRE MOFs |
| Mixture (6 models) | the 14 above + Qst0_CO2, Qst0_N2 (16) | WC, log10 S and log10 APS of N2O+CO2, VSA and PSA | 1685 CoRE MOFs (PLD ≥ 3.64 Å) |

- Split: 80/20 train/test. `sklearn.model_selection.train_test_split(train_size=0.8, test_size=0.2, random_state=42)`
  on the row order of the CoRE_MOF sheet reproduces the `Set` column.
- TPOT 1.1.0: 10 generations, population 50, 5-fold shuffled KFold CV (random_state 42), negative MSE.
- SHAP: permutation explainer on `pipeline.predict`, 100 background samples from the training set, max_evals 500,
  all CoRE MOFs in 48 chunks (seed 42 + chunk index).
- QMOFs were never used in training: 2188 QMOFs for the single model, 2187 QMOFs (PLD ≥ 3.64 Å) for the mixture models.

## Usage

```bash
pip install -r requirements.txt
cd code

# predictions with the trained models (reproduces data/QMOF_ML_predictions.xlsx)
python predict.py ../ml_data/ML_dataset_mixture.xlsx qmof_mixture.csv --model mixture --sheet QMOF_unseen
python predict.py ../data/QMOF_single_N2O_2188.xlsx qmof_single.csv --model single

# retraining (results/ is created at the repository root)
python train_single_N2O.py --n_jobs 8
python train_mixture.py --targets logAPS_VSA logAPS_PSA --n_jobs 8

# SHAP values of the trained models
python shap_analysis.py single
python shap_analysis.py mixture --targets logAPS_VSA logAPS_PSA
```

For new structures, `predict.py` takes any .xlsx/.csv file with a `MOF` column and the descriptor columns named as in
`ml_data/`.

## Reproducibility

- Loading the files in `ml_data/` returns exactly the descriptor and target values used for training. The saved pipelines in `models/` are provided for prediction and comparison with the
  supplied results. The SHAP code implements the documented analysis settings; exact numerical agreement
  requires the original software environment and should be checked against the supplied results.
- The pipelines were saved with Python 3.12, TPOT 1.1.0 and scikit-learn 1.9.1. They contain scikit-learn, XGBoost and
  TPOT components, so loading them needs the same TPOT and scikit-learn versions and xgboost installed.
- A new TPOT search with the same settings is seeded (random_state 42). With parallel workers (`n_jobs` > 1), however,
  it is not guaranteed to select exactly the same pipeline. The pipelines used in the paper are the ones in `models/`.

## Citation

Authors: Mert Beyaz, Gökhan Önder Aksu, and Seda Keskin.

See `CITATION.cff` for citation metadata. A repository URL and a Zenodo DOI will be added after deposit.

## Archive scope

This release contains processed simulation data, ML input tables, seven trained pipelines, and analysis scripts.
It does not include CIF structures, raw GCMC trajectories, or simulation input decks.

## Reuse

This repository is licensed under the MIT License. See the LICENSE file for details.
