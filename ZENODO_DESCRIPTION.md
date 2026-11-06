# Metal-Organic Frameworks for Greenhouse Gas Capture: Machine Learning-Guided Screening for Simultaneous CO2 and N2O Separation from N2

Authors: Mert Beyaz; Gökhan Önder Aksu; Seda Keskin.

This dataset accompanies a computational study of metal-organic frameworks for single-component N2O adsorption and simultaneous CO2 and N2O separation from N2. It contains processed grand canonical Monte Carlo simulation results, geometric, chemical and energetic descriptors, machine-learning input tables, trained TPOT pipelines, and Python scripts for training, prediction and SHAP analysis.

The single-component datasets contain 1976 CoRE MOFs and 2188 QMOFs at 1 bar and 298 K. The mixture dataset contains 1685 CoRE MOFs evaluated for an N2O/CO2/N2 mixture with mole fractions of 0.30/0.30/0.40 at 0.1, 1 and 10 bar. Mixture performance metrics cover vacuum-swing and pressure-swing adsorption conditions. The release also includes predictions for 2187 unseen QMOFs using the mixture models and 10 GCMC validation rows for selected QMOF candidates.

Seven saved pipelines are provided: one for single-component N2O uptake and six for mixture working capacity, log10 selectivity and log10 adsorbent performance score under VSA and PSA conditions.

See README.md for file organization, software requirements and usage. The archive contains processed data and models, rather than raw simulation trajectories or CIF structures.
