# Maritime Forest Ecosystem and CANOPY-TD

This repository contains code, experimental outputs, and publication-ready figures for the **CANOPY-TD** model, the first computational realization of the **Maritime Forest Ecosystem (MFE)** framework for AIS-based maritime traffic self-organization.

The Maritime Forest Ecosystem interprets maritime traffic as a heterogeneous adaptive system inspired by natural forest organization. Vessel trajectories are represented as adaptive maritime canopies competing for navigational space through local density, canopy overlap, ecosystem pressure, and trajectory redistribution.

## Repository contents

```text
src/        Python implementation of CANOPY-TD and evaluation utilities
data/       Dataset access notes and a small synthetic AIS-like sample
results/    Calculated experimental metrics
figures/    Publication-ready figures generated from AIS trajectories and CANOPY-TD results
paper/      Short notes for manuscript reproducibility
docs/       Formula notes and method description
```

## Dataset

The original Ushant TSS AIS dataset is **not redistributed** in this repository due to licensing and data ownership restrictions. Please obtain the original AIS data from the original data provider.

A small synthetic AIS-like sample is included only for testing the code structure.

## Baseline

An independent genetic algorithm (GA) baseline is included in `src/baseline_ga.py`.
The GA operates on the same search space as CANOPY-TD (a 2D displacement vector
for each vessel) and is subject to the same maximum cumulative offset constraint.
Fitness = Conflicts + lambda * MeanRouteOffset. GA parameters: population 80,
200 generations, tournament selection (size 3), blend crossover, Gaussian mutation,
elitism of 2. Five runs with different seeds; the best result is kept.
Unlike CANOPY-TD, the GA does not use canopy overlap, local density, or ecosystem pressure.

## Quick start

```bash
pip install -r requirements.txt
python src/run_experiment.py --data_dir data/sample --out_dir results
python src/visualization.py --results_csv results/CANOPY_TD_results.csv --out_png figures/Figure_8_demo.png
python src/test_ga.py
python src/tune_lambda.py --data_dir data/sample --max_offset_km 5.0 --n_runs 3
```

## Main reported result

CANOPY-Strong reduced potential conflicts from 427 to 120, corresponding to approximately **71.9% conflict reduction** in the processed Ushant TSS proof-of-concept experiment.

## License

Code is released under the MIT License. Raw AIS data are not redistributed.
