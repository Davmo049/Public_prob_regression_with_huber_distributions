# Public_prob_regression_with_huber_distributions
This repository contains the code used for the implementation of the paper "Probabilistic Regression with HuberDistributions"

# Requirements:
conda (download from conda.io)
liblapack-dev (sudo apt-get install liblapack-dev)
blas
make
gcc

# setup:
conda install environment.yaml
cd mathlib; make; cd ..
cd ImageTools; make; cd ..

# How to use:
scripts folder contains all significant main files
most important script is scripts/train.py
run by $python -m scripts.train <CONFIG_FILE> --run_name <NAME>
example config files are in the folder configurations/
tensorflow logs and saved weights will be saved in the folder logs/
