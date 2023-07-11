################# Code for tuning  DAEs ######################################


import run
import pandas as pd
import numpy as np
import sys
import os
import math
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--path", help="Specify the path")
parser.add_argument("--lambdaa", type=int, help="Enter the lambda value (0 or 1)")
parser.add_argument("--features", type=int, help="Enter the number of features (1 for Univariate_DAE, 3 for Multivariate_DAE_1, 4 for Multivariate_DAE_2 and PI-DAE)")
parser.add_argument("--target", type=str, help="Important if features = 1 (q_cool, q_heat, t_ra)")
parser.add_argument("--corr", type=float, help="Corruption rate (0.2, 0.4, 0.6, 0.8)")

args = parser.parse_args()

path = args.path
lambdaa = args.lambdaa
features = args.features
target = args.target
corr = args.corr

dataset_dir = path + '/Data/lbnlbldg59/processed/shuffled_data/multi_feature1_new.pkl'      # directory containing data

# Define a variable tar to select the hyperparameters and feauture_ to define the size of the evaluation metrics
tar = None
if features == 1 and target == 'q_cool':
  tar = 'univariate__q_cool'
elif features == 1 and target == 'q_heat':
  tar = 'univariate__q_heat'
elif features == 1 and target == 't_ra':
  tar = 'univariate__t_ra'
elif features == 3:
  tar = 'multivariate'
elif features == 4:
  tar = 'multivariate__t_oa'
else:
  print("error")

# Read csv containing hyperparameters
df = pd.read_csv(path + '/Results/Tuning/Tuning.csv')

# Initialize the physics-based coefficients to one
a = 1
b = 1
c = 1

run.Tuning(dataset_dir=dataset_dir, missing='continuous', corr=corr, train_rate=0.025, aug=80, lambdaa=lambdaa, a=a, b=b, c=c, strides=1, epochs=2000, features=features, target_=target, threshold_q_cool=0, threshold_q_heat=0)