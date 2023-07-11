################# Code for drawing the corrupted and reconstructed days with DAEs ######################################


import run
import pandas as pd
import numpy as np
import sys
import os
import math
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--path", help="Specify the path")
parser.add_argument("--threshold_q_cool", type=int, help="Enter the threshold for q_cool (0 for Case 1, 50 for Case 2)")
parser.add_argument("--threshold_q_heat", type=int, help="Enter the threshold for q_heat (0 for Case 1, 20 for Case 2")
parser.add_argument("--train_rate", type=float, help="Enter the training rate (0.1, 0.2, 0.3, 0.4, 0.5)")
parser.add_argument("--aug", type=int, help="Enter the augmentation rate (10 for Case 1, 80 for Case 2)")
parser.add_argument("--lambdaa", type=int, help="Enter the lambda value (0 or 1)")
parser.add_argument("--features", type=int, help="Enter the number of features (1 for Univariate_DAE, 3 for Multivariate_DAE_1, 4 for Multivariate_DAE_2 and PI-DAE)")
parser.add_argument("--target", type=str, help="Important if features = 1 (q_cool, q_heat, t_ra)")
parser.add_argument("--corr", type=float, help="Corruption rate (0.2, 0.4, 0.6, 0.8)")

args = parser.parse_args()

path = args.path
threshold_q_cool = args.threshold_q_cool
threshold_q_heat = args.threshold_q_heat
train_rate = args.train_rate
aug = args.aug
lambdaa = args.lambdaa
features = args.features
target = args.target
corr = args.corr

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

# Define the hyperparameters for the model
filters1 = int(df.loc[(df['corr'] == corr) & (df['tar'] == tar), 'filters1'].values[0])
filters2 = int(df.loc[(df['corr'] == corr) & (df['tar'] == tar), 'filters2'].values[0])
filters_size = int(df.loc[(df['corr'] == corr) & (df['tar'] == tar), 'filters_size'].values[0])
lr = df.loc[(df['corr'] == corr) & (df['tar'] == tar), 'lr'].values[0]
batch_size = int(df.loc[(df['corr'] == corr) & (df['tar'] == tar), 'batch_size'].values[0])

dataset_dir = path + '/Data/lbnlbldg59/processed/shuffled_data/multi_feature1_new.pkl'      # directory containing data
results_dir = path + '/Results/seeds'+str(seeds)+'/'                                                     # directory containing the saved models

run.Draw(dataset_dir=dataset_dir, results_dir=results_dir, missing='continuous', corr=corr, train_rate=train_rate, aug=aug,lambdaa=lambdaa, filters1=filters1, filters2=filters2, filters_size=filters_size, strides=1,batch_size=batch_size, features=features, target_=target, threshold_q_cool=threshold_q_cool,threshold_q_heat=threshold_q_heat, print_coeff=False)
