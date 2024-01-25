################# Code for training  DAEs ######################################


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
parser.add_argument("--seeds_coeff", type=int, help="Enter the seeds for the different initialized coefficients (0 [coeff = 1]; 1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 123456789, 1234567899)")
parser.add_argument("--aug", type=int, help="Enter the augmentation rate (10 for Case 1, 80 for Case 2)")
parser.add_argument("--lambdaa", type=int, help="Enter the lambda value (0 or 1)")
parser.add_argument("--features", type=int, help="Enter the number of features (1 for Univariate_DAE, 3 for Multivariate_DAE_1, 4 for Multivariate_DAE_2 and PI-DAE)")
parser.add_argument("--target", type=str, help="Important if features = 1 (q_cool, q_heat, t_ra)")

args = parser.parse_args()

path = args.path
threshold_q_cool = args.threshold_q_cool
threshold_q_heat = args.threshold_q_heat
train_rate = args.train_rate
aug = args.aug
lambdaa = args.lambdaa
features = args.features
seeds_coeff = args.seeds_coeff
target = args.target

if seeds_coeff > 0:
  local_random_state = np.random.RandomState(seeds_coeff)
  coeff = local_random_state.random()
  coeff = np.around(coeff, 2)
else:
  coeff = 1

coeff_str = str(coeff).replace('.', '_')  # String for lambda

# Define a variable tar to select the hyperparameters
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
df = pd.read_csv(path + '/results/Tuning.csv')

# Initialize the physics-based coefficients to one
a = coeff
b = coeff
c = coeff

# Train and evaluate for different corruption rates
for corr in [0.2, 0.4, 0.6, 0.8]:

  # Define the hyperparameters for the model
  filters1 = int(df.loc[(df['corr'] == corr) & (df['tar'] == tar), 'filters1'].values[0])
  filters2 = int(df.loc[(df['corr'] == corr) & (df['tar'] == tar), 'filters2'].values[0])
  filters_size = int(df.loc[(df['corr'] == corr) & (df['tar'] == tar), 'filters_size'].values[0])
  lr = df.loc[(df['corr'] == corr) & (df['tar'] == tar), 'lr'].values[0]
  batch_size = int(df.loc[(df['corr'] == corr) & (df['tar'] == tar), 'batch_size'].values[0])

  # Initialize running time
  running_time = []

  # Train for different random shufflings
  for seeds in [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 123456789, 12345678910]:

    dataset_dir = path + '/processed_data/shuffled_data/matrix_'+str(seeds)+'.pkl'       # directory containing data

    if seeds_coeff > 0:
      results_dir = path + '/results/pre_trained_models_' + coeff_str + '/seeds' + str(seeds) + '/'  # directory containing the saved models
    else:
      results_dir = path + '/results/pre_trained_models/seeds' + str(seeds) + '/'  # directory containing the saved models

    timee_ = run.Training(dataset_dir=dataset_dir, results_dir=results_dir, missing='continuous', corr=corr, train_rate=train_rate,aug=aug, lambdaa=lambdaa, a=a, b=b, c=c, filters1=filters1, filters2=filters2,filters_size=filters_size, strides=1, lr=lr, batch_size=batch_size, epochs=2000, features=features,target_=target, threshold_q_cool=threshold_q_cool, threshold_q_heat=threshold_q_heat)
    running_time = np.append(running_time, timee_)

  # Print
  print("threshold_q_cool", threshold_q_cool)
  print("threshold_q_heat", threshold_q_heat)
  print("train rate", train_rate)
  print("aug", aug)
  print("target", tar)
  print("lambdaa", lambdaa)
  print("corr", corr)

  # Print average metrics over different random shufflings
  print("running timee avg [s]", np.mean(running_time))
  print("********************************")
