################# Code for training and evaluating  linear interpolation ######################################


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

args = parser.parse_args()

path = args.path
threshold_q_cool = args.threshold_q_cool
threshold_q_heat = args.threshold_q_heat
train_rate = args.train_rate

# Train and evaluate for different corruption rates
for corr in [0.2, 0.4, 0.6, 0.8]:

  # Initialize metrics
  RMSE = np.zeros(3)

  # Evaluate for different random shufflings
  for seeds in [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 123456789, 12345678910]:

    dataset_dir = path + '/Data/lbnlbldg59/processed/shuffled_data/multi_feature'+str(seeds)+'_new.pkl'      # directory containing data

    RMSE_avg = run.LIN(dataset_dir=dataset_dir, missing='continuous', corr=corr, train_rate=train_rate,threshold_q_cool=threshold_q_cool, threshold_q_heat=threshold_q_heat)

    # Stack metrics vertically so that the average can be calculated afterwards
    RMSE = np.vstack((RMSE, RMSE_avg))

  # Print
  print("threshold_q_cool", threshold_q_cool)
  print("threshold_q_heat", threshold_q_heat)
  print("train rate", train_rate)
  print("corr", corr)

  # Print average metrics over different random shufflings (skip the first raw as it was initialized to zero)
  print("RMSE avg", np.mean(RMSE[1:], axis=0))
  print("********************************")
