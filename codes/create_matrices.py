################# Create dataset with shuffled days for different seed ######################################


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle5 as pickle
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--input_data", help="Specify the input data")
parser.add_argument("--output_directory", help="Specify the output directory")
parser.add_argument("--seeds", type=int, help="Enter the seeds among 1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 123456789, 12345678910")

args = parser.parse_args()

dataset = pd.read_csv(args.input_data)
path_ = args.output_directory
seeds = args.seeds

# Create an index for the 363 days in the dataset
indice = np.arange(363)

# Create a second index for the 363 days and convert it to a day-to-day matrix with 48 observations per day
multi_indice = np.reshape(np.arange(363*48),(363,48))

# Fix the seeds for the random function
random.seed(seeds)

# Shuffle the days based on the set seeds
random.shuffle(indice)
multi_indice = multi_indice[indice]

# Convert multi_indice to a single column
multi_indice = np.reshape(multi_indice,363*48)

# Extract outdoor air temperature column
t_oa = dataset[dataset.series == 't_oa_avg [°C]']
t_oa_value = t_oa['value'].to_numpy()

# Extract timestamp column
t_oa_timestamp = t_oa['timestamp'].to_numpy()

# Extract cooling flow rate column
Q_cool = dataset[dataset.series == 'Q_cool [kW]']
Q_cool_value = Q_cool['value'].to_numpy()

# Extract heating flow rate column
Q_heat = dataset[dataset.series == 'Q_heat [kW]']
Q_heat_value = Q_heat['value'].to_numpy()

# Extract indoor air temperature column
t_ra = dataset[dataset.series == 't_ra_avg [°C]']
t_ra_value = t_ra['value'].to_numpy()

# Create dataset with shuffled days
multi_feature = np.transpose(np.vstack((t_oa_timestamp, t_oa_value, Q_cool_value, Q_heat_value, t_ra_value)))

# Save to csv and pkl
Path(path_).mkdir(parents=True, exist_ok=True)
np.savetxt(path_+'matrix_'+str(seeds)+'.csv', multi_feature[multi_indice], delimiter=',', fmt='%s')
with open(path_+'matrix_'+str(seeds)+'.pkl', 'wb') as handle:
    pickle.dump(multi_feature[multi_indice], handle, protocol=pickle.HIGHEST_PROTOCOL)
