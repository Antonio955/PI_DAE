################# Code for printing the saved physics-based coefficients ######################################


import numpy as np
import argparse
parser = argparse.ArgumentParser()

# Define prior to running
parser.add_argument("--path", help="Specify the path")
parser.add_argument("--threshold_q_cool", type=int, help="Enter the threshold for q_cool (0 for Case 1, 50 for Case 2)")
parser.add_argument("--threshold_q_heat", type=int, help="Enter the threshold for q_heat (0 for Case 1, 20 for Case 2")
parser.add_argument("--seeds_coeff", type=int, help="Enter the seeds for the different initialized coefficients (0 [coeff = 1]; 1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 123456789, 1234567899)")

args = parser.parse_args()

missing = 'continuous'

path = args.path
threshold_q_cool = args.threshold_q_cool
threshold_q_heat = args.threshold_q_heat
seeds_coeff = args.seeds_coeff

if seeds_coeff > 0:
  local_random_state = np.random.RandomState(seeds_coeff)
  coeff = local_random_state.random()
  coeff = np.around(coeff, 2)
else:
  coeff = 1

coeff_str = str(coeff).replace('.', '_')  # String for lambda

a_tensor_arr = []
b_tensor_arr = []
c_tensor_arr = []
for train_rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
  a_tensor = []
  b_tensor = []
  c_tensor = []
  for corr in [0.2, 0.4, 0.6, 0.8]:
    for seeds in [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 123456789, 12345678910]:

      if seeds_coeff > 0:
        results_dir = path + '/results/pre_trained_models_' + coeff_str + '/seeds' + str(seeds) + '/'  # directory containing the saved models
      else:
        results_dir = path + '/results/pre_trained_models/seeds' + str(seeds) + '/'  # directory containing the saved models

      name = str(corr).replace('.', '_')
      esp = str(train_rate).replace('.', '_')

      a_tensor = np.append(a_tensor, np.load(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/multivariate__t_oa/' + missing + '/' + name + '/' + 'best_a_tensor.npy'))
      b_tensor = np.append(b_tensor, np.load(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/multivariate__t_oa/' + missing + '/' + name + '/' + 'best_b_tensor.npy'))
      c_tensor = np.append(c_tensor, np.load(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/multivariate__t_oa/' + missing + '/' + name + '/' + 'best_c_tensor.npy'))

  a_tensor_avg = np.mean(a_tensor)
  b_tensor_avg = np.mean(b_tensor)
  c_tensor_avg = np.mean(c_tensor)

  a_tensor_arr = np.append(a_tensor_arr, a_tensor_avg)
  b_tensor_arr = np.append(b_tensor_arr, b_tensor_avg)
  c_tensor_arr = np.append(c_tensor_arr, c_tensor_avg)

np.save(path + '/results/coefficients/a_tensor_' + coeff_str + '_' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '.npy', a_tensor_arr)
np.save(path + '/results/coefficients/b_tensor_' + coeff_str + '_' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '.npy', b_tensor_arr)
np.save(path + '/results/coefficients/c_tensor_' + coeff_str + '_' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '.npy', c_tensor_arr)

print('a', a_tensor_arr)
print('b', b_tensor_arr)
print('c', c_tensor_arr)

