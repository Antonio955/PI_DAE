################# Code for printing the saved physics-based coefficients ######################################


import numpy as np
import argparse
parser = argparse.ArgumentParser()

# Define prior to running
parser.add_argument("--path", help="Specify the path")
parser.add_argument("--threshold_q_cool", type=int, help="Enter the threshold for q_cool (0 for Case 1, 50 for Case 2)")
parser.add_argument("--threshold_q_heat", type=int, help="Enter the threshold for q_heat (0 for Case 1, 20 for Case 2")
parser.add_argument("--train_rate", type=float, help="Enter the training rate (0.1, 0.2, 0.3, 0.4, 0.5)")

args = parser.parse_args()

path = args.path
threshold_q_cool = args.threshold_q_cool
threshold_q_heat = args.threshold_q_heat
train_rate = args.train_rate

a_tensor = []
b_tensor = []
c_tensor = []
for corr in [0.2,0.4,0.6,0.8]:
  for seeds in [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 123456789, 12345678910]:

    results_dir = path + '/Results/seeds'+str(seeds)+'/'                                                     # directory containing the saved models

    name = str(corr).replace('.', '_')
    esp = str(train_rate).replace('.', '_')

    a_tensor = np.append(a_tensor, np.load(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/multivariate__t_oa/' + missing + '/' + name + '/' + 'best_a_tensor.npy'))
    b_tensor = np.append(b_tensor, np.load(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/multivariate__t_oa/' + missing + '/' + name + '/' + 'best_b_tensor.npy'))
    c_tensor = np.append(c_tensor, np.load(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/multivariate__t_oa/' + missing + '/' + name + '/' + 'best_c_tensor.npy'))

print("a_tensor", np.mean(a_tensor))
print("b_tensor", np.mean(b_tensor))
print("c_tensor", np.mean(c_tensor))