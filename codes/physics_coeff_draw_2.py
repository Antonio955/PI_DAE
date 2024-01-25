import matplotlib.pyplot as plt
import numpy as np
import argparse
parser = argparse.ArgumentParser()
#****************** plot figures with computer modern font ************************
import matplotlib.font_manager as font_manager
import matplotlib
matplotlib.rcParams['font.family']='serif'
matplotlib.rcParams['font.serif']=['Times New Roman'] + plt.rcParams['font.serif']
matplotlib.rcParams['mathtext.fontset']='cm'
matplotlib.rcParams['axes.unicode_minus']=False
#**********************************************************************************
parser.add_argument("--path", help="Specify the path")
args = parser.parse_args()
path = args.path

coeffs_str = ['0_93', '0_53', '0_42', '0_25', '0_24', '0_19', '0_15', '0_13', '0_7', '0_6']
coeffs_arr = np.asarray([0.93, 0.53, 0.42, 0.25, 0.24, 0.19, 0.15, 0.13, 0.7, 0.6])

case1_a_arr = []
case1_b_arr = []
case1_c_arr = []
case2_a_arr = []
case2_b_arr = []
case2_c_arr = []

for c in coeffs_str:

    case1_a = np.load(path + '/results/coefficients/a_tensor_' + c + '_0_0.npy')
    case1_b = np.load(path + '/results/coefficients/b_tensor_' + c + '_0_0.npy')
    case1_c = np.load(path + '/results/coefficients/c_tensor_' + c + '_0_0.npy')

    case2_a = np.load(path + '/results/coefficients/a_tensor_' + c + '_50_20.npy')
    case2_b = np.load(path + '/results/coefficients/b_tensor_' + c + '_50_20.npy')
    case2_c = np.load(path + '/results/coefficients/c_tensor_' + c + '_50_20.npy')

    case1_a_arr = np.append(case1_a_arr, case1_a[-1])
    case1_b_arr = np.append(case1_b_arr, case1_b[-1])
    case1_c_arr = np.append(case1_c_arr, case1_c[-1])

    case2_a_arr = np.append(case2_a_arr, case2_a[-1])
    case2_b_arr = np.append(case2_b_arr, case2_b[-1])
    case2_c_arr = np.append(case2_c_arr, case2_c[-1])

n = 20

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

print(np.shape(coeffs_arr))
axs[0].scatter(np.arange(10), coeffs_arr, c='black', marker='o', label='Starting point')
axs[0].scatter(np.arange(10), case1_a_arr, c='blue', marker='x', label='a')
axs[0].scatter(np.arange(10), case1_b_arr, c='orange', marker='x', label='b')
axs[0].scatter(np.arange(10), case1_c_arr, c='green', marker='x', label='c')
# Plotting arrows from initial to final points
for j in range(10):
        axs[0].arrow(j, coeffs_arr[j],0, case1_a_arr[j] - coeffs_arr[j],color='gray', linestyle='dashed', linewidth=0.5, head_width=0.2, head_length=0.02)
        axs[0].arrow(j, coeffs_arr[j],0, case1_b_arr[j] - coeffs_arr[j],color='gray', linestyle='dashed', linewidth=0.5, head_width=0.2, head_length=0.02)
        axs[0].arrow(j, coeffs_arr[j],0, case1_c_arr[j] - coeffs_arr[j],color='gray', linestyle='dashed', linewidth=0.5, head_width=0.2, head_length=0.02)

axs[0].axhline(y=0.106, color='blue', linestyle='--')
axs[0].axhline(y=0.021, color='orange', linestyle='--')
axs[0].axhline(y=0.079, color='green', linestyle='--')

axs[0].set_xlabel('Trial', fontsize=n)
axs[0].set_ylabel('Coefficient', fontsize=n)
axs[0].set_title("Case 1", fontsize=n)
axs[0].set_ylim([0, 1.3])
axs[0].tick_params(axis='both', which='major', labelsize=n)



axs[1].scatter(np.arange(10), coeffs_arr, c='black', marker='o', label='Starting point')
axs[1].scatter(np.arange(10), case2_a_arr, c='blue', marker='x', label='a')
axs[1].scatter(np.arange(10), case2_b_arr, c='orange', marker='x', label='b')
axs[1].scatter(np.arange(10), case2_c_arr, c='green', marker='x', label='c')
# Plotting arrows from initial to final points
for j in range(10):
        axs[1].arrow(j, coeffs_arr[j],0, case2_a_arr[j] - coeffs_arr[j],color='gray', linestyle='dashed', linewidth=0.5, head_width=0.2, head_length=0.02)
        axs[1].arrow(j, coeffs_arr[j],0, case2_b_arr[j] - coeffs_arr[j],color='gray', linestyle='dashed', linewidth=0.5, head_width=0.2, head_length=0.02)
        axs[1].arrow(j, coeffs_arr[j],0, case2_c_arr[j] - coeffs_arr[j],color='gray', linestyle='dashed', linewidth=0.5, head_width=0.2, head_length=0.02)

axs[1].axhline(y=1.089, color='blue', linestyle='--')
axs[1].axhline(y=0.267, color='orange', linestyle='--')
axs[1].axhline(y=0.353, color='green', linestyle='--')

axs[1].set_xlabel('Trial', fontsize=n)
axs[1].set_ylabel('Coefficient', fontsize=n)
axs[1].set_title("Case 2", fontsize=n)
axs[1].set_ylim([0, 1.3])
axs[1].tick_params(axis='both', which='major', labelsize=n)
axs[1].legend(loc='center left', fontsize=n, bbox_to_anchor=(1, 0.5))

#plt.subplots_adjust(wspace=0.4)
plt.subplots_adjust(top=0.85)  # Modify the top parameter to reduce the space
plt.subplots_adjust(hspace=0.5)  # Increase the hspace value to add extra space between subplots
plt.show()