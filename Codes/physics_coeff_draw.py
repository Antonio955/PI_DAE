################# Code for drawing the physics-based coefficients ######################################


import matplotlib.pyplot as plt
import numpy as np

#****************** plot figures with computer modern font ************************
import matplotlib.font_manager as font_manager
import matplotlib
matplotlib.rcParams['font.family']='serif'
matplotlib.rcParams['font.serif']=['Times New Roman'] + plt.rcParams['font.serif']
matplotlib.rcParams['mathtext.fontset']='cm'
matplotlib.rcParams['axes.unicode_minus']=False
#**********************************************************************************

# Please refer to results.csv (coefficients sheet)
# Values obtained running physics_coeff_print.py
case1_qcool = np.asarray([0.196, 0.145, 0.095, 0.112, 0.106])
case1_qheat = np.asarray([0.050, 0.011, 0.006, 0.023, 0.021])
case1_tra = np.asarray([0.146, 0.116, 0.085, 0.080, 0.079])
case2_qcool = np.asarray([1.226, 1.154, 1.108, 1.084, 1.089])
case2_qheat = np.asarray([0.179, 0.194, 0.259, 0.264, 0.267])
case2_tra = np.asarray([0.457, 0.385, 0.343, 0.343, 0.353])

x = [0.1, 0.2, 0.3, 0.4, 0.5]

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

n = 20

axs[0].plot(x, case1_qcool, label= 'a')
axs[0].plot(x, case1_qheat, label= 'b')
axs[0].plot(x, case1_tra, label= 'c')
axs[0].axhline(y=1, color='red', linestyle='--')
axs[0].set_xlabel('Training rate [-]', fontsize=n)
axs[0].set_ylabel('Coefficient', fontsize=n)
axs[0].set_title("Case 1", fontsize=n)
axs[0].set_ylim([0, 1.3])
axs[0].tick_params(axis='both', which='major', labelsize=n)

axs[1].plot(x, case2_qcool, label= 'a')
axs[1].plot(x, case2_qheat, label= 'b')
axs[1].plot(x, case2_tra, label= 'c')
axs[1].axhline(y=1, color='red', linestyle='--')
axs[1].set_xlabel('Training rate [-]', fontsize=n)
axs[1].set_title("Case 2", fontsize=n)
axs[1].set_yticks([])
axs[1].set_ylim([0, 1.3])
axs[1].tick_params(axis='both', which='major', labelsize=n)

plt.tight_layout()
plt.show()
plt.close()