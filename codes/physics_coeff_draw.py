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
case1_a = np.asarray([0.196, 0.145, 0.095, 0.112, 0.106])
case1_b = np.asarray([0.050, 0.011, 0.006, 0.023, 0.021])
case1_c = np.asarray([0.146, 0.116, 0.085, 0.080, 0.079])
case2_a = np.asarray([1.226, 1.154, 1.108, 1.084, 1.089])
case2_b = np.asarray([0.179, 0.194, 0.259, 0.264, 0.267])
case2_c = np.asarray([0.457, 0.385, 0.343, 0.343, 0.353])

case1_t_oa__t_ra = np.asarray([0.576184624, 0.558864488, 0.532431598, 0.531902216, 0.526113634])
case1_t_ra__q_cool = np.asarray([0.34082728, 0.332189744, 0.315651288, 0.309693965, 0.295072595])
case1_t_ra__q_heat = np.asarray([-0.042193081, -0.067756927, -0.065482297, -0.087155536, -0.080030147])
case2_t_oa__t_ra = np.asarray([0.911084079, 0.823181427, 0.831588349, 0.813253187, 0.807330043])
case2_t_ra__q_cool = np.asarray([0.622320096, 0.608627531, 0.63239163, 0.619269431, 0.624753487])
case2_t_ra__q_heat = np.asarray([-0.794649034, -0.673457846, -0.644092642, -0.634583826, -0.633367182])

x = [0.1, 0.2, 0.3, 0.4, 0.5]

fig, axs = plt.subplots(2, 2)

n = 20

axs[0,0].plot(x, case1_a, color='blue', label= 'a')
axs[0,0].plot(x, case1_b, color='orange', label= 'b')
axs[0,0].plot(x, case1_c, color='green', label= 'c')
axs[0,0].axhline(y=1, color='red', linestyle='--')
axs[0,0].set_xlabel('Training rate [-]', fontsize=n)
axs[0,0].set_ylabel('Coefficient', fontsize=n)
axs[0,0].set_title("Case 1", fontsize=n)
axs[0,0].set_xticks(x)
axs[0,0].set_ylim([0, 1.3])
axs[0,0].tick_params(axis='both', which='major', labelsize=n)

axs[0,1].plot(x, case2_a, color='blue', label= 'a')
axs[0,1].plot(x, case2_b, color='orange', label= 'b')
axs[0,1].plot(x, case2_c, color='green', label= 'c')
axs[0,1].plot(x, np.full(len(x), np.nan), color='purple', label= 'PCC_6')
axs[0,1].plot(x, np.full(len(x), np.nan), color='red', label= 'PCC_1')
axs[0,1].plot(x, np.full(len(x), np.nan), color='magenta', label= 'PCC_2')
axs[0,1].axhline(y=1, color='red', linestyle='--')
axs[0,1].set_xlabel('Training rate [-]', fontsize=n)
axs[0,1].set_ylabel('Coefficient', fontsize=n)
axs[0,1].set_title("Case 2", fontsize=n)
axs[0,1].set_xticks(x)
axs[0,1].set_ylim([0, 1.3])
axs[0,1].tick_params(axis='both', which='major', labelsize=n)
axs[0,1].legend(loc='center left', fontsize=n, bbox_to_anchor=(1, 0.1))

axs[1,0].plot(x, np.abs(case1_t_oa__t_ra), color='purple', label= 'PCC_6')
axs[1,0].plot(x, np.abs(case1_t_ra__q_cool), color='red', label= 'PCC_1')
axs[1,0].plot(x, np.abs(case1_t_ra__q_heat), color='magenta', label= 'PCC_2')
axs[1,0].set_xlabel('Training rate [-]', fontsize=n)
axs[1,0].set_ylabel('Correlation', fontsize=n)
axs[1,0].set_title("Case 1", fontsize=n)
axs[1,0].set_xticks(x)
axs[1,0].set_ylim([0, 1])
axs[1,0].tick_params(axis='both', which='major', labelsize=n)

axs[1,1].plot(x, np.abs(case2_t_oa__t_ra), color='purple', label= 'PCC_6')
axs[1,1].plot(x, np.abs(case2_t_ra__q_cool), color='red', label= 'PCC_1')
axs[1,1].plot(x, np.abs(case2_t_ra__q_heat), color='magenta', label= 'PCC_2')
axs[1,1].set_xlabel('Training rate [-]', fontsize=n)
axs[1,1].set_ylabel('Correlation', fontsize=n)
axs[1,1].set_title("Case 2", fontsize=n)
axs[1,1].set_xticks(x)
axs[1,1].set_ylim([0, 1])
axs[1,1].tick_params(axis='both', which='major', labelsize=n)

plt.subplots_adjust(top=0.85)  # Modify the top parameter to reduce the space
plt.subplots_adjust(hspace=0.5)  # Increase the hspace value to add extra space between subplotsplt.show()
plt.show()