################# Code for drawing the scatterplots ######################################


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

# Values obtained by running scatterplot_print.py
length = [363, 99, 32, 11, 4, 54, 24, 6, 14, 5, 5, 4, 8, 19, 44]
corr__t_ra__q_cool = [0.295, 0.28,  0.404,  0.374, 0.422, 0.333, 0.3151, 0.3032, 0.333, 0.3034, 0.303, 0.422, 0.569, 0.615, 0.381]
corr__t_ra__q_heat = [-0.083, -0.505, -0.333, -0.2043, -0.482, -0.441, -0.354, -0.5906, -0.221, -0.5558, -0.555, -0.482, -0.608, -0.613, -0.428] #
corr__q_heat__q_cool = [-0.35, -0.462, -0.558, -0.546, -0.6707, -0.5186, -0.488,  -0.583, -0.525, -0.5861, -0.586, -0.670, -0.661, -0.609, -0.533] #
corr__t_oa__q_cool = [0.73, 0.71,  0.747,  0.7802,  0.7823,  0.6827,  0.6653, 0.7027, 0.737, 0.712,0.712,0.782, 0.823, 0.825, 0.822]
corr__t_oa__q_heat = [-0.48, -0.598, -0.663, -0.7453, -0.8322, -0.6703, -0.701, -0.8374, -0.701, -0.8398,-0.839,-0.832, -0.797, -0.731, -0.554] #
corr__t_oa__t_ra = [0.52, 0.55, 0.6122, 0.5057, 0.6440, 0.6183, 0.591,  0.6828, 0.536, 0.675, 0.675, 0.644, 0.758, 0.782, 0.53]
threshold___q_cool = [0, 10, 20, 30, 40, 10, 10, 10, 20, 20, 30, 50, 50, 50, 50]
threshold___q_heat = [0, 10, 20, 30, 40, 20, 30, 40, 30, 40, 40, 40, 30, 20, 10]

# Create a figure with 7 subplots
fig = plt.figure(figsize=(12, 6))

# Create a 3D axes object for each subplot
ax1 = fig.add_subplot(2, 4, 1, projection='3d')
ax2 = fig.add_subplot(2, 4, 2, projection='3d')
ax3 = fig.add_subplot(2, 4, 3, projection='3d')
ax4 = fig.add_subplot(2, 4, 4, projection='3d')
ax5 = fig.add_subplot(2, 4, 5, projection='3d')
ax6 = fig.add_subplot(2, 4, 6, projection='3d')
ax7 = fig.add_subplot(2, 4, 7, projection='3d')

# Create scatterplots for each subplot
scatter1 = ax1.scatter(threshold___q_cool, threshold___q_heat, length, c='b', alpha=0.2)
scatter2 = ax2.scatter(threshold___q_cool, threshold___q_heat, corr__t_ra__q_cool, c='b', alpha=0.2)
scatter3 = ax3.scatter(threshold___q_cool, threshold___q_heat, corr__t_ra__q_heat, c='b', alpha=0.2)
scatter4 = ax4.scatter(threshold___q_cool, threshold___q_heat, corr__q_heat__q_cool, c='b', alpha=0.2)
scatter5 = ax5.scatter(threshold___q_cool, threshold___q_heat, corr__t_oa__q_cool, c='b', alpha=0.2)
scatter6 = ax6.scatter(threshold___q_cool, threshold___q_heat, corr__t_oa__q_heat, c='b', alpha=0.2)
scatter7 = ax7.scatter(threshold___q_cool, threshold___q_heat, corr__t_oa__t_ra, c='b', alpha=0.2)

# Set the axis labels
ax1.set_xlabel('Q_cool_tot [kW]', fontsize=15)
ax1.set_ylabel('Q_hw [kW]', fontsize=15)
ax1.set_title('length [days]', fontsize=15)
ax1.tick_params(axis='x', which='major', labelsize=15)
ax1.tick_params(axis='y', which='major', labelsize=15)
ax1.tick_params(axis='z', which='major', labelsize=15)
ax2.set_xlabel('Q_cool_tot [kW]', fontsize=15)
ax2.set_ylabel('Q_hw [kW]', fontsize=15)
ax2.set_title('T_ra_avg - Q_cool_tot', fontsize=15)
ax2.tick_params(axis='x', which='major', labelsize=15)
ax2.tick_params(axis='y', which='major', labelsize=15)
ax2.tick_params(axis='z', which='major', labelsize=15)
ax3.set_xlabel('Q_cool_tot [kW]', fontsize=15)
ax3.set_ylabel('Q_hw [kW]', fontsize=15)
ax3.set_title('T_ra_avg - Q_hw', fontsize=15)
ax3.tick_params(axis='x', which='major', labelsize=15)
ax3.tick_params(axis='y', which='major', labelsize=15)
ax3.tick_params(axis='z', which='major', labelsize=15)
ax4.set_xlabel('Q_cool_tot [kW]', fontsize=15)
ax4.set_ylabel('Q_hw [kW]', fontsize=15)
ax4.set_title('Q_hw - Q_cool_tot', fontsize=15)
ax4.tick_params(axis='x', which='major', labelsize=15)
ax4.tick_params(axis='y', which='major', labelsize=15)
ax4.tick_params(axis='z', which='major', labelsize=15)
ax5.set_xlabel('Q_cool_tot [kW]', fontsize=15)
ax5.set_ylabel('Q_hw [kW]', fontsize=15)
ax5.set_title('T_oa_avg - Q_cool_tot', fontsize=15)
ax5.tick_params(axis='x', which='major', labelsize=15)
ax5.tick_params(axis='y', which='major', labelsize=15)
ax5.tick_params(axis='z', which='major', labelsize=15)
ax6.set_xlabel('Q_cool_tot [kW]', fontsize=15)
ax6.set_ylabel('Q_hw [kW]', fontsize=15)
ax6.set_title('T_oa_avg - Q_hw', fontsize=15)
ax6.tick_params(axis='x', which='major', labelsize=15)
ax6.tick_params(axis='y', which='major', labelsize=15)
ax6.tick_params(axis='z', which='major', labelsize=15)
ax7.set_xlabel('Q_cool_tot [kW]', fontsize=15)
ax7.set_ylabel('Q_hw [kW]', fontsize=15)
ax7.set_title('T_oa_avg - T_ra_avg', fontsize=15)
ax7.tick_params(axis='x', which='major', labelsize=15)
ax7.tick_params(axis='y', which='major', labelsize=15)
ax7.tick_params(axis='z', which='major', labelsize=15)

max1 = np.max(length)
max2 = np.max(corr__t_ra__q_cool)
max3 = np.min(corr__t_ra__q_heat)
max4 = np.min(corr__q_heat__q_cool)
max5 = np.max(corr__t_oa__q_cool)
max6 = np.min(corr__t_oa__q_heat)
max7 = np.max(corr__t_oa__t_ra)

ax1.scatter(threshold___q_cool[np.argmax(length)], threshold___q_heat[np.argmax(length)], max1, c='k', marker='o')
ax2.scatter(threshold___q_cool[np.argmax(corr__t_ra__q_cool)], threshold___q_heat[np.argmax(corr__t_ra__q_cool)], max2, c='k', marker='o')
ax3.scatter(threshold___q_cool[np.argmin(corr__t_ra__q_heat)], threshold___q_heat[np.argmin(corr__t_ra__q_heat)], max3, c='k', marker='o')
ax4.scatter(threshold___q_cool[np.argmin(corr__q_heat__q_cool)], threshold___q_heat[np.argmin(corr__q_heat__q_cool)], max4, c='k', marker='o')
ax5.scatter(threshold___q_cool[np.argmax(corr__t_oa__q_cool)], threshold___q_heat[np.argmax(corr__t_oa__q_cool)], max5, c='k', marker='o')
ax6.scatter(threshold___q_cool[np.argmin(corr__t_oa__q_heat)], threshold___q_heat[np.argmin(corr__t_oa__q_heat)], max6, c='k', marker='o')
ax7.scatter(threshold___q_cool[np.argmax(corr__t_oa__t_ra)], threshold___q_heat[np.argmax(corr__t_oa__t_ra)], max7, c='k', marker='o')

ax1.scatter(50, 20, 19, c='r', marker='o')
ax2.scatter(50, 20, 0.615, c='r', marker='o')
ax3.scatter(50, 20, -0.613, c='r', marker='o')
ax4.scatter(50, 20, -0.609, c='r', marker='o')
ax5.scatter(50, 20, 0.825 , c='r', marker='o')
ax6.scatter(50, 20, -0.731, c='r', marker='o')
ax7.scatter(50, 20, 0.782, c='r', marker='o')

plt.tight_layout()
fig.canvas.draw()
plt.subplots_adjust(top=0.85)  # Modify the top parameter to reduce the space
plt.subplots_adjust(hspace=0.5)  # Increase the hspace value to add extra space between subplots
plt.show()