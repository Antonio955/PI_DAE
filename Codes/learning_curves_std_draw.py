################# Code for drawing the standard deviation of the results ######################################


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

# Please refer to results.csv (results sheet)
# RMSEs obtained by running evaluation.py.
# Average and standard deviations computed using the excel file
RMSE_case1_1_qcool = np.asarray([26.611, 24.419, 24.103, 24.106, 23.975])
RMSE_case1_1_qheat = np.asarray([8.601,8.364 , 8.307, 8.154, 8.151])
RMSE_case1_1_tra = np.asarray([0.560,  0.526, 0.447, 0.458, 0.468])
RMSE_case1_2_qcool = np.asarray([28.294, 27.691, 29.573, 28.076, 28.892])
RMSE_case1_2_qheat = np.asarray([11.445, 11.298, 11.773, 11.004, 11.191])
RMSE_case1_2_tra = np.asarray([0.646, 0.665, 0.749 , 0.741, 0.749])
RMSE_case1_3_qcool = np.asarray([19.099, 18.226, 17.096, 16.647, 16.553])
RMSE_case1_3_qheat = np.asarray([9.777, 9.449, 9.136, 8.735, 8.811])
RMSE_case1_3_tra = np.asarray([0.556, 0.543, 0.554, 0.583, 0.627])
RMSE_case1_4_qcool = np.asarray([19.093, 18.091, 17.122, 16.804, 16.436])
RMSE_case1_4_qheat = np.asarray([9.849, 9.556, 9.112, 8.873, 8.767])
RMSE_case1_4_tra = np.asarray([0.551, 0.525, 0.551, 0.571, 0.595])
RMSE_case2_1_qcool = np.asarray([54.560, 46.361, 43.381, 40.165, 40.600])
RMSE_case2_1_qheat = np.asarray([14.750, 11.199, 10.244, 9.788, 9.586])
RMSE_case2_1_tra = np.asarray([0.253, 0.200, 0.206, 0.198, 0.177])
RMSE_case2_2_qcool = np.asarray([58.469, 48.156, 45.279, 42.845, 41.007])
RMSE_case2_2_qheat = np.asarray([15.372, 13.166, 12.260, 11.565, 11.100])
RMSE_case2_2_tra = np.asarray([0.295, 0.249, 0.236, 0.225, 0.221])
RMSE_case2_3_qcool = np.asarray([39.166, 29.185, 26.717, 24.328, 24.339])
RMSE_case2_3_qheat = np.asarray([12.884, 10.925, 10.423, 9.903, 9.508])
RMSE_case2_3_tra = np.asarray([0.252, 0.230, 0.215, 0.220, 0.218])
RMSE_case2_4_qcool = np.asarray([40.415, 29.256, 26.044, 23.988, 24.042])
RMSE_case2_4_qheat = np.asarray([13.191, 10.854, 10.335, 9.892, 9.362])
RMSE_case2_4_tra = np.asarray([0.254, 0.220, 0.203, 0.207, 0.210])

RMSE_case1_1_qcool_std = np.asarray([8.118, 6.828, 6.875, 7.687, 6.927])
RMSE_case1_1_qheat_std = np.asarray([0.841, 0.938, 0.899, 0.894, 0.865])
RMSE_case1_1_tra_std = np.asarray([0.152, 0.089, 0.049, 0.052, 0.020])
RMSE_case1_2_qcool_std = np.asarray([3.033, 3.597, 5.017, 4.655, 5.370])
RMSE_case1_2_qheat_std = np.asarray([0.844, 1.014, 1.518, 0.986,1.265])
RMSE_case1_2_tra_std = np.asarray([0.101, 0.132, 0.169, 0.137, 0.163])
RMSE_case1_3_qcool_std = np.asarray([1.359, 1.208, 1.661, 1.450, 0.693])
RMSE_case1_3_qheat_std = np.asarray([0.514, 0.494, 0.352, 0.227, 0.390])
RMSE_case1_3_tra_std = np.asarray([0.072, 0.065, 0.050, 0.044, 0.033])
RMSE_case1_4_qcool_std = np.asarray([1.347, 1.343, 1.357, 1.387, 1.027])
RMSE_case1_4_qheat_std = np.asarray([0.439, 0.545, 0.349, 0.348, 0.318])
RMSE_case1_4_tra_std = np.asarray([0.058, 0.054, 0.050, 0.056, 0.044])
RMSE_case2_1_qcool_std = np.asarray([9.316, 8.681, 7.788, 7.527, 8.423])
RMSE_case2_1_qheat_std = np.asarray([1.193, 1.028, 0.871, 0.930, 1.109])
RMSE_case2_1_tra_std = np.asarray([0.037, 0.017, 0.043, 0.041, 0.021])
RMSE_case2_2_qcool_std = np.asarray([5.849, 3.899, 4.270, 4.417, 3.984])
RMSE_case2_2_qheat_std = np.asarray([0.768, 0.685, 0.686, 0.658, 0.478])
RMSE_case2_2_tra_std = np.asarray([0.023, 0.010, 0.017, 0.017, 0.014])
RMSE_case2_3_qcool_std = np.asarray([4.900, 4.226, 3.840, 2.716, 2.497])
RMSE_case2_3_qheat_std = np.asarray([0.772, 0.316, 0.187, 0.511, 0.458])
RMSE_case2_3_tra_std = np.asarray([0.005, 0.014, 0.016, 0.021, 0.021])
RMSE_case2_4_qcool_std = np.asarray([2.535, 3.730, 3.496, 2.634, 3.023])
RMSE_case2_4_qheat_std = np.asarray([0.644, 0.265, 0.261, 0.468, 0.344])
RMSE_case2_4_tra_std = np.asarray([0.008, 0.011, 0.014, 0.014, 0.022])


x = [0.1, 0.2, 0.3, 0.4, 0.5]

RMSE_fig, RMSE_axs = plt.subplots(2, 6, figsize=(26, 8))

n = 30

RMSE_axs[0,0].plot(x, RMSE_case1_1_qcool, color='purple', label= 'Univariate__DAE_3')
RMSE_axs[0,0].plot(x, RMSE_case1_4_qcool, color='red', label= 'PI-DAE')
RMSE_axs[0,0].fill_between(x, RMSE_case1_1_qcool - RMSE_case1_1_qcool_std, RMSE_case1_1_qcool + RMSE_case1_1_qcool_std, alpha=0.2, color='purple')
RMSE_axs[0,0].fill_between(x, RMSE_case1_4_qcool - RMSE_case1_4_qcool_std, RMSE_case1_4_qcool + RMSE_case1_4_qcool_std, alpha=0.2, color='red')
RMSE_axs[0,0].set_ylabel('RMSE [kW]', fontsize=n)
RMSE_axs[0,0].set_title("Case 1 - Q_cool_tot", fontsize=n)
RMSE_axs[0,0].tick_params(axis='both', which='major', labelsize=n)

RMSE_axs[1,0].plot(x, RMSE_case1_3_qcool, color='green', label= ' Multivariate_DAE_2')
RMSE_axs[1,0].plot(x, RMSE_case1_4_qcool, color='red', label= 'PI-DAE')
RMSE_axs[1,0].fill_between(x, RMSE_case1_3_qcool - RMSE_case1_3_qcool_std, RMSE_case1_3_qcool + RMSE_case1_3_qcool_std, alpha=0.2, color='green')
RMSE_axs[1,0].fill_between(x, RMSE_case1_4_qcool - RMSE_case1_4_qcool_std, RMSE_case1_4_qcool + RMSE_case1_4_qcool_std, alpha=0.2, color='red')
RMSE_axs[1,0].set_xlabel('Training rate [-]', fontsize=n)
RMSE_axs[1,0].set_ylabel('RMSE [kW]', fontsize=n)
RMSE_axs[1,0].set_title("Case 1 - Q_cool_tot", fontsize=n)
RMSE_axs[1,0].tick_params(axis='both', which='major', labelsize=n)

RMSE_axs[0,1].plot(x, RMSE_case1_1_qheat, color='black', label= 'Univariate_DAE_2')
RMSE_axs[0,1].plot(x, RMSE_case1_4_qheat, color='red', label= 'PI-DAE')
RMSE_axs[0,1].fill_between(x, RMSE_case1_1_qheat - RMSE_case1_1_qheat_std, RMSE_case1_1_qheat + RMSE_case1_1_qheat_std, alpha=0.2, color='black')
RMSE_axs[0,1].fill_between(x, RMSE_case1_4_qheat - RMSE_case1_4_qheat_std, RMSE_case1_4_qheat + RMSE_case1_4_qheat_std, alpha=0.2, color='red')
RMSE_axs[0,1].set_ylabel('RMSE [kW]', fontsize=n)
RMSE_axs[0,1].set_title("Case 1 - Q_hw", fontsize=n)
RMSE_axs[0,1].tick_params(axis='both', which='major', labelsize=n)

RMSE_axs[1,1].plot(x, RMSE_case1_3_qheat, color='green', label= ' Multivariate_DAE_2')
RMSE_axs[1,1].plot(x, RMSE_case1_4_qheat, color='red', label= 'PI-DAE')
RMSE_axs[1,1].fill_between(x, RMSE_case1_3_qheat - RMSE_case1_3_qheat_std, RMSE_case1_3_qheat + RMSE_case1_3_qheat_std, alpha=0.2, color='green')
RMSE_axs[1,1].fill_between(x, RMSE_case1_4_qheat - RMSE_case1_4_qheat_std, RMSE_case1_4_qheat + RMSE_case1_4_qheat_std, alpha=0.2, color='red')
RMSE_axs[1,1].set_xlabel('Training rate [-]', fontsize=n)
RMSE_axs[1,1].set_ylabel('RMSE [kW]', fontsize=n)
RMSE_axs[1,1].set_title("Case 1 - Q_hw", fontsize=n)
RMSE_axs[1,1].tick_params(axis='both', which='major', labelsize=n)

RMSE_axs[0,2].plot(x, RMSE_case1_1_tra, color='blue', label= 'Univariate_DAE_1')
RMSE_axs[0,2].plot(x, RMSE_case1_4_tra, color='red', label= 'PI-DAE')
RMSE_axs[0,2].fill_between(x, RMSE_case1_1_tra - RMSE_case1_1_tra_std, RMSE_case1_1_tra + RMSE_case1_1_tra_std, alpha=0.2, color='blue')
RMSE_axs[0,2].fill_between(x, RMSE_case1_4_tra - RMSE_case1_4_tra_std, RMSE_case1_4_tra + RMSE_case1_4_tra_std, alpha=0.2, color='red')
RMSE_axs[0,2].set_ylabel('RMSE [deg C]', fontsize=n)
RMSE_axs[0,2].set_title("Case 1 - T_ra_avg", fontsize=n)
RMSE_axs[0,2].tick_params(axis='both', which='major', labelsize=n)

RMSE_axs[1,2].plot(x, RMSE_case1_3_tra, color='green', label= ' Multivariate_DAE_2')
RMSE_axs[1,2].plot(x, RMSE_case1_4_tra, color='red', label= 'PI-DAE')
RMSE_axs[1,2].fill_between(x, RMSE_case1_3_tra - RMSE_case1_3_tra_std, RMSE_case1_3_tra + RMSE_case1_3_tra_std, alpha=0.2, color='green')
RMSE_axs[1,2].fill_between(x, RMSE_case1_4_tra - RMSE_case1_4_tra_std, RMSE_case1_4_tra + RMSE_case1_4_tra_std, alpha=0.2, color='red')
RMSE_axs[1,2].set_xlabel('Training rate [-]', fontsize=n)
RMSE_axs[1,2].set_ylabel('RMSE [deg C]', fontsize=n)
RMSE_axs[1,2].set_title("Case 1 - T_ra_avg", fontsize=n)
RMSE_axs[1,2].tick_params(axis='both', which='major', labelsize=n)


RMSE_axs[0,3].plot(x, RMSE_case2_1_qcool, color='purple', label= 'Univariate__DAE_3')
RMSE_axs[0,3].plot(x, RMSE_case2_4_qcool, color='red', label= 'PI-DAE')
RMSE_axs[0,3].fill_between(x, RMSE_case2_1_qcool - RMSE_case2_1_qcool_std, RMSE_case2_1_qcool + RMSE_case2_1_qcool_std, alpha=0.2, color='purple')
RMSE_axs[0,3].fill_between(x, RMSE_case2_4_qcool - RMSE_case2_4_qcool_std, RMSE_case2_4_qcool + RMSE_case2_4_qcool_std, alpha=0.2, color='red')
RMSE_axs[0,3].set_ylabel('RMSE [kW]', fontsize=n)
RMSE_axs[0,3].set_title("Case 2 - Q_cool_tot", fontsize=n)
RMSE_axs[0,3].tick_params(axis='both', which='major', labelsize=n)

RMSE_axs[1,3].plot(x, RMSE_case2_3_qcool, color='green', label= ' Multivariate_DAE_2')
RMSE_axs[1,3].plot(x, RMSE_case2_4_qcool, color='red', label= 'PI-DAE')
RMSE_axs[1,3].fill_between(x, RMSE_case2_3_qcool - RMSE_case2_3_qcool_std, RMSE_case2_3_qcool + RMSE_case2_3_qcool_std, alpha=0.2, color='green')
RMSE_axs[1,3].fill_between(x, RMSE_case2_4_qcool - RMSE_case2_4_qcool_std, RMSE_case2_4_qcool + RMSE_case2_4_qcool_std, alpha=0.2, color='red')
RMSE_axs[1,3].set_xlabel('Training rate [-]', fontsize=n)
RMSE_axs[1,3].set_ylabel('RMSE [kW]', fontsize=n)
RMSE_axs[1,3].set_title("Case 2 - Q_cool_tot", fontsize=n)
RMSE_axs[1,3].tick_params(axis='both', which='major', labelsize=n)

RMSE_axs[0,4].plot(x, RMSE_case2_1_qheat, color='black', label= 'Univariate_DAE_2')
RMSE_axs[0,4].plot(x, RMSE_case2_4_qheat, color='red', label= 'PI-DAE')
RMSE_axs[0,4].fill_between(x, RMSE_case2_1_qheat - RMSE_case2_1_qheat_std, RMSE_case2_1_qheat + RMSE_case2_1_qheat_std, alpha=0.2, color='black')
RMSE_axs[0,4].fill_between(x, RMSE_case2_4_qheat - RMSE_case2_4_qheat_std, RMSE_case2_4_qheat + RMSE_case2_4_qheat_std, alpha=0.2, color='red')
RMSE_axs[0,4].set_ylabel('RMSE [kW]', fontsize=n)
RMSE_axs[0,4].set_title("Case 2 - Q_hw", fontsize=n)
RMSE_axs[0,4].tick_params(axis='both', which='major', labelsize=n)

RMSE_axs[1,4].plot(x, RMSE_case2_3_qheat, color='green', label= ' Multivariate_DAE_2')
RMSE_axs[1,4].plot(x, RMSE_case2_4_qheat, color='red', label= 'PI-DAE')
RMSE_axs[1,4].fill_between(x, RMSE_case2_3_qheat - RMSE_case2_3_qheat_std, RMSE_case2_3_qheat + RMSE_case2_3_qheat_std, alpha=0.2, color='green')
RMSE_axs[1,4].fill_between(x, RMSE_case2_4_qheat - RMSE_case2_4_qheat_std, RMSE_case2_4_qheat + RMSE_case2_4_qheat_std, alpha=0.2, color='red')
RMSE_axs[1,4].set_xlabel('Training rate [-]', fontsize=n)
RMSE_axs[1,4].set_ylabel('RMSE [kW]', fontsize=n)
RMSE_axs[1,4].set_title("Case 2 - Q_hw", fontsize=n)
RMSE_axs[1,4].tick_params(axis='both', which='major', labelsize=n)


RMSE_axs[0,5].plot(x, RMSE_case2_1_tra, color='blue', label= 'Univariate_DAE_1')
RMSE_axs[0,5].plot(x, RMSE_case2_4_tra, color='red', label= 'PI-DAE')
RMSE_axs[0,5].fill_between(x, RMSE_case2_1_tra - RMSE_case2_1_tra_std, RMSE_case2_1_tra + RMSE_case2_1_tra_std, alpha=0.2, color='blue')
RMSE_axs[0,5].fill_between(x, RMSE_case2_4_tra - RMSE_case2_4_tra_std, RMSE_case2_4_tra + RMSE_case2_4_tra_std, alpha=0.2, color='red')
RMSE_axs[0,5].set_ylabel('RMSE [deg C]', fontsize=n)
RMSE_axs[0,5].set_title("Case 2 - T_ra_avg", fontsize=n)
RMSE_axs[0,5].tick_params(axis='both', which='major', labelsize=n)


RMSE_axs[1,5].plot(x, RMSE_case2_3_tra, color='green', label= ' Multivariate_DAE_2')
RMSE_axs[1,5].plot(x, RMSE_case2_4_tra, color='red', label= 'PI-DAE')
RMSE_axs[1,5].fill_between(x, RMSE_case2_3_tra - RMSE_case2_3_tra_std, RMSE_case2_3_tra + RMSE_case2_3_tra_std, alpha=0.2, color='green')
RMSE_axs[1,5].fill_between(x, RMSE_case2_4_tra - RMSE_case2_4_tra_std, RMSE_case2_4_tra + RMSE_case2_4_tra_std, alpha=0.2, color='red')
RMSE_axs[1,5].set_xlabel('Training rate [-]', fontsize=n)
RMSE_axs[1,5].set_ylabel('RMSE [deg C]', fontsize=n)
RMSE_axs[1,5].set_title("Case 2 - T_ra_avg", fontsize=n)
RMSE_axs[1,5].tick_params(axis='both', which='major', labelsize=n)

plt.tight_layout()
plt.show()