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
# Average and standard deviations computed using the excel file# Please refer to results.csv (results sheet)
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
RMSE_case1_lin_qcool = np.asarray([21.563, 21.437, 21.318, 20.947, 20.816])
RMSE_case1_lin_qheat = np.asarray([9.753, 9.741, 9.768, 9.743, 9.755])
RMSE_case1_lin_tra = np.asarray([0.329, 0.327, 0.327, 0.327, 0.334])
RMSE_case1_knn_qcool = np.asarray([26.513, 25.935, 25.412, 25.042, 24.787])
RMSE_case1_knn_qheat = np.asarray([18.355, 18.401, 18.586, 18.552, 18.621])
RMSE_case1_knn_tra = np.asarray([0.878, 0.868, 0.874, 0.884, 0.891])

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
RMSE_case2_lin_qcool = np.asarray([58.770, 58.206, 58.708, 59.009, 58.142])
RMSE_case2_lin_qheat = np.asarray([11.410, 11.366, 11.344, 11.299, 11.234])
RMSE_case2_lin_tra = np.asarray([0.237, 0.238, 0.238, 0.238, 0.240])
RMSE_case2_knn_qcool = np.asarray([34.395, 31.691, 30.064, 29.217, 28.961])
RMSE_case2_knn_qheat = np.asarray([13.777, 13.234, 13.013, 12.662, 12.591])
RMSE_case2_knn_tra = np.asarray([0.230, 0.230, 0.229, 0.227, 0.232])

x = [0.1, 0.2, 0.3, 0.4, 0.5]

RMSE_fig, RMSE_axs = plt.subplots(3, 2)
RMSE_fig.set_figheight(15)
RMSE_fig.set_figwidth(4)

n = 25

RMSE_axs[0,0].plot(x, RMSE_case1_1_qcool, color='purple', label= 'Univariate__DAE_3')
RMSE_axs[0,0].plot(x, RMSE_case1_2_qcool, color='orange', label= 'Multivariate_DAE_1')
RMSE_axs[0,0].plot(x, RMSE_case1_3_qcool, color='green', label= ' Multivariate_DAE_2')
RMSE_axs[0,0].plot(x, RMSE_case1_4_qcool, color='red', label= 'PI-DAE')
RMSE_axs[0,0].plot(x, RMSE_case1_lin_qcool, color='magenta', label= 'LIN', ls="--")
RMSE_axs[0,0].plot(x, RMSE_case1_knn_qcool, color='brown', label= 'KNN', ls="--")
RMSE_axs[0,0].set_ylabel('RMSE [kW] \n (Q_cool_tot)', fontsize=n)
RMSE_axs[0,0].set_title("Case 1", fontsize=n)
RMSE_axs[0,0].tick_params(axis='both', which='major', labelsize=n)
RMSE_axs[0, 0].set_xticks([])

RMSE_axs[1,0].plot(x, RMSE_case1_1_qheat, color='black', label= 'Univariate_DAE_2')
RMSE_axs[1,0].plot(x, RMSE_case1_2_qheat, color='orange', label= 'Multivariate_DAE_1')
RMSE_axs[1,0].plot(x, RMSE_case1_3_qheat, color='green', label= ' Multivariate_DAE_2')
RMSE_axs[1,0].plot(x, RMSE_case1_4_qheat, color='red', label= 'PI-DAE')
RMSE_axs[1,0].plot(x, RMSE_case1_lin_qheat, color='magenta', label= 'LIN', ls="--")
RMSE_axs[1,0].plot(x, RMSE_case1_knn_qheat, color='brown', label= 'KNN', ls="--")
RMSE_axs[1,0].set_ylabel('RMSE [kW] \n (Q_hw)', fontsize=n)
RMSE_axs[1,0].tick_params(axis='both', which='major', labelsize=n)
RMSE_axs[1, 0].set_xticks([])

RMSE_axs[2,0].plot(x, RMSE_case1_1_tra, color='blue', label= 'Univariate_DAE_1')
RMSE_axs[2,0].plot(x, RMSE_case1_2_tra, color='orange', label= 'Multivariate_DAE_1')
RMSE_axs[2,0].plot(x, RMSE_case1_3_tra, color='green', label= ' Multivariate_DAE_2')
RMSE_axs[2,0].plot(x, RMSE_case1_4_tra, color='red', label= 'PI-DAE')
RMSE_axs[2,0].plot(x, RMSE_case1_lin_tra, color='magenta', label= 'LIN', ls="--")
RMSE_axs[2,0].plot(x, RMSE_case1_knn_tra, color='brown', label= 'KNN', ls="--")
RMSE_axs[2,0].set_xlabel('Training rate [-]', fontsize=n)
RMSE_axs[2,0].set_ylabel('RMSE [deg C] \n (T_ra_avg)', fontsize=n)
RMSE_axs[2,0].tick_params(axis='both', which='major', labelsize=n)

RMSE_axs[0,1].plot(x, RMSE_case2_1_qcool, color='purple', label= 'Univariate__DAE_3')
RMSE_axs[0,1].plot(x, RMSE_case2_2_qcool, color='orange', label= 'Multivariate_DAE_1')
RMSE_axs[0,1].plot(x, RMSE_case2_3_qcool, color='green', label= ' Multivariate_DAE_2')
RMSE_axs[0,1].plot(x, RMSE_case2_4_qcool, color='red', label= 'PI-DAE')
RMSE_axs[0,1].plot(x, RMSE_case2_lin_qcool, color='magenta', label= 'LIN', ls="--")
RMSE_axs[0,1].plot(x, RMSE_case2_knn_qcool, color='brown', label= 'KNN', ls="--")
RMSE_axs[0,1].set_title("Case 2", fontsize=n)
RMSE_axs[0,1].tick_params(axis='both', which='major', labelsize=n)
RMSE_axs[0, 1].set_xticks([])

RMSE_axs[1,1].plot(x, RMSE_case2_1_qheat, color='black', label= 'Univariate_DAE_2')
RMSE_axs[1,1].plot(x, RMSE_case2_2_qheat, color='orange', label= 'Multivariate_DAE_1')
RMSE_axs[1,1].plot(x, RMSE_case2_3_qheat, color='green', label= ' Multivariate_DAE_2')
RMSE_axs[1,1].plot(x, RMSE_case2_4_qheat, color='red', label= 'PI-DAE')
RMSE_axs[1,1].plot(x, RMSE_case2_lin_qheat, color='magenta', label= 'LIN', ls="--")
RMSE_axs[1,1].plot(x, RMSE_case2_knn_qheat, color='brown', label= 'KNN', ls="--")
RMSE_axs[1,1].tick_params(axis='both', which='major', labelsize=n)
RMSE_axs[1, 1].set_xticks([])

RMSE_axs[2,1].plot(x, RMSE_case2_1_tra, color='blue', label= 'Univariate_DAE_1')
RMSE_axs[2,1].plot(x, RMSE_case2_2_tra, color='orange', label= 'Multivariate_DAE_1')
RMSE_axs[2,1].plot(x, RMSE_case2_3_tra, color='green', label= ' Multivariate_DAE_2')
RMSE_axs[2,1].plot(x, RMSE_case2_4_tra, color='red', label= 'PI-DAE')
RMSE_axs[2,1].plot(x, RMSE_case2_lin_tra, color='magenta', label= 'LIN', ls="--")
RMSE_axs[2,1].plot(x, RMSE_case2_knn_tra, color='brown', label= 'KNN', ls="--")
RMSE_axs[2,1].set_xlabel('Training rate [-]', fontsize=n)
RMSE_axs[2,1].tick_params(axis='both', which='major', labelsize=n)

plt.subplots_adjust(wspace=0.4)
plt.show()