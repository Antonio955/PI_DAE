################# Code for drawing the inference and running curves ######################################


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

# Please refer to the values in results.csv (computational requirements sheet)
Univariate_DAE_3_Case2_running = [66.624, 77.493, 92.699, 125.451, 144.559]
Univariate_DAE_2_Case2_running = [52.189, 53.817, 80.913, 129.726, 130.408]
Univariate_DAE_1_Case2_running = [58.678, 101.607, 123.730, 183.497, 240.983]
Multivariate_DAE_1_Case2_running = [63.489, 116.479, 177.898, 238.893, 308.494]
Multivariate_DAE_2_Case2_running = [44.718, 63.580, 90.857, 94.434, 151.359]
PIDAE_Case2_running = [41.553, 80.259, 96.288, 127.954, 171.306]
x_running = [0.1, 0.2, 0.3, 0.4, 0.5]

nan2 = np.empty((4,2))
nan2[:] = np.nan
nan4 = np.empty((4,4))
nan4[:] = np.nan
nan6 = np.empty((4,6))
nan6[:] = np.nan
nan8 = np.empty((4,8))
nan8[:] = np.nan

file_path = 'C:/Users/Antonio/Desktop/Projects/PIANN_Singapore_ACM/Results/Inference_time/'

Univariate_DAE_1_Case2_inference_01 = np.loadtxt(file_path+'time_univariate__t_ra_lambda0_Case2_len16.csv')
Univariate_DAE_2_Case2_inference_01 = np.loadtxt(file_path+'time_univariate__q_heat_lambda0_Case2_len16.csv')
Univariate_DAE_3_Case2_inference_01 = np.loadtxt(file_path+'time_univariate__q_cool_lambda0_Case2_len16.csv')
Multivariate_DAE_1_Case2_inference_01 = np.loadtxt(file_path+'time_multivariate_lambda0_Case2_len16.csv')
Multivariate_DAE_2_Case2_inference_01 = np.loadtxt(file_path+'time_multivariate__t_oa_lambda0_Case2_len16.csv')
PIDAE_Case2_inference_01 = np.loadtxt(file_path+'time_multivariate__t_oa_lambda1_Case2_len16.csv')
x_inference_01 = np.arange(1, Univariate_DAE_1_Case2_inference_01.shape[1] + 1)

Univariate_DAE_1_Case2_inference_02 = np.hstack((np.loadtxt(file_path+'time_univariate__t_ra_lambda0_Case2_len14.csv'),nan2))
Univariate_DAE_2_Case2_inference_02 = np.hstack((np.loadtxt(file_path+'time_univariate__q_heat_lambda0_Case2_len14.csv'),nan2))
Univariate_DAE_3_Case2_inference_02 = np.hstack((np.loadtxt(file_path+'time_univariate__q_cool_lambda0_Case2_len14.csv'),nan2))
Multivariate_DAE_1_Case2_inference_02 = np.hstack((np.loadtxt(file_path+'time_multivariate_lambda0_Case2_len14.csv'),nan2))
Multivariate_DAE_2_Case2_inference_02 = np.hstack((np.loadtxt(file_path+'time_multivariate__t_oa_lambda0_Case2_len14.csv'),nan2))
PIDAE_Case2_inference_02 = np.hstack((np.loadtxt(file_path+'time_multivariate__t_oa_lambda1_Case2_len14.csv'),nan2))
x_inference_02 = np.arange(1, Univariate_DAE_1_Case2_inference_02.shape[1] + 1)

Univariate_DAE_1_Case2_inference_03 = np.hstack((np.loadtxt(file_path+'time_univariate__t_ra_lambda0_Case2_len12.csv'),nan4))
Univariate_DAE_2_Case2_inference_03 = np.hstack((np.loadtxt(file_path+'time_univariate__q_heat_lambda0_Case2_len12.csv'),nan4))
Univariate_DAE_3_Case2_inference_03 = np.hstack((np.loadtxt(file_path+'time_univariate__q_cool_lambda0_Case2_len12.csv'),nan4))
Multivariate_DAE_1_Case2_inference_03 = np.hstack((np.loadtxt(file_path+'time_multivariate_lambda0_Case2_len12.csv'),nan4))
Multivariate_DAE_2_Case2_inference_03 = np.hstack((np.loadtxt(file_path+'time_multivariate__t_oa_lambda0_Case2_len12.csv'),nan4))
PIDAE_Case2_inference_03 = np.hstack((np.loadtxt(file_path+'time_multivariate__t_oa_lambda1_Case2_len12.csv'),nan4))
x_inference_03 = np.arange(1, Univariate_DAE_1_Case2_inference_03.shape[1] + 1)

Univariate_DAE_1_Case2_inference_04 = np.hstack((np.loadtxt(file_path+'time_univariate__t_ra_lambda0_Case2_len10.csv'),nan6))
Univariate_DAE_2_Case2_inference_04 = np.hstack((np.loadtxt(file_path+'time_univariate__q_heat_lambda0_Case2_len10.csv'),nan6))
Univariate_DAE_3_Case2_inference_04 = np.hstack((np.loadtxt(file_path+'time_univariate__q_cool_lambda0_Case2_len10.csv'),nan6))
Multivariate_DAE_1_Case2_inference_04 = np.hstack((np.loadtxt(file_path+'time_multivariate_lambda0_Case2_len10.csv'),nan6))
Multivariate_DAE_2_Case2_inference_04 = np.hstack((np.loadtxt(file_path+'time_multivariate__t_oa_lambda0_Case2_len10.csv'),nan6))
PIDAE_Case2_inference_04 = np.hstack((np.loadtxt(file_path+'time_multivariate__t_oa_lambda1_Case2_len10.csv'),nan6))
x_inference_04 = np.arange(1, Univariate_DAE_1_Case2_inference_04.shape[1] + 1)

Univariate_DAE_1_Case2_inference_05 = np.hstack((np.loadtxt(file_path+'time_univariate__t_ra_lambda0_Case2_len8.csv'),nan8))
Univariate_DAE_2_Case2_inference_05 = np.hstack((np.loadtxt(file_path+'time_univariate__q_heat_lambda0_Case2_len8.csv'),nan8))
Univariate_DAE_3_Case2_inference_05 = np.hstack((np.loadtxt(file_path+'time_univariate__q_cool_lambda0_Case2_len8.csv'),nan8))
Multivariate_DAE_1_Case2_inference_05 = np.hstack((np.loadtxt(file_path+'time_multivariate_lambda0_Case2_len8.csv'),nan8))
Multivariate_DAE_2_Case2_inference_05 = np.hstack((np.loadtxt(file_path+'time_multivariate__t_oa_lambda0_Case2_len8.csv'),nan8))
PIDAE_Case2_inference_05 = np.hstack((np.loadtxt(file_path+'time_multivariate__t_oa_lambda1_Case2_len8.csv'),nan8))
x_inference_05 = np.arange(1, Univariate_DAE_1_Case2_inference_05.shape[1] + 1)


n = 20

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

axs[0].plot(x_running, Univariate_DAE_1_Case2_running, label= 'Univariate_DAE_1', color='blue')
axs[0].plot(x_running, Univariate_DAE_2_Case2_running, label= 'Univariate_DAE_2', color='black')
axs[0].plot(x_running, Univariate_DAE_3_Case2_running, label= 'Univariate_DAE_3', color='purple')
axs[0].plot(x_running, Multivariate_DAE_1_Case2_running, label= 'Multivariate_DAE_1', color='orange')
axs[0].plot(x_running, Multivariate_DAE_2_Case2_running, label= 'Multivariate_DAE_2', color='green')
axs[0].plot(x_running, PIDAE_Case2_running, label= 'PIDAE', color='red')
axs[0].set_xlabel('Training rate [-]', fontsize=n)
axs[0].set_ylabel('Running time [s]', fontsize=n)
axs[0].tick_params(axis='both', which='major', labelsize=n)

axs[1].plot(x_inference_01,  np.nanmean(np.vstack((Univariate_DAE_1_Case2_inference_01,Univariate_DAE_1_Case2_inference_02,Univariate_DAE_1_Case2_inference_03,Univariate_DAE_1_Case2_inference_04,Univariate_DAE_1_Case2_inference_05)),axis=0), label= 'Univariate_DAE_1', color='blue')
axs[1].plot(x_inference_01,  np.nanmean(np.vstack((Univariate_DAE_2_Case2_inference_01,Univariate_DAE_2_Case2_inference_02,Univariate_DAE_2_Case2_inference_03,Univariate_DAE_2_Case2_inference_04,Univariate_DAE_2_Case2_inference_05)),axis=0), label= 'Univariate_DAE_2', color='black')
axs[1].plot(x_inference_01,  np.nanmean(np.vstack((Univariate_DAE_3_Case2_inference_01,Univariate_DAE_3_Case2_inference_02,Univariate_DAE_3_Case2_inference_03,Univariate_DAE_3_Case2_inference_04,Univariate_DAE_3_Case2_inference_05)),axis=0), label= 'Univariate_DAE_3', color='purple')
axs[1].plot(x_inference_01,  np.nanmean(np.vstack((Multivariate_DAE_1_Case2_inference_01,Multivariate_DAE_1_Case2_inference_02,Multivariate_DAE_1_Case2_inference_03,Multivariate_DAE_1_Case2_inference_04,Multivariate_DAE_1_Case2_inference_05)),axis=0), label= 'Multivariate_DAE_1', color='orange')
axs[1].plot(x_inference_01,  np.nanmean(np.vstack((Multivariate_DAE_2_Case2_inference_01,Multivariate_DAE_2_Case2_inference_02,Multivariate_DAE_2_Case2_inference_03,Multivariate_DAE_2_Case2_inference_04,Multivariate_DAE_2_Case2_inference_05)),axis=0), label= 'Multivariate_DAE_2', color='green')
axs[1].plot(x_inference_01,  np.nanmean(np.vstack((PIDAE_Case2_inference_01,PIDAE_Case2_inference_02,PIDAE_Case2_inference_03,PIDAE_Case2_inference_04,PIDAE_Case2_inference_05)),axis=0), label= 'PIDAE', color='red')
axs[1].set_xlabel('Days', fontsize=n)
axs[1].set_ylabel('Inference time [s]', fontsize=n)
axs[1].tick_params(axis='both', which='major', labelsize=n)

plt.tight_layout()
plt.show()
plt.close()
