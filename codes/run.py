################# Code containing tuning, training, evaluation, drawing functions and baselines ######################################


import tensorflow as tf
import numpy as np
import pickle5 as pickle
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import pandas as pd
from pathlib import Path
import gc
import sys
import optuna
from optuna.trial import TrialState
import math
import support_fun
import support_class
from sklearn.impute import KNNImputer
import time

# Plot figures with computer modern font
import matplotlib.font_manager as font_manager
import matplotlib
matplotlib.rcParams['font.family']='serif'
matplotlib.rcParams['font.serif']=['Times New Roman'] + plt.rcParams['font.serif']
matplotlib.rcParams['mathtext.fontset']='cm'
matplotlib.rcParams['axes.unicode_minus']=False


def Tuning(dataset_dir, missing, aug, corr, train_rate, lambdaa, a, b, c, strides, epochs, features, target_, threshold_q_cool, threshold_q_heat):

    # Convert target into an integer
    my_dict = {'q_cool': 0, 'q_heat': 1, 't_ra': 2}
    target = my_dict[target_]

    # Load dataset and convert it into a matrix (days, timesteps per day, features)
    with open(dataset_dir, 'rb') as handle:  # load preprocessed data
        dataset = (pickle.load(handle))  # [timestamp, t_oa, p_cool, p_heat, t_ra]

    dataset = np.reshape(dataset, (int(len(dataset) / 48), 48, 5))

    # Select building operation periods
    indice = support_fun.generate_indice_full(dataset, threshold_q_cool=threshold_q_cool,threshold_q_heat=threshold_q_heat)
    dataset = dataset[indice]

    # Devide into training, validation and evaluation
    trX_dataset, teX_dataset, ttX_dataset = np.split(dataset, [int(train_rate * len(dataset)),int((train_rate + 0.1) * len(dataset))])

    # Prepare data for tuning
    trX_n, teX_n, trX_mask_noisy, teX_mask_noisy = support_fun.prepare_data_training(np.copy(trX_dataset),np.copy(teX_dataset), corr,missing, aug)

    trX_n_noisy = np.copy(trX_n)    # Corrupted normalized training data
    teX_n_noisy = np.copy(teX_n)    # Corrupted normalized validation data

    # Replace values at the corrupted indexes with zeros
    trX_n_noisy[trX_mask_noisy] = 0
    teX_n_noisy[teX_mask_noisy] = 0


    if features == 1:  # univariate autoencoder

        # Select single variable
        trX_n_noisy = trX_n_noisy[:, :, target + 1:target + 2]
        trX_n = trX_n[:, :, target + 1:target + 2]
        teX_n_noisy = teX_n_noisy[:, :, target + 1:target + 2]
        teX_n = teX_n[:, :, target + 1:target + 2]
    elif features == 3:  # multivariate autoencoder

        # Drop outdoor air temperature variable
        trX_n_noisy = trX_n_noisy[:, :, 1:]
        trX_n = trX_n[:, :, 1:]
        teX_n_noisy = teX_n_noisy[:, :, 1:]
        teX_n = teX_n[:, :, 1:]
    elif features == 4:  # multivariate autoencoder with t_oa

        # De-corrupt outdoor air temperature variable
        trX_n_noisy[:, :, 0] = trX_n[:, :, 0]
        teX_n_noisy[:, :, 0] = teX_n[:, :, 0]
    else:
        print("error")

    # Optimize hyperparameters using optuna
    study = optuna.create_study()

    study.optimize(support_fun.tune_fun(trX_n_noisy, trX_n, teX_n_noisy, teX_n, lambdaa, a, b, c, strides, epochs, features), n_trials=200)

    complete_trials = study.get_trials(deepcopy=False)

    print("Study statistics: ")
    print("Number of complete trials:", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("Value:", trial.value)

    print("Params:")
    for key, value in trial.params.items():
        print("{}: {}".format(key, value))

def Training(dataset_dir, results_dir, missing, aug, corr, train_rate, lambdaa, a, b, c, filters1, filters2, filters_size, strides, lr, batch_size, epochs, features, target_, threshold_q_cool, threshold_q_heat):

        # Start recording running time
        start_time = time.time()

        # Define new variables to save the model
        name = str(corr).replace('.', '_')       # String for corruption rate
        esp = str(train_rate).replace('.', '_')  # String for training rate
        lam = str(lambdaa).replace('.', '')      # String for lambda

        # Convert target into an integer
        my_dict = {'q_cool': 0, 'q_heat': 1, 't_ra': 2}
        target = my_dict[target_]

        # Load dataset and convert it into a matrix (days, timesteps per day, features)
        with open(dataset_dir,'rb') as handle:  # load preprocessed data
            dataset = (pickle.load(handle))  # [timestamp, t_oa, p_cool, p_heat, t_ra]

        dataset = np.reshape(dataset, (int(len(dataset) / 48), 48, 5))

        # Select building operation periods
        indice = support_fun.generate_indice_full(dataset, threshold_q_cool=threshold_q_cool, threshold_q_heat=threshold_q_heat)
        dataset = dataset[indice]

        # Devide into training, validation and evaluation
        trX_dataset, teX_dataset, ttX_dataset = np.split(dataset, [int(train_rate * len(dataset)),int((train_rate + 0.1) * len(dataset))])
        print(len(trX_dataset))

        # Initialize the validation loss
        MSE_array = []

        # Run 10 times
        for i in range(10):
          g = tf.Graph()
          with g.as_default():

                # Prepare data for training
                trX_n, teX_n, trX_mask_noisy, teX_mask_noisy = support_fun.prepare_data_training(np.copy(trX_dataset), np.copy(teX_dataset), corr, missing, aug)

                trX_n_noisy = np.copy(trX_n)  # Corrupted normalized training data
                teX_n_noisy = np.copy(teX_n)  # Corrupted normalized validation data

                # Replace values at the corrupted indexes with zeros
                trX_n_noisy[trX_mask_noisy] = 0
                teX_n_noisy[teX_mask_noisy] = 0

                if features == 1:  # univariate autoencoder

                    # Select single variable
                    trX_n_noisy = trX_n_noisy[:, :, target+1:target+2]
                    trX_n = trX_n[:, :, target+1:target+2]
                    teX_n_noisy = teX_n_noisy[:, :, target+1:target+2]
                    teX_n = teX_n[:, :, target+1:target+2]
                    tar = 'univariate__' + target_
                elif features == 3:  # multivariate autoencoder

                    # Drop outdoor air temperature variable
                    trX_n_noisy = trX_n_noisy[:, :, 1:]
                    trX_n = trX_n[:, :, 1:]
                    teX_n_noisy = teX_n_noisy[:, :, 1:]
                    teX_n = teX_n[:, :, 1:]
                    tar = 'multivariate'
                elif features == 4:  # multivariate autoencoder with t_oa

                    # De-corrupt outdoor air temperature variable
                    trX_n_noisy[:, :, 0] = trX_n[:, :, 0]
                    teX_n_noisy[:, :, 0] = teX_n[:, :, 0]
                    tar = 'multivariate__t_oa'
                else:
                    print("error")

                # Assign float32 type to the data
                trX_n_noisy = trX_n_noisy.astype('float32')
                trX_n = trX_n.astype('float32')
                teX_n_noisy = teX_n_noisy.astype('float32')
                teX_n = teX_n.astype('float32')

                # Fit the model
                autoencoder, best_val_custom_metric, best_a_tensor, best_b_tensor, best_c_tensor = support_fun.fit_predict(trX_n_noisy, trX_n, teX_n_noisy, teX_n, lambdaa, a, b, c, filters1, filters2, filters_size, strides, lr, batch_size, epochs, features)

                # Save the trained model
                support_fun.save_models(results_dir, threshold_q_cool, threshold_q_heat, esp, tar, missing, name, i, lam, aug, features, lambdaa, best_a_tensor, best_b_tensor, best_c_tensor, autoencoder)

                # Save the validation loss for each of the 10 run
                MSE_array = np.append(MSE_array, best_val_custom_metric)

        # Select best model (minimum validation loss)
        best = np.int(np.where(MSE_array == np.min(MSE_array))[0])

        # Delete the rest of the models
        support_fun.select_models(best, results_dir, threshold_q_cool, threshold_q_heat, esp, tar, missing, name, lam, aug, features, lambdaa, n=10)

        # Get total running time
        timee = time.time() - start_time

        return timee

def Evaluation(dataset_dir, results_dir, missing,aug,corr,train_rate, lambdaa, filters1, filters2, filters_size, strides, batch_size, features, target_, threshold_q_cool, threshold_q_heat, print_coeff):

    # Define new variables to save the model
    name = str(corr).replace('.', '_')
    esp = str(train_rate).replace('.', '_')
    lam = str(lambdaa).replace('.', '')

    # Convert target into an integer
    my_dict = {'q_cool': 0, 'q_heat': 1, 't_ra': 2}
    target = my_dict[target_]

    # Load dataset and convert it into a matrix (days, timesteps per day, features)
    with open(dataset_dir, 'rb') as handle:  # load preprocessed data
        dataset = (pickle.load(handle))  # [timestamp, t_oa, p_cool, p_heat, t_ra]

    dataset = np.reshape(dataset, (int(len(dataset) / 48), 48, 5))

    # Select building operation periods
    indice = support_fun.generate_indice_full(dataset, threshold_q_cool=threshold_q_cool, threshold_q_heat=threshold_q_heat)
    dataset = dataset[indice]

    # Devide into training, validation and evaluation
    trX_dataset, teX_dataset, ttX_dataset = np.split(dataset, [int(train_rate * len(dataset)),int((train_rate + 0.1) * len(dataset))])

    # Define a variable tar to load the model and feauture_ to define the size of the evaluation metrics
    tar = None
    features_ = None
    ttX_p = None
    if features == 1:  # univariate autoencoder
        tar = 'univariate__' + target_
        features_ = features
    elif features == 3:  # multivariate autoencoder
        tar = 'multivariate'
        features_ = features
    elif features == 4:  # multivariate autoencoder with t_oa
        tar = 'multivariate__t_oa'
        features_ = 3
    else:
        print("error")

    #  Load the model
    autoencoder = support_fun.load_models(results_dir, threshold_q_cool, threshold_q_heat, esp, tar, missing, name, lam, aug, features, lambdaa, filters1, filters2, filters_size, strides, print_coeff)

    # Initialize the evaluation metrics
    RMSE_arr = np.zeros((len(ttX_dataset), features_))

    # Evaluate 100 times to account for different missing elements
    for i in range(100):

        # Prepare data for evaluation
        timestamp, max, min, ttX_n, ttX_mask_noisy = support_fun.prepare_data_evaluation(np.copy(trX_dataset), np.copy(ttX_dataset), corr, missing)

        ttX_n_noisy = np.copy(ttX_n)    # Corrupted normalized evaluation data

        # Replace values at the corrupted indexes with zeros
        ttX_n_noisy[ttX_mask_noisy] = 0

        if features == 1:  # univariate autoencoder

            # Select single variable
            ttX_n_noisy = ttX_n_noisy[:, :, target + 1:target + 2]
            ttX_n = ttX_n[:, :, target + 1:target + 2]
            ttX_mask_noisy = tuple(np.transpose(tuple(t for t in zip(*ttX_mask_noisy) if t[-1] not in (1, 2, 3))))
            max = max[target + 1]
            min = min[target + 1]
        elif features == 3:  # multivariate autoencoder

            # Drop outdoor air temperature variable
            ttX_n_noisy = ttX_n_noisy[:, :, 1:]
            ttX_n = ttX_n[:, :, 1:]
            ttX_mask_noisy = tuple(np.transpose(tuple(t for t in zip(*ttX_mask_noisy) if t[-1] != 3)))
            max = max[1:]
            min = min[1:]
        elif features == 4:  # multivariate autoencoder

            # De-corrupt outdoor air temperature variable
            ttX_n_noisy[:, :, 0] = ttX_n[:, :, 0]
            ttX_n = ttX_n[:, :, 1:]
            ttX_mask_noisy = tuple(np.transpose(tuple(t for t in zip(*ttX_mask_noisy) if t[-1] != 3)))
            max = max[1:]
            min = min[1:]
        else:
            print("error")

        # Predict
        decoded = support_fun.predict(autoencoder, ttX_n_noisy, batch_size, features)

        # De-normalize the reconstructed data
        final_result_finale = decoded * (max - min) + min

        # De-normalize the normalized evaluation data
        ttX_p = ttX_n * (max - min) + min

        # Assign float type to the data
        final_result_finale = final_result_finale.astype(float)
        ttX_p = ttX_p.astype(float)

        # Select only the elements that were corrupted
        obs = np.int(np.round(corr * 48))
        ttX_p_corr = np.reshape(ttX_p[ttX_mask_noisy], (int(len(ttX_p[ttX_mask_noisy])/(features_*obs)), obs, features_))
        final_result_finale = np.reshape(final_result_finale[ttX_mask_noisy], (int(len(final_result_finale[ttX_mask_noisy])/(features_*obs)), obs, features_))

        # Apply the evaluation metrics and sum to devide over the 100 evaluations afterwards
        RMSE_value = support_fun.rmse_ses(ttX_p_corr, final_result_finale)
        RMSE_arr += RMSE_value

    # Average over the 100 evaluations
    RMSE = RMSE_arr / 100

    # Compute the average RMSE over the days
    RMSE_avg = np.mean(RMSE, axis=0)

    return RMSE_avg

def LIN(dataset_dir, missing,corr,train_rate, threshold_q_cool, threshold_q_heat):

    # Load dataset and convert it into a matrix (days, timesteps per day, features)
    with open(dataset_dir, 'rb') as handle:  # load preprocessed data
        dataset = (pickle.load(handle))  # [timestamp, t_oa, p_cool, p_heat, t_ra]

    dataset = np.reshape(dataset, (int(len(dataset) / 48), 48, 5))

    # Select building operation periods
    indice = support_fun.generate_indice_full(dataset, threshold_q_cool=threshold_q_cool, threshold_q_heat=threshold_q_heat)
    dataset = dataset[indice]

    # Devide into training, validation and evaluation
    trX_dataset, teX_dataset, ttX_dataset = np.split(dataset, [int(train_rate * len(dataset)),int((train_rate + 0.1) * len(dataset))])

    # Define a variable tar to load the model and feauture_ to define the size of the evaluation metrics
    features_ = 3

    # Initialize the evaluation metrics
    RMSE_arr = np.zeros((len(ttX_dataset), features_))

    # Evaluate 100 times to account for different missing elements
    for i in range(100):

        # Prepare data for evaluation
        timestamp, max, min, ttX_n, ttX_mask_noisy = support_fun.prepare_data_evaluation(np.copy(trX_dataset), np.copy(ttX_dataset), corr, missing)

        ttX_n_noisy = np.copy(ttX_n)    # Corrupted normalized evaluation data

        # Replace values at the corrupted indexes with NaNs
        ttX_n_noisy[ttX_mask_noisy] = np.NaN

        # Drop outdoor air temperature variable
        ttX_n_noisy = ttX_n_noisy[:, :, 1:]
        ttX_n = ttX_n[:, :, 1:]
        ttX_mask_noisy = tuple(np.transpose(tuple(t for t in zip(*ttX_mask_noisy) if t[-1] != 3)))
        max = max[1:]
        min = min[1:]

        # De-normalize the normalized evaluation data
        ttX_n_noisy = ttX_n_noisy * (max - min) + min
        ttX_p = ttX_n * (max - min) + min

        # Predict
        ttX_n_noisy = np.reshape(ttX_n_noisy, (int(len(ttX_n_noisy)*48), features_))
        final_result_finale = pd.DataFrame(ttX_n_noisy)
        final_result_finale = (final_result_finale.interpolate(axis=0)).to_numpy()
        final_result_finale = np.reshape(final_result_finale, (int(len(final_result_finale)/48), 48, features_))

        # Assign float type to the data
        final_result_finale = final_result_finale.astype(float)
        ttX_p = ttX_p.astype(float)

        # Select only the elements that were corrupted
        obs = np.int(np.round(corr * 48))
        ttX_p_corr = np.reshape(ttX_p[ttX_mask_noisy], (int(len(ttX_p[ttX_mask_noisy])/(features_*obs)), obs, features_))
        final_result_finale = np.reshape(final_result_finale[ttX_mask_noisy], (int(len(final_result_finale[ttX_mask_noisy])/(features_*obs)), obs, features_))

        # Apply the evaluation metrics and sum to devide over the 100 evaluations afterwards
        RMSE_value = support_fun.rmse_ses(ttX_p_corr, final_result_finale)
        RMSE_arr += RMSE_value

    # Average over the 100 evaluations
    RMSE = RMSE_arr / 100

    # Compute the average RMSE over the days
    RMSE_avg = np.mean(RMSE, axis=0)

    return RMSE_avg

def KNN(dataset_dir, missing,corr,train_rate, threshold_q_cool, threshold_q_heat):

    # Load dataset and convert it into a matrix (days, timesteps per day, features)
    with open(dataset_dir, 'rb') as handle:  # load preprocessed data
        dataset = (pickle.load(handle))  # [timestamp, t_oa, p_cool, p_heat, t_ra]

    dataset = np.reshape(dataset, (int(len(dataset) / 48), 48, 5))

    # Select building operation periods
    indice = support_fun.generate_indice_full(dataset, threshold_q_cool=threshold_q_cool, threshold_q_heat=threshold_q_heat)
    dataset = dataset[indice]

    # Devide into training, validation and evaluation
    trX_dataset, teX_dataset, ttX_dataset = np.split(dataset, [int(train_rate * len(dataset)),int((train_rate + 0.1) * len(dataset))])

    # Define a variable eauture_ to define the size of the evaluation metrics
    features_ = 3
    features = 4

    # Prepare data for training
    trX_n, teX_n, trX_mask_noisy, teX_mask_noisy = support_fun.prepare_data_training(np.copy(trX_dataset),np.copy(teX_dataset), corr,missing, aug=0)
    trX_n = np.reshape(trX_n, (int(len(trX_n) * 48), features))

    # Initialize the evaluation metrics
    RMSE_arr = np.zeros((len(ttX_dataset), features_))

    # Evaluate 100 times to account for different missing elements
    for i in range(100):

        # Prepare data for evaluation
        timestamp, max, min, ttX_n, ttX_mask_noisy = support_fun.prepare_data_evaluation(np.copy(trX_dataset), np.copy(ttX_dataset), corr, missing)

        ttX_n_noisy = np.copy(ttX_n)    # Corrupted normalized evaluation data

        # Replace values at the corrupted indexes with zeros
        ttX_n_noisy[ttX_mask_noisy] = np.NaN


        # De-corrupt outdoor air temperature variable
        ttX_n_noisy[:, :, 0] = ttX_n[:, :, 0]
        ttX_n = ttX_n[:, :, 1:]
        ttX_mask_noisy = tuple(np.transpose(tuple(t for t in zip(*ttX_mask_noisy) if t[-1] != 3)))
        max = max[1:]
        min = min[1:]

        # Reshape
        ttX_n_noisy = np.reshape(ttX_n_noisy, (int(len(ttX_n_noisy) * 48), features))

        # Create a KNNImputer object with n_neighbors=default and fit on the training data
        imputer = KNNImputer(n_neighbors=5)

        # Predict
        decoded_ = imputer.fit_transform(np.vstack((trX_n, ttX_n_noisy)))
        decoded = decoded_[len(trX_n):]

        # Reshape
        decoded = np.reshape(decoded, (int(len(decoded)/48), 48, features))

        # De-normalize the reconstructed data (drop outdoor air temperature column)
        final_result_finale = decoded[:, :, 1:] * (max - min) + min

        # De-normalize the normalized evaluation data
        ttX_p = ttX_n * (max - min) + min

        # Assign float type to the data
        final_result_finale = final_result_finale.astype(float)
        ttX_p = ttX_p.astype(float)

        # Select only the elements that were corrupted
        obs = np.int(np.round(corr * 48))
        ttX_p_corr = np.reshape(ttX_p[ttX_mask_noisy], (int(len(ttX_p[ttX_mask_noisy])/(features_*obs)), obs, features_))
        final_result_finale = np.reshape(final_result_finale[ttX_mask_noisy], (int(len(final_result_finale[ttX_mask_noisy])/(features_*obs)), obs, features_))

        # Apply the evaluation metrics and sum to devide over the 100 evaluations afterwards
        RMSE_value = support_fun.rmse_ses(ttX_p_corr, final_result_finale)
        RMSE_arr += RMSE_value

    # Average over the 100 evaluations
    RMSE = RMSE_arr / 100

    # Compute the average RMSE over the days
    RMSE_avg = np.mean(RMSE, axis=0)

    return RMSE_avg

def Draw(dataset_dir, results_dir, missing,aug,corr,train_rate, lambdaa, filters1, filters2, filters_size, strides, batch_size, features, target_, threshold_q_cool, threshold_q_heat, print_coeff):

    # Define new variables to save the model
    name = str(corr).replace('.', '_')
    esp = str(train_rate).replace('.', '_')
    lam = str(lambdaa).replace('.', '')

    # Convert target into an integer
    my_dict = {'q_cool': 0, 'q_heat': 1, 't_ra': 2}
    target = my_dict[target_]

    # Load dataset and convert it into a matrix (days, timesteps per day, features)
    with open(dataset_dir, 'rb') as handle:  # load preprocessed data
        dataset = (pickle.load(handle))  # [timestamp, t_oa, p_cool, p_heat, t_ra]

    dataset = np.reshape(dataset, (int(len(dataset) / 48), 48, 5))

    # Select building operation periods
    indice = support_fun.generate_indice_full(dataset, threshold_q_cool=threshold_q_cool,threshold_q_heat=threshold_q_heat)
    dataset = dataset[indice]

    # Devide into training, validation and evaluation
    trX_dataset, teX_dataset, ttX_dataset = np.split(dataset, [int(train_rate * len(dataset)),int((train_rate + 0.1) * len(dataset))])

    # Prepare data for drawing
    timestamp, max, min, ttX_n, ttX_mask_noisy,ttX_mask_NOTnoisy, trX_teX_p = support_fun.prepare_data_draw(dataset, trX_dataset, teX_dataset, ttX_dataset, corr, missing)

    # Define a variable tar to load the model and feauture_ to define the size of the evaluation metrics
    tar = None
    features_ = None
    if features == 1:  # univariate autoencoder
        tar = 'univariate__' + target_
        features_ = features
    elif features == 3:  # multivariate autoencoder
        tar = 'multivariate'
        features_ = features
    elif features == 4:  # multivariate autoencoder with t_oa
        tar = 'multivariate__t_oa'
        features_ = 3
    else:
        print("error")

    #  Load the model
    autoencoder = support_fun.load_models(results_dir, threshold_q_cool, threshold_q_heat, esp, tar, missing, name, lam, aug, features, lambdaa, filters1, filters2, filters_size, strides, print_coeff)

    ttX_n_noisy = np.copy(ttX_n) # Corrupted normalized evaluation data

    # Replace values at the corrupted indexes with zeros
    ttX_n_noisy[ttX_mask_noisy] = 0

    if features == 1:  # univariate autoencoder

        # Select single variable
        trX_teX_p = trX_teX_p[:, :, target + 1:target + 2]
        ttX_n_noisy = ttX_n_noisy[:, :, target + 1:target + 2]
        ttX_n = ttX_n[:, :, target + 1:target + 2]
        ttX_mask_NOTnoisy = tuple(np.transpose(tuple(t for t in zip(*ttX_mask_NOTnoisy) if t[-1] not in (1, 2, 3))))
        ttX_mask_noisy = tuple(np.transpose(tuple(t for t in zip(*ttX_mask_noisy) if t[-1] not in (1, 2, 3))))
        max = max[target + 1]
        min = min[target + 1]
    elif features == 3:  # multivariate autoencoder

        # Drop outdoor air temperature variable
        trX_teX_p = trX_teX_p[:, :, 1:]
        ttX_n_noisy = ttX_n_noisy[:, :, 1:]
        ttX_n = ttX_n[:, :, 1:]
        ttX_mask_NOTnoisy = tuple(np.transpose(tuple(t for t in zip(*ttX_mask_NOTnoisy) if t[-1] != 3)))
        ttX_mask_noisy = tuple(np.transpose(tuple(t for t in zip(*ttX_mask_NOTnoisy) if t[-1] != 3)))
        max = max[1:]
        min = min[1:]
    elif features == 4:  # multivariate autoencoder

        # De-corrupt outdoor air temperature variable
        trX_teX_p = trX_teX_p[:, :, 1:]
        ttX_n = ttX_n[:, :, 1:]
        ttX_mask_NOTnoisy = tuple(np.transpose(tuple(t for t in zip(*ttX_mask_NOTnoisy) if t[-1] != 3)))
        ttX_mask_noisy = tuple(np.transpose(tuple(t for t in zip(*ttX_mask_NOTnoisy) if t[-1] != 3)))
        max = max[1:]
        min = min[1:]
    else:
        print("error")

    # Predict
    decoded = support_fun.predict(autoencoder, ttX_n_noisy, batch_size, features)

    # De-normalize the reconstructed data
    final_result_finale = decoded * (max - min) + min

    # De-normalize the normalized evaluation data
    ttX_p = ttX_n * (max - min) + min

    # Assign float type to the data
    final_result_finale = final_result_finale.astype(float)
    ttX_p = ttX_p.astype(float)

    # Consider only the corrupted values (the rest is the same as the original data)
    final_result_finale[ttX_mask_NOTnoisy] = ttX_p[ttX_mask_NOTnoisy]

    # Define corrupted evaluation data with NaNs
    ttX_p_noisy_nan = np.copy(ttX_p)

    # Replace values at the corrupted indexes with NaNs
    ttX_p_noisy_nan[ttX_mask_noisy] = np.nan

    # Create a dataset with real training and validation data and reconstructed evaluation data
    datset_final_result_finale = np.vstack((trX_teX_p, final_result_finale))

    # Create a dataset with real training, validation data and evaluation data
    datset_p = np.vstack((trX_teX_p,ttX_p))

    # Create a dataset with real training and validation data and corrupted evaluation data
    datset_p_nan = np.vstack((trX_teX_p,ttX_p_noisy_nan))

    # Define a timestamp array
    timestamp = np.reshape(timestamp, (len(timestamp) * 48, 1))

    # Reshape the datasets for visualization
    datset_final_result_finale = np.reshape(datset_final_result_finale, (len(datset_final_result_finale) * 48, features_))
    datset_p = np.reshape(datset_p, (len(datset_p) * 48, features_))
    datset_p_nan = np.reshape(datset_p_nan, (len(datset_p_nan) * 48, features_))

    # Create a figure and axis object
    if features == 1:  # univariate autoencoder

        # Sort days in chronological order
        df = pd.DataFrame(data=np.hstack((timestamp, datset_final_result_finale, datset_p, datset_p_nan)),columns=['timestamp', tar+'_recon', tar+'_real', tar+'_nan'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # Define unit of measure
        if target == 0 or target == 1:
            unit = '[kW]'
        else:
            unit = '[deg C]'

        # Plot with recnstructed values
        fig, ax = plt.subplots()
        ax.plot(df.index, df[tar+'_recon'], label='Reconstructed', color='r')
        ax.plot(df.index, df[tar+'_real'], label='Corrupted', color='b', ls="--")
        ax.plot(df.index, df[tar+'_nan'], label='Real', color='b')
        ax.set_xlabel('Timestamp [-]', fontsize=50)
        #ax.set_ylabel(target_+'\u2002'+unit, fontsize=50)
        ax.set_ylabel('T_ra_avg\u2002'+unit, fontsize=50)
        #ax.set_title("Univariate_DAE"+"\u2002"+"(CR = " + str(corr * 100) + "%)", fontsize=50)
        ax.set_title("Univariate_DAE_1"+"\u2002"+"(CR = " + str(corr * 100) + "%)", fontsize=50)
        ax.legend(fontsize=50)
        xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
        ax.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45)  # Rotate x-axis tick labels by 45 degrees
        ax.tick_params(axis='both', which='major', labelsize=50)
        plt.show()
        plt.close()

        # Plot without reconstructed values
        fig, ax = plt.subplots()
        ax.plot(df.index, df[tar+'_real'], label='Corrupted', color='b', ls="--")
        ax.plot(df.index, df[tar+'_nan'], label='Real', color='b')
        ax.set_xlabel('Timestamp [-]', fontsize=50)
        #ax.set_ylabel(target_+'\u2002'+unit, fontsize=50)
        ax.set_ylabel('T_ra_avg\u2002'+unit, fontsize=50)
        #ax.set_title("Univariate_DAE"+"\u2002"+"(CR = " + str(corr * 100) + "%)", fontsize=50)
        ax.set_title("Univariate_DAE_1"+"\u2002"+"(CR = " + str(corr * 100) + "%)", fontsize=50)
        ax.legend(fontsize=50)
        xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
        ax.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45)  # Rotate x-axis tick labels by 45 degrees
        ax.tick_params(axis='both', which='major', labelsize=50)
        plt.show()

    else:  # multivariate autoencoder

        # Sort days in chronological order
        df = pd.DataFrame(data= np.hstack((timestamp, datset_final_result_finale, datset_p, datset_p_nan)),columns=['timestamp', 'q_cool_recon', 'q_heat_recon', 't_ra_recon', 'q_cool_real', 'q_heat_real', 't_ra_real', 'q_cool_nan', 'q_heat_nan', 't_ra_nan'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        tar_array = ['q_cool', 'q_heat', 't_ra']

        # Iterate over every variable to represent them singularly
        for i in range(3):

            variable = tar_array[i]

            # Define unit of measure
            if variable == 'q_heat' or variable == 'q_cool':
                unit = '[kW]'
            else:
                unit = '[deg C]'

            # Plot with recnstructed values
            fig, ax = plt.subplots()
            ax.plot(df.index, df[variable + '_recon'], label='Reconstructed', color='r')
            ax.plot(df.index, df[variable + '_real'], label='Corrupted', color='b', ls="--")
            ax.plot(df.index, df[variable + '_nan'], label='Real', color='b')
            ax.set_xlabel('Timestamp [-]', fontsize=50)
            ax.set_ylabel(variable + '\u2002' + unit, fontsize=50)
            if features == 3:
               ax.set_title("Multivariate_DAE_1" + "\u2002" + "(CR = " + str(corr * 100) + "%)", fontsize=50)
            elif features == 4 and lambdaa == 0:
                ax.set_title("Multivariate_DAE_2" + "\u2002" + "(CR = " + str(corr * 100) + "%)", fontsize=50)
            else:
                ax.set_title("PI-DAE" + "\u2002" + "(CR = " + str(corr * 100) + "%)", fontsize=50)
            ax.legend(fontsize=50)
            xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
            ax.xaxis.set_major_formatter(xfmt)
            plt.xticks(rotation=45)  # Rotate x-axis tick labels by 45 degrees
            ax.tick_params(axis='both', which='major', labelsize=50)
            plt.show()
            plt.close()

            # Plot without reconstructed values
            fig, ax = plt.subplots()
            ax.plot(df.index, df[variable + '_real'], label='Corrupted', color='b', ls="--")
            ax.plot(df.index, df[variable + '_nan'], label='Real', color='b')
            ax.set_xlabel('Timestamp [-]', fontsize=50)
            ax.set_ylabel(variable + '\u2002' + unit, fontsize=50)
            if features == 3:
               ax.set_title("Multivariate_DAE_1" + "\u2002" + "(CR = " + str(corr * 100) + "%)", fontsize=50)
            elif features == 4 and lambdaa == 0:
                ax.set_title("Multivariate_DAE_2" + "\u2002" + "(CR = " + str(corr * 100) + "%)", fontsize=50)
            else:
                ax.set_title("PI-DAE" + "\u2002" + "(CR = " + str(corr * 100) + "%)", fontsize=50)
            ax.legend(fontsize=50)
            xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
            ax.xaxis.set_major_formatter(xfmt)
            plt.xticks(rotation=45)  # Rotate x-axis tick labels by 45 degrees
            ax.tick_params(axis='both', which='major', labelsize=50)
            plt.show()
            plt.close()


def CompReq(dataset_dir, results_dir, missing,aug,corr,train_rate, lambdaa, filters1, filters2, filters_size, strides, batch_size, features, target_, threshold_q_cool, threshold_q_heat, print_coeff):

    # Define new variables to save the model
    name = str(corr).replace('.', '_')
    esp = str(train_rate).replace('.', '_')
    lam = str(lambdaa).replace('.', '')

    # Convert target into an integer
    my_dict = {'q_cool': 0, 'q_heat': 1, 't_ra': 2}
    target = my_dict[target_]

    # Load dataset and convert it into a matrix (days, timesteps per day, features)
    with open(dataset_dir, 'rb') as handle:  # load preprocessed data
        dataset = (pickle.load(handle))  # [timestamp, t_oa, p_cool, p_heat, t_ra]

    dataset = np.reshape(dataset, (int(len(dataset) / 48), 48, 5))

    # Select building operation periods
    indice = support_fun.generate_indice_full(dataset, threshold_q_cool=threshold_q_cool, threshold_q_heat=threshold_q_heat)
    dataset = dataset[indice]

    # Devide into training, validation and evaluation
    trX_dataset, teX_dataset, ttX_dataset = np.split(dataset, [int(train_rate * len(dataset)),int((train_rate + 0.1) * len(dataset))])

    # Define a variable tar to load the model and feauture_ to define the size of the evaluation metrics
    tar = None

    # Prepare data for evaluation
    timestamp, max, min, ttX_n, ttX_mask_noisy = support_fun.prepare_data_evaluation(np.copy(trX_dataset), np.copy(ttX_dataset), corr, missing)

    ttX_n_noisy = np.copy(ttX_n)    # Corrupted normalized evaluation data

    # Replace values at the corrupted indexes with zeros
    ttX_n_noisy[ttX_mask_noisy] = 0

    if features == 1:  # univariate autoencoder

        # Select single variable
        tar = 'univariate__' + target_
        ttX_n_noisy = ttX_n_noisy[:, :, target + 1:target + 2]
    elif features == 3:  # multivariate autoencoder

        # Drop outdoor air temperature variable
        tar = 'multivariate'
        ttX_n_noisy = ttX_n_noisy[:, :, 1:]
    elif features == 4:  # multivariate autoencoder

        # De-corrupt outdoor air temperature variable
        tar = 'multivariate__t_oa'
        ttX_n_noisy[:, :, 0] = ttX_n[:, :, 0]
    else:
        print("error")

    #  Load the model
    autoencoder = support_fun.load_models(results_dir, threshold_q_cool, threshold_q_heat, esp, tar, missing, name, lam, aug, features, lambdaa, filters1, filters2, filters_size, strides, print_coeff)

    # Get inference time and memory variables
    time = []

    for dd in range(len(ttX_n_noisy)):
        t = support_fun.computational_fun(ttX_n_noisy, autoencoder, batch_size, dd)
        time = np.append(time, t)

    return time

