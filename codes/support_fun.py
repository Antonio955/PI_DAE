################# Support functions ######################################


import numpy as np
import random
import os
import pandas as pd
from pathlib import Path
import gc
import sys
import support_class
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, Callback
from optuna.integration import TFKerasPruningCallback
import tensorflow as tf
import time
import psutil
import os

tf.logging.set_verbosity(tf.logging.ERROR)

def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted

def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)),axis=1)

def mae_ses(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(_error(actual, predicted)),axis=1)

def rmse_ses(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))

def corruption(input,corr,missing):
    """ corruption function """
    mask_noisy = np.zeros(shape=(input.shape[0], input.shape[1], input.shape[2]))
    if missing == 'continuous':
        length = np.int(np.round(corr * input.shape[1]))
        for row in range(input.shape[0]):
            start = np.random.randint(2, input.shape[1] - length - 2)
            mask_noisy[row, start:start + length, :] = np.nan
        return np.where(np.isnan(mask_noisy)), np.where(np.isfinite(mask_noisy))
    elif missing == 'random':
        for row in range(input.shape[0]):
            indices = np.random.choice(np.arange(2,input.shape[1]-2), replace=False,size=np.int(np.round(input.shape[1] * corr)))
            mask_noisy[row,indices, :] = np.nan
        return np.where(np.isnan(mask_noisy)), np.where(np.isfinite(mask_noisy))
    else:
        print("error: missing")

def augmentation(aug,t_p):
    """ augmentation function """
    if aug >= 1:
        t_p2 = np.copy(t_p)

        if aug == 1:
            t_p = np.vstack((t_p, t_p2))
        else:
            for rip in range(aug):
                t_p = np.vstack((t_p, t_p2))

        ind = np.arange(len(t_p))
        random.seed(123)
        random.shuffle(ind)

        t_p = t_p[ind,:,:]

    return t_p

def custom_metric(y_true, y_pred):
    """ define custom metric for the autoencoder """
    return tf.math.reduce_sum(tf.math.reduce_mean(tf.math.square(y_true - y_pred), [0, 1]))

def prepare_data_training(trX_dataset, teX_dataset, corr, missing, aug):
    """ aprepare data for training """
    trX_dataset = augmentation(aug, trX_dataset)
    teX_dataset2 = np.copy(teX_dataset)
    for r in range(10):
        teX_dataset = np.vstack((teX_dataset, teX_dataset2))


    trX_p = (trX_dataset[:, :, 1:]).astype(float) # [t_oa, p_cool, p_heat, t_ra]
    teX_p = (teX_dataset[:, :, 1:]).astype(float)

    max = np.max(trX_p, axis=(0, 1))
    min = np.min(trX_p, axis=(0, 1))

    trX_n = (trX_p - min) / (max - min)
    teX_n = (teX_p - min) / (max - min)

    trX_mask_noisy, _ = corruption(trX_p, corr, missing)
    teX_mask_noisy, _ = corruption(teX_p, corr, missing)

    return trX_n, teX_n, trX_mask_noisy, teX_mask_noisy


def prepare_data_evaluation(trX_dataset, ttX_dataset, corr, missing):
    """ prepare data for evaluation """
    timestamp = ttX_dataset[:, :, 0:1]  # [timestamp]
    trX_p = (trX_dataset[:, :, 1:]).astype(float) # [t_oa, q_cool, q_heat, t_ra]
    ttX_p = (ttX_dataset[:, :, 1:]).astype(float)

    max = np.max(trX_p, axis=(0, 1))
    min = np.min(trX_p, axis=(0, 1))

    ttX_n = (ttX_p - min) / (max - min)

    ttX_mask_noisy, _ = corruption(ttX_p, corr, missing)

    return timestamp, max, min, ttX_n, ttX_mask_noisy

def prepare_data_draw(dataset, trX_dataset, teX_dataset, ttX_dataset, corr, missing):
    """ prepare data for drawing """
    timestamp = dataset[:, :, 0:1]  # [timestamp]
    trX_p = (trX_dataset[:, :, 1:]).astype(float) # [t_oa, p_cool, p_heat, t_ra]
    teX_p = (teX_dataset[:, :, 1:]).astype(float)
    ttX_p = (ttX_dataset[:, :, 1:]).astype(float)

    trX_teX_p = np.vstack((trX_p, teX_p))

    max = np.max(trX_p, axis=(0, 1))
    min = np.min(trX_p, axis=(0, 1))


    ttX_n = (ttX_p - min) / (max - min)

    ttX_mask_noisy, ttX_mask_NOTnoisy = corruption(ttX_p, corr, missing)

    return timestamp, max, min, ttX_n, ttX_mask_noisy,ttX_mask_NOTnoisy, trX_teX_p

def generate_indice_full(dataset, threshold_q_cool, threshold_q_heat):
    """ get index to select the building operation period """
    if threshold_q_cool > 0 and threshold_q_heat > 0:
        indice = []
        for r in range(len(dataset)):
            per_75 = np.percentile(dataset[r,:,2:], 75, axis=0)
            per_25 = np.percentile(dataset[r,:,2:], 25, axis=0)
            if (per_75[0] - per_25[0]) > threshold_q_cool and (per_75[1] - per_25[1]) > threshold_q_heat and (per_75[2] - per_25[2]) < 1 and np.all(per_75 - per_25 > 0):
              indice.append(r)
            else:
                continue
    else:
        indice = np.arange(dataset.shape[0])
    return indice

def save_models(results_dir, threshold_q_cool, threshold_q_heat, esp, tar, missing, name, i, lam, aug, features, lambdaa, best_a_tensor, best_b_tensor, best_c_tensor, autoencoder):
    """ save models (and related physics-based coefficients if any) """
    Path(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/').mkdir(parents=True, exist_ok=True)
    autoencoder.save_weights(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + str(i) + 'CAE_' + lam + '_' + str(aug) + '.h5')

    if features == 4 and lambdaa > 0:
        # Save the best a_tensor and b_tensor values to disk
        np.save(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + str(i) + 'best_a_tensor.npy', best_a_tensor)
        np.save(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + str(i) + 'best_b_tensor.npy', best_b_tensor)
        np.save(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + str(i) + 'best_c_tensor.npy', best_c_tensor)

def load_models(results_dir, threshold_q_cool, threshold_q_heat, esp, tar, missing, name, lam, aug, features, lambdaa, filters1, filters2, filters_size, strides, print_coeff):
    """ load model and physics-based coefficients """
    if features == 4 and lambdaa > 0:
        best_a_tensor = np.load(results_dir + '/'+str(threshold_q_cool)+'_'+str(threshold_q_heat) +'/' +esp+'/'+tar+'/'+missing+'/'+name+'/'+'best_a_tensor.npy')
        best_b_tensor = np.load(results_dir + '/'+str(threshold_q_cool)+'_'+str(threshold_q_heat) +'/' +esp+'/'+tar+'/'+missing+'/'+name+'/'+'best_b_tensor.npy')
        best_c_tensor = np.load(results_dir + '/'+str(threshold_q_cool)+'_'+str(threshold_q_heat) +'/' +esp+'/'+tar+'/'+missing+'/'+name+'/'+'best_c_tensor.npy')
        if print_coeff == True:
            print("best_a_tensor", best_a_tensor)
            print("best_b_tensor", best_b_tensor)
            print("best_c_tensor", best_c_tensor)
    else:
        best_a_tensor = 1
        best_b_tensor = 1
        best_c_tensor = 1

    autoencoder = support_class.CAE_model(filters1, filters2, filters_size, strides, features,lambdaa, best_a_tensor, best_b_tensor, best_c_tensor)
    autoencoder.load_weights(results_dir + '/'+str(threshold_q_cool)+'_'+str(threshold_q_heat) +'/' +esp+'/'+tar+'/'+missing+'/'+name+'/'+'CAE_'+lam+'_'+str(aug)+'.h5')  # load the correct model

    return autoencoder

def select_models(best, results_dir, threshold_q_cool, threshold_q_heat, esp, tar, missing, name, lam, aug, features, lambdaa, n):
    """ discard non good models (and related physics-based coefficients if any) """
    for i in range(n):
        if i != best:
            os.remove(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + str(i) + 'CAE_' + lam + '_' + str(aug) + '.h5')

            if features == 4 and lambdaa > 0:
                os.remove(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + str(i) + 'best_a_tensor.npy')
                os.remove(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + str(i) + 'best_b_tensor.npy')
                os.remove(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + str(i) + 'best_c_tensor.npy')

    # Remove existing old models (and related physics-based coefficients if any)
    if os.path.exists(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + 'CAE_' + lam + '_' + str(aug) + '.h5'):
        os.remove(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + 'CAE_' + lam + '_' + str(aug) + '.h5')  # delete the existing file

    if features == 4 and lambdaa > 0 and os.path.exists(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + 'best_a_tensor.npy'):
        os.remove(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + 'best_a_tensor.npy')  # delete the existing file
    if features == 4 and lambdaa > 0 and os.path.exists(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + 'best_b_tensor.npy'):
        os.remove(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + 'best_b_tensor.npy')  # delete the existing file
    if features == 4 and lambdaa > 0 and os.path.exists(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + 'best_c_tensor.npy'):
        os.remove(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + 'best_c_tensor.npy')  # delete the existing file

    # Rename the best model (and related physics-based coefficients if any)
    os.rename(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + str(best) + 'CAE_' + lam + '_' + str(aug) + '.h5', results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + 'CAE_' + lam + '_' + str(aug) + '.h5')

    if features == 4 and lambdaa > 0:
        os.rename(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + str(best) + 'best_a_tensor.npy', results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + 'best_a_tensor.npy')
        os.rename(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + str(best) + 'best_b_tensor.npy', results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + 'best_b_tensor.npy')
        os.rename(results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + str(best) + 'best_c_tensor.npy', results_dir + '/' + str(threshold_q_cool) + '_' + str(threshold_q_heat) + '/' + esp + '/' + tar + '/' + missing + '/' + name + '/' + 'best_c_tensor.npy')


def tune_fun(trX_n_noisy, trX_n, teX_n_noisy, teX_n, lambdaa, a, b, c, strides, epochs, features):
    """ tune hyperparameters """
    def score(trial):
        filters1 = trial.suggest_int('filters1', 5, 200)
        filters2 = trial.suggest_int('filters2', 5, 200)
        filters_size = trial.suggest_int('filters_size', 1, 10)
        lr = trial.suggest_float('lr', 0.0001, 0.1)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

        if features >= 4 and lambdaa > 0:
            a_constraint = support_class.ScalarMinMaxConstraint(min_value=0.0, max_value=float('inf'))
            b_constraint = support_class.ScalarMinMaxConstraint(min_value=0.0, max_value=float('inf'))
            c_constraint = support_class.ScalarMinMaxConstraint(min_value=0.0, max_value=float('inf'))

            a_tensor = K.variable(a, dtype=tf.float32, name='a_tensor', constraint=a_constraint)
            b_tensor = K.variable(b, dtype=tf.float32, name='b_tensor', constraint=b_constraint)
            c_tensor = K.variable(c, dtype=tf.float32, name='c_tensor', constraint=c_constraint)

        else:
            a_tensor = K.constant(a, dtype=tf.float32, name='a_tensor')
            b_tensor = K.constant(b, dtype=tf.float32, name='b_tensor')
            c_tensor = K.constant(c, dtype=tf.float32, name='c_tensor')

        # Define callbacks
        save_best_coefficients_callback = support_class_.SaveBestCoefficients(a_tensor, b_tensor, c_tensor)
        pruning_callback = TFKerasPruningCallback(trial, monitor='val_custom_metric')
        early_stopping_callback = EarlyStopping(monitor='val_custom_metric', verbose=0, min_delta=0.0001, patience=20, mode='auto', restore_best_weights=True)

        # Dedine customized model
        autoencoder = support_class.CAE_model(filters1, filters2, filters_size, strides, features, lambdaa, a_tensor,b_tensor, c_tensor)

        # Compile the model
        autoencoder.compile(optimizer=optimizers.Adam(lr=lr), loss=autoencoder.loss,metrics=[custom_metric])

        # Fit
        autoencoder.fit(trX_n_noisy, trX_n, verbose=1,callbacks=[save_best_coefficients_callback, early_stopping_callback, pruning_callback], epochs=epochs,batch_size=batch_size, shuffle=False, validation_data=(teX_n_noisy, teX_n))

        # Get the best custom_metric from the SaveBestCoefficients callback
        best_custom_metric = save_best_coefficients_callback.best_val_loss

        return best_custom_metric
    return score

def fit_predict(trX_n_noisy, trX_n, teX_n_noisy, teX_n, lambdaa, a, b, c, filters1, filters2, filters_size, strides, lr, batch_size, epochs, features):
    """ fit and validate the model """
    if features == 4 and lambdaa > 0:
        a_constraint = support_class.ScalarMinMaxConstraint(min_value=0, max_value=float('inf'))
        b_constraint = support_class.ScalarMinMaxConstraint(min_value=0, max_value=float('inf'))
        c_constraint = support_class.ScalarMinMaxConstraint(min_value=0, max_value=float('inf'))

        a_tensor = K.variable(a, dtype=tf.float32, name='a_tensor', constraint=a_constraint)
        b_tensor = K.variable(b, dtype=tf.float32, name='b_tensor', constraint=b_constraint)
        c_tensor = K.variable(c, dtype=tf.float32, name='c_tensor', constraint=c_constraint)

    else:
        a_tensor = K.constant(a, dtype=tf.float32, name='a_tensor')
        b_tensor = K.constant(b, dtype=tf.float32, name='b_tensor')
        c_tensor = K.constant(c, dtype=tf.float32, name='c_tensor')

    # Define callbacks
    save_best_coefficients_callback = support_class.SaveBestCoefficients()
    early_stopping_callback = EarlyStopping(monitor='val_custom_metric', verbose=0, min_delta=0.0001, patience=20, mode='auto', restore_best_weights=True)

    # Dedine customized model
    autoencoder = support_class.CAE_model(filters1, filters2, filters_size, strides, features, lambdaa, a_tensor, b_tensor, c_tensor)

    # Compile the model
    autoencoder.compile(optimizer=optimizers.Adam(lr=lr), loss=autoencoder.loss, metrics=[custom_metric])

    # Fit
    autoencoder.fit(trX_n_noisy, trX_n, verbose=0, callbacks=[save_best_coefficients_callback, early_stopping_callback], epochs=epochs, batch_size=batch_size, shuffle=False, validation_data=(teX_n_noisy, teX_n))

    #autoencoder.summary()

    # Get the best a_tensor, b_tensor and custom_metric from the SaveBestCoefficients callback
    best_a_tensor = save_best_coefficients_callback.best_a_value
    best_b_tensor = save_best_coefficients_callback.best_b_value
    best_c_tensor = save_best_coefficients_callback.best_c_value
    best_val_custom_metric = save_best_coefficients_callback.best_val_custom_metric

    return autoencoder, best_val_custom_metric, best_a_tensor, best_b_tensor, best_c_tensor

def predict(autoencoder, ttX_n_noisy, batch_size, features):
    """ reconcstruct values """
    decoded_imgs = autoencoder.predict(ttX_n_noisy, batch_size=batch_size, verbose=0)

    # drop t_oa so that not to include it in the final evaluation
    if features == 4:
        decoded_imgs = decoded_imgs[:,:,1:]
    else:
        pass

    return decoded_imgs

def computational_fun(ttX_n_noisy, autoencoder, batch_size, i):
    """ inference time """
    # Wake up the model
    ttX_n_noisy__ = ttX_n_noisy[:2]
    start_time = time.time()
    autoencoder.predict(ttX_n_noisy__, batch_size=batch_size, verbose=0)
    time_i = time.time() - start_time

    ttX_n_noisy_ = ttX_n_noisy[:i+1]

    start_time = time.time()

    autoencoder.predict(ttX_n_noisy_, batch_size=batch_size, verbose=0)

    time_i = time.time() - start_time

    return time_i