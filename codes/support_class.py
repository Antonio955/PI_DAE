################# Support classes ######################################


from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, concatenate, Dense, Flatten, concatenate, Reshape
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras import optimizers
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import backend as K
import numpy as np
import pickle5 as pickle
import support_fun
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

class CAE_model(Model):
    """define the CAE model"""

    def __init__(self, filters1, filters2, filters_size, strides, features, lambdaa, a_tensor, b_tensor, c_tensor, name='CAE_model'):
        super().__init__(name=name)

        self.a_tensor = a_tensor
        self.b_tensor = b_tensor
        self.c_tensor = c_tensor
        self.features = features
        self.lambdaa = lambdaa

        model = Sequential()

        # endoder
        model.add(Conv1D(filters2, filters_size, strides=strides, activation='relu', padding='same', input_shape=(48, self.features)))
        model.add(MaxPooling1D((2), padding='same'))
        model.add(Conv1D(filters1, filters_size, strides=strides, activation='relu', padding='same'))
        model.add(MaxPooling1D((2), padding='same'))

        # decoder
        model.add(Conv1D(filters1, filters_size, strides=strides, activation='tanh', padding='same'))
        model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',gamma_initializer='ones', moving_mean_initializer='zeros',moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,beta_constraint=None, gamma_constraint=None))
        model.add(UpSampling1D((2)))
        model.add(Conv1D(filters2, filters_size, strides=strides, activation='tanh', padding='same'))
        model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',gamma_initializer='ones', moving_mean_initializer='zeros',moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,beta_constraint=None, gamma_constraint=None))
        model.add(UpSampling1D((2)))
        model.add(Conv1D(self.features, filters_size, strides=strides, activation='tanh', padding='same'))
        model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,beta_initializer='zeros', gamma_initializer='ones',moving_mean_initializer='zeros', moving_variance_initializer='ones',beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,gamma_constraint=None))

        self.model = model

    def call(self, x, training):

        # Forward pass
        y_pred = self.model(x, training)

        return y_pred

    def loss(self, y_true, y_pred):
        # regression loss
        loss_tot = tf.math.reduce_sum(tf.math.reduce_mean(tf.math.square(y_true - y_pred), [0, 1]))

        if self.features == 4 and self.lambdaa > 0:
            # select the reconstructed variables from y_pred
            dy = y_pred[:, 1:, 3:4] - y_pred[:, :-1, 3:4]
            t_oa = y_pred[:, :-1, 0:1] # reconstructed outdoor air temperature
            q_cool = y_pred[:, :-1, 1:2] # reconstructed cooling load
            q_heat = y_pred[:, :-1, 2:3] # reconstructed heating load
            t_ra = y_pred[:, :-1, 3:4] # reconstructed indoor air temperature
            q_amb = (t_oa - t_ra)

            # compute the ODE
            zero_comp = dy - (self.a_tensor * q_amb - self.b_tensor * q_cool + self.c_tensor * q_heat) / 10

            # force the ODE to be zero with a physics-based loss
            loss_phys = tf.math.reduce_mean(tf.math.square(zero_comp))

            # combine the losses
            loss_tot += self.lambdaa * loss_phys
        return loss_tot

    @tf.function
    def train_step(self, data):
        x, y = data

        if self.features == 4 and self.lambdaa > 0:
            # explicitly add a_tensor and b_tensor to the list of trainable weights
            trainable_vars = self.trainable_weights + [self.a_tensor, self.b_tensor, self.c_tensor]
        else:
            trainable_vars = self.trainable_weights

        with tf.GradientTape() as tape:
            tape.watch(x)
            y_pred = self.call(x, True)
            loss = self.loss(y, y_pred)

        # compute gradient of the trainable_vars with resect to loss
        grads_l = tape.gradient(loss, trainable_vars)
        grads_and_vars = zip(grads_l, trainable_vars)
        self.optimizer.apply_gradients(grads_and_vars)

        return loss

class SaveBestCoefficients(Callback):
    """define callback for physics-based coefficients"""
    def __init__(self):
        self.best_a_value = None
        self.best_b_value = None
        self.best_c_value = None

    def on_epoch_end(self, epoch, logs=None):
        val_custom_metric = logs['val_custom_metric']
        val_loss = logs['val_loss']
        custom_metric = logs['custom_metric']
        loss = logs['loss']
        a_ = K.get_session().run(self.model.a_tensor)
        b_ = K.get_session().run(self.model.b_tensor)
        c_ = K.get_session().run(self.model.c_tensor)
        if self.best_a_value is None or val_custom_metric < self.best_val_custom_metric:
            self.best_a_value = a_
            self.best_b_value = b_
            self.best_c_value = c_
            self.best_val_custom_metric = val_custom_metric

        # Print a_tensor, b_tensor, and c_tensor at the end of each epoch
        #print(f"epoch {epoch+1}: loss: {loss}, custom_metric: {custom_metric}, val_loss: {val_loss}, val_custom_metric: {val_custom_metric}, a_tensor: {a_}, b_tensor: {b_}, c_tensor: {c_}")

class ScalarMinMaxConstraint(Constraint):
    """constrain the weights to be between min_value and max_value"""

    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}