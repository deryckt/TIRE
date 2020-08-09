import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
import warnings
import time, copy

import utils

def create_parallel_aes(window_size_per_ae,
                       intermediate_dim=0,
                       latent_dim=1,
                       nr_ae=3,
                       nr_shared=1,
                       loss_weight=1):
    """
    Create a Tensorflow model with parallel autoencoders, as visualized in Figure 1 of the TIRE paper.
    
    Args:
        window_size_per_ae: window size for the AE
        intermediate_dim: intermediate dimension for stacked AE, for single-layer AE use 0
        latent_dim: latent dimension of AE
        nr_ae: number of parallel AEs (K in paper)
        nr_shared: number of shared features (should be <= latent_dim)
        loss_weight: lambda in paper
        
    Returns:
        A parallel AE model instance, its encoder part and its decoder part
    """
    wspa = window_size_per_ae
    
    x = Input(shape=(nr_ae,wspa,))
    
    if intermediate_dim == 0:
        y=x
    else:
        y = Dense(intermediate_dim, activation=tf.nn.relu)(x)
        #y = tf.keras.layers.BatchNormalization()(y)
        
    z_shared = Dense(nr_shared, activation=tf.nn.tanh)(y)
    z_unshared = Dense(latent_dim-nr_shared, activation=tf.nn.tanh)(y)
    z = tf.concat([z_shared,z_unshared],-1)
    
    
    if intermediate_dim == 0:
        y=z
    else:
        y = Dense(intermediate_dim, activation=tf.nn.relu)(z)
        #y = tf.keras.layers.BatchNormalization()(y)
        
    x_decoded = Dense(wspa,activation=tf.nn.tanh)(y)
    
    pae = Model(x,x_decoded)
    encoder = Model(x,z)
    
    input_decoder = Input(shape=(nr_ae, latent_dim,))
    if intermediate_dim == 0:
        layer1 = pae.layers[-1]
        decoder = Model(input_decoder, layer1(input_decoder))
    else:
        layer1 = pae.layers[-1]
        layer2 = pae.layers[-2]
        decoder = Model(input_decoder, layer1(layer2(input_decoder)))
            
    pae.summary()
    
    def pae_loss(x,x_decoded):
        squared_diff = K.square(x-x_decoded)
        mse_loss = tf.reduce_mean(squared_diff)
        
        square_diff2 = K.square(z_shared[:,1:,:]-z_shared[:,:nr_ae-1,:])
        shared_loss = tf.reduce_mean(square_diff2)
        
        return mse_loss + loss_weight*shared_loss
    
    squared_diff = K.square(x-x_decoded)
    mse_loss = tf.reduce_mean(squared_diff)
        
    square_diff2 = K.square(z_shared[:,1:,:]-z_shared[:,:nr_ae-1,:])
    shared_loss = tf.reduce_mean(square_diff2)
    total_loss = mse_loss + loss_weight*shared_loss
    
    pae.add_loss(total_loss)
    
    return pae, encoder, decoder

def prepare_input_paes(windows,nr_ae):
    """
    Prepares input for create_parallel_ae
    
    Args:
        windows: list of windows
        nr_ae: number of parallel AEs (K in paper)
        
    Returns:
        array with shape (nr_ae, (nr. of windows)-K+1, window size)
    """
    new_windows = []
    nr_windows = windows.shape[0]
    for i in range(nr_ae):
        new_windows.append(windows[i:nr_windows-nr_ae+1+i])
    return np.transpose(new_windows,(1,0,2))

def train_AE(windows, intermediate_dim=0, latent_dim=1, nr_shared=1, nr_ae=3, loss_weight=1, nr_epochs=200, nr_patience=200):
    """
    Creates and trains an autoencoder with a Time-Invariant REpresentation (TIRE)
    
    Args:
        windows: time series windows (i.e. {y_t}_t or {z_t}_t in the notation of the paper)
        intermediate_dim: intermediate dimension for stacked AE, for single-layer AE use 0
        latent_dim: latent dimension of AE
        nr_shared: number of shared features (should be <= latent_dim)
        nr_ae: number of parallel AEs (K in paper)
        loss_weight: lambda in paper
        nr_epochs: number of epochs for training
        nr_patience: patience for early stopping
        
    Returns:
        returns the TIRE encoded windows for all windows
    """
    window_size_per_ae = windows.shape[-1]
    
    new_windows = prepare_input_paes(windows,nr_ae)

    pae, encoder, decoder = create_parallel_aes(window_size_per_ae,intermediate_dim,latent_dim,nr_ae,nr_shared,loss_weight)
    pae.compile(optimizer='adam')

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=nr_patience)

    pae.fit(new_windows,
                                  epochs=nr_epochs,
                                  verbose=1,
                                  batch_size=64,
                                  shuffle=True,
                                  validation_split=0.0,
                                  initial_epoch=0,
                                  callbacks=[callback]
                                  )

    #reconstruct = pae.predict(new_windows)
    encoded_windows_pae = encoder.predict(new_windows)
    encoded_windows = np.concatenate((encoded_windows_pae[:,0,:nr_shared],encoded_windows_pae[-nr_ae+1:,nr_ae-1,:nr_shared]),axis=0)

    return encoded_windows

def smoothened_dissimilarity_measures(encoded_windows, encoded_windows_fft, domain, window_size):
    """
    Calculation of smoothened dissimilarity measures
    
    Args:
        encoded_windows: TD latent representation of windows
        encoded_windows_fft:  FD latent representation of windows
        domain: TD/FD/both
        parameters: array with used parameters
        window_size: window size used
        par_smooth
        
    Returns:
        smoothened dissimilarity measures
    """
    
    if domain == "TD":
        encoded_windows_both = encoded_windows
    elif domain == "FD":
        encoded_windows_both = encoded_windows_fft
    elif domain == "both":
        beta = np.quantile(utils.distance(encoded_windows, window_size), 0.95)
        alpha = np.quantile(utils.distance(encoded_windows_fft, window_size), 0.95)
        encoded_windows_both = np.concatenate((encoded_windows*alpha, encoded_windows_fft*beta),axis=1)
    
    encoded_windows_both = utils.matched_filter(encoded_windows_both, window_size)
    distances = utils.distance(encoded_windows_both, window_size)
    distances = utils.matched_filter(distances, window_size)
    
    return distances

def change_point_score(distances, window_size):
    """
    Gives the change point score for each time stamp. A change point score > 0 indicates that a new segment starts at that time stamp.
    
    Args:
    distances: postprocessed dissimilarity measure for all time stamps
    window_size: window size used in TD for CPD
        
    Returns:
    change point scores for every time stamp (i.e. zero-padded such that length is same as length time series)
    """
    
    prominences = np.array(utils.new_peak_prominences(distances)[0])
    prominences = prominences/np.amax(prominences)
    return np.concatenate((np.zeros((window_size,)), prominences, np.zeros((window_size-1,))))
