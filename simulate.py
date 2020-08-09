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

def ar2(value1,value2,coef1,coef2,mu,sigma):
    """
    AR(2) model, cfr. paper
    """
    return coef1*value1+coef2*value2 + np.random.randn()*sigma+mu

def ar1(value1,coef1,mu,sigma):
    """
    AR(1) model, cfr. paper
    """
    return coef1*value1 + np.random.randn()*sigma+mu

def generate_jumpingmean(window_size, stride=1, nr_cp=49, delta_t_cp = 100, delta_t_cp_std = 10, scale_min=-1, scale_max=1):
    """
    Generates one instance of a jumping mean time series, together with the corresponding windows and parameters
    """
    mu = np.zeros((nr_cp,))
    parameters_jumpingmean = []
    for n in range(1,nr_cp):
        mu[n] = mu[n-1] + n/16
    for n in range(nr_cp):
        nr = int(delta_t_cp+np.random.randn()*np.sqrt(delta_t_cp_std))
        parameters_jumpingmean.extend(mu[n]*np.ones((nr,)))
    
    parameters_jumpingmean = np.array([parameters_jumpingmean]).T

    ts_length = len(parameters_jumpingmean)
    timeseries = np.zeros((ts_length))
    for i in range(2,ts_length):
        #print(ar2(timeseries[i-1],timeseries[i-2], 0.6,-0.5, parameters_jumpingmean[i], 1.5))
        timeseries[i] = ar2(timeseries[i-1],timeseries[i-2], 0.6,-0.5, parameters_jumpingmean[i], 1.5)

    windows = utils.ts_to_windows(timeseries, 0, window_size, stride)
    windows = utils.minmaxscale(windows,scale_min,scale_max)
    
    return timeseries, windows, parameters_jumpingmean

def generate_scalingvariance(window_size, stride=1, nr_cp=49, delta_t_cp = 100, delta_t_cp_std = 10, scale_min=-1, scale_max=1):
    """
    Generates one instance of a scaling variance time series, together with the corresponding windows and parameters
    """
    sigma = np.ones((nr_cp,))
    parameters_scalingvariance = []
    for n in range(1,nr_cp-1,2):
        sigma[n] = np.log(np.exp(1)+n/4)
    for n in range(nr_cp):
        nr = int(delta_t_cp+np.random.randn()*np.sqrt(delta_t_cp_std))
        parameters_scalingvariance.extend(sigma[n]*np.ones((nr,)))

    parameters_scalingvariance = np.array([parameters_scalingvariance]).T

    
    ts_length = len(parameters_scalingvariance)
    timeseries = np.zeros((ts_length))
    for i in range(2,ts_length):
        #print(ar2(timeseries[i-1],timeseries[i-2], 0.6,-0.5, parameters_jumpingmean[i], 1.5))
        timeseries[i] = ar2(timeseries[i-1],timeseries[i-2], 0.6,-0.5, 0, parameters_scalingvariance[i])

    windows = utils.ts_to_windows(timeseries, 0, window_size, stride)
    windows = utils.minmaxscale(windows,scale_min,scale_max)
    
    return timeseries, windows, parameters_scalingvariance

def generate_gaussian(window_size, stride=1, nr_cp=49, delta_t_cp = 100, delta_t_cp_std = 10, scale_min=-1, scale_max=1):
    """
    Generates one instance of a Gaussian mixtures time series, together with the corresponding windows and parameters
    """
    mixturenumber = np.zeros((nr_cp,))
    parameters_gaussian = []
    for n in range(1,nr_cp-1,2):
        mixturenumber[n] = 1
    for n in range(nr_cp):
        nr = int(delta_t_cp+np.random.randn()*np.sqrt(delta_t_cp_std))
        parameters_gaussian.extend(mixturenumber[n]*np.ones((nr,)))

    parameters_gaussian = np.array([parameters_gaussian]).T

    ts_length = len(parameters_gaussian)
    timeseries = np.zeros((ts_length))
    for i in range(2,ts_length):
        #print(ar2(timeseries[i-1],timeseries[i-2], 0.6,-0.5, parameters_jumpingmean[i], 1.5))
        if parameters_gaussian[i] == 0:
            timeseries[i] = 0.5*0.5*np.random.randn()+0.5*0.5*np.random.randn()
        else:
            timeseries[i] = -0.6 - 0.8*1*np.random.randn() + 0.2*0.1*np.random.randn()

    windows = utils.ts_to_windows(timeseries, 0, window_size, stride)
    windows = utils.minmaxscale(windows,scale_min,scale_max)
    
    return timeseries, windows, parameters_gaussian

def generate_changingcoefficients(window_size, stride=1, nr_cp=49, delta_t_cp = 1000, delta_t_cp_std = 10, scale_min=-1, scale_max=1):
    """
    Generates one instance of a changing coefficients time series, together with the corresponding windows and parameters
    """
    coeff = np.ones((nr_cp,))
    parameters = []
    for n in range(0,nr_cp,2):
        coeff[n] = np.random.rand()*0.5
    for n in range(1,nr_cp-1,2):
        coeff[n] = np.random.rand()*0.15+0.8
    
    for n in range(nr_cp):
        nr = int(delta_t_cp+np.random.randn()*np.sqrt(delta_t_cp_std))
        parameters.extend(coeff[n]*np.ones((nr,)))
        
    #parameters = np.array([parameters]).T
    parameters = ts_to_windows(parameters,0,1,stride)

    ts_length = len(parameters)
    timeseries = np.zeros((ts_length))
    for i in range(1,ts_length):
        #print(ar2(timeseries[i-1],timeseries[i-2], 0.6,-0.5, parameters_jumpingmean[i], 1.5))
        timeseries[i] = ar1(timeseries[i-1],parameters[i], 0,1)

    windows = utils.ts_to_windows(timeseries, 0, window_size, stride)
    windows = utils.minmaxscale(windows,scale_min,scale_max)
    
    return timeseries, windows, parameters
