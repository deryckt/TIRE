# TIRE

**Please note that this repository is still in progress.**

**TIRE** is an autoencoder-based change point detection algorithm for time series data that uses a TIme-Invariant Representation (TIRE). More information can be found in the 2020 preprint *Autoencoder-based Change Point Detection in Time Series Data using a Time-Invariant Representation*. 

The authors of this paper are:

- Tim De Ryck ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven; now [SAM](https://math.ethz.ch/sam), Dept. Mathematics, ETH Zürich)
- [Maarten De Vos](https://www.esat.kuleuven.be/stadius/person.php?id=203) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven and Dept. Development and Regeneration, KU Leuven)
- [Alexander Bertrand](https://www.esat.kuleuven.be/stadius/person.php?id=331) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven)

All authors are affiliated to LEUVEN.AI - KU Leuven institute for AI. Note that work based on TIRE should cite our paper: 

    (to be added)

## Abstract

*Change point detection (CPD) aims to locate abrupt property changes in time series data. Recent CPD methods demonstrated the potential of using deep learning techniques, but often lack the ability to identify changes in the frequency domain and suffer from a high false alarm rate. We extend autoencoder- based methodology with a novel loss function, through which the used autoencoders learn a partially time-invariant representation that is tailored for change point detection and also includes spectral information. The result is a flexible method that allows the user to indicate whether change points should be sought in the time domain, frequency domain or both. Detectable change points include abrupt changes in the slope, mean, variance, autocorrelation function and frequency spectrum. We demonstrate that our proposed method is consistently highly competitive or superior to baseline methods on diverse simulated and real-life benchmark data sets. Finally, we mitigate the issue of false detection alarms through the use of a postprocessing procedure that combines a matched filter and a newly proposed change point score. We show that this combination drastically improves the performance of our method as well as all baseline methods.*

## Goal

More concretely, the goal of TIRE is the following. Given raw time series data, TIRE returns for each time stamp of the time series a change point score. This score reflects the probability that there is a change point at (or near) the corresponding time stamp. Note that the absolute value of this change point score has no meaning. It is then common practice to discard the change point for which the change point score is below some user-defined treshold. For more information on how the change point scores are obtained we refer to our paper. 

Detectable change points include abrupt changes in: 
- Mean
- Slope
- Variance
- Autocorrelation
- Frequency spectrum

## Guidelines

First install all required packages, these can be found in the beginning of each file. We provided a Jupyter notebook `TIRE_example_notebook.ipynb` that demonstrates how the TIRE change point scores can be obtained from the raw time series data. In addition, the change points obtained by TIRE are compared in the notebook to the ground truth both visually and through the calculation of the AUC score. Alternatively, you can run `main.py` to obtain a txt-file containing the change point scores. 

## Contact

In case of comments or questions, please contact me at <tim.deryck@math.ethz.ch>. 
