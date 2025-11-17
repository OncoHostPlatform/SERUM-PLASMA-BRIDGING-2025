"""
Created on Tue May 9 2023

@author: Coren Lahav
"""


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pytest


class IterativeScaler:
    '''Standardize features by removing the mean and scaling to unit variance
    iteratively until no further outliers are discovered.
    
    On each iteration, the standard score of a sample `x` is calculated as:
     
          z = (x - u) / s
          
    where `u` is the mean of the training samples, and `s` is the standard
    deviation of the training samples.
   
    On each iteration outlier values (defined by z_threshold standard
    deviations) are set to nan. The process is repeated until no outliers are
    discovered.
    If impute_threshold==True, outlier values are imputed with +z_threshold or
    -z_threshold, depending on outlier direction.
    
    Usage example:
        # Dataset parameters
        SAMPLE_SIZE = 200
        VAL_SET_SIZE = 50
        
        # Generate random dataset
        np.random.seed(42)
        dat = (np.random.randn(SAMPLE_SIZE,3) + 3) * 4
        
        # Add outlier and nan values
        dat[5,1] = 50; dat[90,1] = 120; dat[91,2] = -50       # dev set outliers
        dat[92,1] = np.nan; dat[186,0] = np.nan               # nan values that are not outliers will not be imputed
        dat[179,1] = 1000; dat[180,1] = 500; dat[185,2] = -60 # validation set outliers
        
        # Convert dataset to dataframe format
        dat = pd.DataFrame(dat, index=[f'Sample{str(row+1)}' for row in range(dat.shape[0])],
                                columns=[f'Prot{str(col+1)}' for col in range(dat.shape[1])])

        # Add categorical column (not for scaling)
        dat['Sex'] = np.random.random_sample(SAMPLE_SIZE)
        dat['Sex'] = (dat['Sex'] >= 0.5).map({False: 'Male', True: 'Female'}).astype(bool)
        
        # Divide dataset into dev and validation sets
        dat_dev = dat.iloc[:-VAL_SET_SIZE]
        dat_val = dat.iloc[-VAL_SET_SIZE:]
        
        # Run iterative scaler on dev set
        iscaler = IterativeScaler(z_threshold=4, impute_threshold=True, verbose=True)
        dat_dev_scaled = iscaler.fit_transform(dat_dev, scale_cols=['Prot1', 'Prot2', 'Prot3'])
        
        # Transform an independent set of samples based on dev set distribution parameters
        dat_val_scaled = iscaler.transform(dat_val)
    '''
    
    def __init__(self, z_threshold, impute_threshold=True, verbose=False):
        """
        Initialize the IterativeScaler.

        Parameters
        ----------
        z_threshold : float
            The z-score threshold for identifying outliers.
        impute_threshold : bool, optional
            Whether to impute outlier values with +/- z_threshold. Default is True.
        verbose : bool, optional
            If True, print progress messages during scaling (default is False).
        """
        # Initialize parameters
        self.z_threshold = z_threshold
        self.impute_threshold = impute_threshold
        self.verbose = verbose
        
        # Initialize list to hold scalers for each iteration
        self.scalers = []
        
        # Initialize array to hold names of columns to be scaled
        self.scale_cols = np.array([])
        
        
    def fit_transform(self, fit_df, scale_cols=None):
        """
        Fits the scaler to the provided DataFrame and transforms it by iteratively
        scaling and removing outliers.

        Args:
            fit_df (pd.DataFrame): The input DataFrame to fit and transform.
            scale_cols (list or None): List of column names to scale. If None, all columns are used.

        Returns:
            pd.DataFrame: The scaled DataFrame with outliers handled according to the z_threshold and impute_threshold.
        """
        # Print dataset size if verbose
        if self.verbose:
            print(f'Fit-transforming dataset of size {fit_df.shape}.')
        # Determine columns to scale
        if scale_cols is None:
            self.scale_cols = np.array(fit_df.columns)
        else:
            self.scale_cols = np.array(scale_cols)
        
        # Initial scaling
        self.scalers.append(StandardScaler())
        zscored = self.scalers[0].fit_transform(fit_df[self.scale_cols].values)
        
        # Copy of z-scored data for outlier exclusion
        z_masked = zscored.copy()
        
        # Outlier exclusion loop
        rnd = 0; n_excluded = 1; n_excluded_overall = 0
        while n_excluded > 0:
            n_excluded = np.sum(np.abs(z_masked) > self.z_threshold)
            n_excluded_overall += n_excluded
            
            z_masked[np.abs(z_masked) > self.z_threshold] = np.nan
            if self.verbose:
                if n_excluded == 0:
                    print(f'Round {rnd+1}: no values excluded.')
                else:    
                    print(f'Round {rnd+1} excluded: {n_excluded} values.')
            if n_excluded > 0:
                rnd += 1
                self.scalers.append(StandardScaler())
                z_masked = self.scalers[rnd].fit_transform(z_masked)
            
        # Outlier imputation if required
        if n_excluded_overall > 0 and self.impute_threshold == True:
            zscored[~np.isnan(z_masked)]=z_masked[~np.isnan(z_masked)] # Update zscored with final scaled values to ensure correct imputation signs
            z_imputed = z_masked.copy()  # Create output array
            pos_imputations = np.logical_and(np.isnan(z_masked), zscored > 0)
            neg_imputations = np.logical_and(np.isnan(z_masked), zscored < 0)
            if self.verbose:
                print(f'Imputing {np.sum(pos_imputations)} values with {self.z_threshold:.2f}')
                print(f'Imputing {np.sum(neg_imputations)} values with -{self.z_threshold:.2f}')
            
            z_imputed[pos_imputations] = self.z_threshold
            z_imputed[neg_imputations] = -self.z_threshold
        else:
            z_imputed = z_masked # No imputation needed
        iscaled_df = pd.DataFrame(z_imputed, index=fit_df.index, columns=self.scale_cols)

        return iscaled_df
    
    
    def transform(self, transform_df):
        """
        Transform the input DataFrame using the fitted scalers, applying iterative outlier exclusion. Optionally, perform imputation as determined during fitting.

        Parameters
        ----------
        transform_df : pandas.DataFrame
            The DataFrame to be transformed. Must contain the columns specified in `scale_cols`.

        Returns
        -------
        pandas.DataFrame
            The transformed DataFrame with outliers handled and values scaled according to the fitted parameters.
        """
        if self.verbose:
            print(f'Transforming dataset of size {transform_df.shape}.')
        
        # Outlier exclusion loop
        n_excluded_overall = 0
        for rnd, scaler in enumerate(self.scalers):
            if rnd == 0:
                zscored = self.scalers[rnd].transform(transform_df[self.scale_cols].values)
                z_masked = zscored.copy()
            else:
                z_masked = self.scalers[rnd].transform(z_masked)
            n_excluded = np.sum(np.abs(z_masked) > self.z_threshold)
            n_excluded_overall += n_excluded
            if self.verbose:
                print(f'Round {rnd+1} excluded: {n_excluded} values.')
            z_masked[np.abs(z_masked) > self.z_threshold] = np.nan
        
        # Outlier imputation if required
        if n_excluded_overall > 0 and self.impute_threshold == True:
            zscored[~np.isnan(z_masked)]=z_masked[~np.isnan(z_masked)] # Important for extreme cases where a value changes sign after imputation
            z_imputed = z_masked.copy()  # Create output array
            pos_imputations = np.logical_and(np.isnan(z_masked), zscored > 0)
            neg_imputations = np.logical_and(np.isnan(z_masked), zscored < 0)
            if self.verbose:
                print(f'Imputing {np.sum(pos_imputations)} values with {self.z_threshold:.2f}')
                print(f'Imputing {np.sum(neg_imputations)} values with -{self.z_threshold:.2f}')
            z_imputed[pos_imputations] = self.z_threshold
            z_imputed[neg_imputations] = -self.z_threshold
        else:
            z_imputed = z_masked # No imputation needed
        iscaled_df = pd.DataFrame(z_imputed, index=transform_df.index, columns=self.scale_cols)
        return iscaled_df