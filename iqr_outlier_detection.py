"""
Created on Mon Nov 17 2025

@author: Coren Lahav
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple


def detect_outliers_iqr(
    data: Union[np.ndarray, pd.Series, list],
    k: float = 1.5
) -> Tuple[float, float]:
    """
    Calculate outlier bounds using the Interquartile Range (IQR) method.
    
    Parameters
    ----------
    data : array-like
        Input data as numpy array, pandas Series, or list. Must be 1-dimensional.
    k : float, optional (default=1.5)
        Multiplier for the IQR to define outlier boundaries. Common values are:
        - 1.5: Standard outlier detection (default)
        - 3.0: Detection of extreme outliers only
    
    Returns
    -------
    bounds : tuple of (float, float)
        Lower and upper bounds for outlier detection (lower_bound, upper_bound).
        Values below lower_bound or above upper_bound are considered outliers.
    
    Raises
    ------
    ValueError
        If input data is empty or contains only NaN values.
    
    Notes
    -----
    - NaN values are automatically excluded from quartile calculations.
    - The method uses numpy's percentile function with linear interpolation
      (default behavior) for quartile calculation.
    
    References
    ----------
    Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.
    
    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    >>> lower_bound, upper_bound = detect_outliers_iqr(data)
    >>> print(f"Outlier bounds: [{lower_bound:.1f}, {upper_bound:.1f}]")
    Outlier bounds: [-5.5, 16.5]
    
    >>> # Identify outliers using the bounds
    >>> outliers = (data < lower_bound) | (data > upper_bound)
    >>> print(f"Outlier values: {data[outliers]}")
    Outlier values: [100]
    """
    # Convert input to numpy array
    data_array = np.asarray(data)
    
    # Ensure 1-dimensional data
    if data_array.ndim != 1:
        raise ValueError(f"Input data must be 1-dimensional, got {data_array.ndim}D array")
    
    # Handle NaN values
    valid_data = data_array[~np.isnan(data_array)]
    
    # Check if there's any valid data
    if len(valid_data) == 0:
        raise ValueError("Input data contains no valid (non-NaN) values")
    
    # Calculate quartiles and IQR
    q1 = np.percentile(valid_data, 25)
    q3 = np.percentile(valid_data, 75)
    iqr = q3 - q1
    
    # Calculate outlier bounds
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    return lower_bound, upper_bound


def get_outlier_statistics(
    data: Union[np.ndarray, pd.Series, list],
    k: float = 1.5
) -> dict:
    """
    Calculate summary statistics for outlier detection using the IQR method.
    
    Parameters
    ----------
    data : array-like
        Input data as numpy array, pandas Series, or list.
    k : float, optional (default=1.5)
        Multiplier for the IQR.
    
    Returns
    -------
    stats : dict
        Dictionary containing:
        - 'n_total': Total number of observations
        - 'n_valid': Number of valid (non-NaN) observations
        - 'n_outliers': Number of detected outliers
        - 'outlier_percentage': Percentage of outliers among valid observations
        - 'q1': First quartile
        - 'q3': Third quartile
        - 'iqr': Interquartile range
        - 'lower_bound': Lower outlier threshold
        - 'upper_bound': Upper outlier threshold
        - 'outlier_indices': Array of indices where outliers occur
        - 'outlier_values': Array of outlier values
    
    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    >>> stats = get_outlier_statistics(data)
    >>> print(f"Detected {stats['n_outliers']} outliers ({stats['outlier_percentage']:.1f}%)")
    Detected 1 outliers (10.0%)
    """
    # Convert to array
    data_array = np.asarray(data)
    
    # Get outlier bounds
    lower_bound, upper_bound = detect_outliers_iqr(data_array, k=k)
    
    # Calculate statistics
    valid_mask = ~np.isnan(data_array)
    valid_data = data_array[valid_mask]
    
    q1 = np.percentile(valid_data, 25)
    q3 = np.percentile(valid_data, 75)
    iqr = q3 - q1
    
    # Identify outliers using the bounds
    outlier_mask = (data_array < lower_bound) | (data_array > upper_bound)
    outlier_mask[~valid_mask] = False  # Don't count NaN as outliers
    
    n_outliers = np.sum(outlier_mask)
    n_valid = len(valid_data)
    outlier_percentage = (n_outliers / n_valid * 100) if n_valid > 0 else 0
    
    outlier_indices = np.where(outlier_mask)[0]
    outlier_values = data_array[outlier_mask]
    
    return {
        'n_total': len(data_array),
        'n_valid': n_valid,
        'n_outliers': n_outliers,
        'outlier_percentage': outlier_percentage,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outlier_indices': outlier_indices,
        'outlier_values': outlier_values
    }


if __name__ == "__main__":
    # Example usage and demonstration
    print("IQR Outlier Detection - Example Usage\n" + "=" * 50)
    
    # Example 1: Simple outlier detection
    print("\nExample 1: Basic outlier detection")
    data1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    lower, upper = detect_outliers_iqr(data1)
    outliers1 = (data1 < lower) | (data1 > upper)
    print(f"Data: {data1}")
    print(f"Bounds: [{lower:.1f}, {upper:.1f}]")
    print(f"Outliers detected: {data1[outliers1]}")
    print(f"Outlier positions: {np.where(outliers1)[0]}")
    
    # Example 2: With statistics
    print("\nExample 2: Outlier detection with statistics")
    data2 = np.array([10, 12, 13, 14, 15, 16, 17, 18, 19, 50, 55])
    stats = get_outlier_statistics(data2)
    print(f"Data: {data2}")
    print(f"Number of outliers: {stats['n_outliers']} ({stats['outlier_percentage']:.1f}%)")
    print(f"Q1: {stats['q1']:.2f}, Q3: {stats['q3']:.2f}, IQR: {stats['iqr']:.2f}")
    print(f"Bounds: [{stats['lower_bound']:.2f}, {stats['upper_bound']:.2f}]")
    print(f"Outlier values: {stats['outlier_values']}")
    
    # Example 3: Data with NaN values
    print("\nExample 3: Handling NaN values")
    data3 = np.array([1, 2, 3, np.nan, 5, 6, 7, 8, 9, 100, np.nan])
    lower3, upper3 = detect_outliers_iqr(data3)
    outliers3 = (data3 < lower3) | (data3 > upper3)
    print(f"Data: {data3}")
    print(f"Bounds: [{lower3:.1f}, {upper3:.1f}]")
    print(f"Outliers detected at indices: {np.where(outliers3)[0]}")
    print(f"Outlier values: {data3[outliers3]}")
    
    # Example 4: Different k values
    print("\nExample 4: Comparing different k values")
    data4 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 25])
    for k_val in [1.5, 2.0, 3.0]:
        lower_k, upper_k = detect_outliers_iqr(data4, k=k_val)
        outliers_k = (data4 < lower_k) | (data4 > upper_k)
        n_outliers = np.sum(outliers_k)
        print(f"k={k_val}: {n_outliers} outliers detected - {data4[outliers_k]}")
