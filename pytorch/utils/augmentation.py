# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:30:26 2023

@author: benjaminstoeckigt
"""

from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
def apply_scaling(X, doy, sigma=0.1, sigma_change_freq=365):
    """
    Apply Gaussian scaling to each band of the time series data, with scale changing every specified frequency of days (default 365).

    Args:
    X : torch.Tensor
        Multi-band time series data where each row corresponds to a different day.
    doy : numpy.array
        Array containing the day of the year for each observation in X.
    sigma : float, optional
        Standard deviation of the normal distribution used for scaling.
    sigma_change_freq : int, optional
        Frequency in days at which the scaling factor changes.

    Returns:
    torch.Tensor
        Scaled multi-band time series data.
    """
    # Determine how many years are covered in the data
    years = int(torch.max(doy).item()) // sigma_change_freq + 1
    # Generate a unique scale factor for each year
    alphas = torch.normal(mean=1.0, std=sigma, size=(years,))

    # Apply the appropriate scale factor for each year
    scaled_X = torch.zeros_like(X)
    for year in range(years):
        year_mask = (doy >= year * sigma_change_freq) & (doy < (year + 1) * sigma_change_freq)
        scaled_X[year_mask] = alphas[year] * X[year_mask]

    return scaled_X

def time_warp(X, sigma=0.001):
    """Apply the same time warping to each band of a multi-band time series array using a cubic spline."""
    length = X.shape[0]
    original_indices = np.linspace(0, length - 1, num=length)
    num_knots = np.random.randint(3, 6)  # Randomly choosing number of knots for variability
    knot_positions = np.linspace(0, length - 1, num=num_knots)
    knot_values = knot_positions + np.random.normal(loc=0, scale=sigma * length, size=knot_positions.shape)
    knot_values = np.clip(knot_values, 0, length - 1)  # Ensuring knots don't go out of sequence bounds
    spline = CubicSpline(knot_positions, knot_values, bc_type='clamped')
    warped_indices = spline(original_indices)

    if X.ndim > 1 and X.shape[1] > 1:  # Check if X is multi-dimensional and has multiple bands
        warped_X = torch.zeros_like(X)
        for i in range(X.shape[1]):  # Iterate over each band
            warped_X[:, i] = torch.from_numpy(
                np.interp(warped_indices, original_indices, X[:, i].numpy())
            )
        return warped_X
    else:
        return torch.from_numpy(np.interp(warped_indices, original_indices, X.numpy()))


def year_shifting(doy, shift_range=16):
    """
    Shifts the day of year (DOY) randomly within a specified range for each year.

    Args:
    doy : torch.Tensor
        Tensor containing the day of the year for each observation.
    shift_range : int
        Maximum range of days for shifting, both positive and negative.

    Returns:
    torch.Tensor
        Adjusted DOY values that stay within the original boundaries.
    """
    min_doy = torch.min(doy)
    max_doy = torch.max(doy)

    # Calculate the minimum and maximum day of year values for each year
    years = (doy - 1) // 365
    unique_years = torch.unique(years)
    min_doy_per_year = unique_years * 365 + 1
    max_doy_per_year = (unique_years + 1) * 365

    # Generate random shifts within the specified range for each year
    shifts = torch.randint(-shift_range, shift_range + 1, size=(len(unique_years),), dtype=torch.int32)

    # Apply shifts to each year and clamp within the min and max DOY for each year
    shifted_doy = torch.zeros_like(doy)
    for year, min_doy_year, max_doy_year, shift in zip(unique_years, min_doy_per_year, max_doy_per_year, shifts):
        year_mask = years == year
        shifted_doy[year_mask] = torch.clamp(doy[year_mask] + shift, min_doy_year, max_doy_year)

    # Clamp again to ensure overall min and max DOY are respected
    shifted_doy = torch.clamp(shifted_doy, min_doy, max_doy)

    return shifted_doy



def day_shifting(doy, shift_range=16):
    """
    Shifts the day of year (DOY) randomly within a specified range, skipping indices specified by a mask.

    Args:
    doy : torch.Tensor
        Tensor containing the day of the year for each observation.
    shift_range : int
        Maximum range of days for shifting, both positive and negative.
    skip_mask : torch.Tensor (boolean mask), optional
        Indices to skip when applying shifts.

    Returns:
    torch.Tensor
        Adjusted DOY values that stay within the original boundaries and maintain the same tensor length.
    """
    min_doy = torch.min(doy)
    max_doy = torch.max(doy)
    shifts = torch.randint(-shift_range, shift_range + 1, size=doy.shape, dtype=torch.int32)

    shifted_doy = torch.clamp(doy + shifts, min_doy, max_doy)

    return shifted_doy

def plot(X, X_aug, doy, doy_aug, method, band):
    # Modified to handle plotting of DOY adjustments
    plt.figure(figsize=(10, 5))
    plt.plot(doy.numpy(), X.numpy()[:, band-1], 'o-', label='Original X', marker='o')
    plt.plot(doy_aug.numpy(), X_aug.numpy()[:, band-1], 'o', linestyle='', label=f'Augmented X ({method}); Band {band}', marker='x')
    plt.xlabel('Day of Year (DOY)')
    plt.ylabel('Values')
    plt.title('Comparison of Original and Augmented Data')
    plt.legend()
    plt.grid(True)
    plt.show()
    time.sleep(1000)  # Note: Consider removing or decreasing this sleep time in production

def zero_out_data(X, doy, percentage=5):
    """
    Sets a random percentage of X values and the corresponding doy values to zero.

    Args:
    X : torch.Tensor
        Multi-band time series data.
    doy : torch.Tensor
        Tensor containing the day of the year for each observation.
    percentage : int, optional
        Percentage of values to zero out.

    Returns:
    torch.Tensor, torch.Tensor, torch.Tensor
        Zeroed-out data, day of year values, and a mask of zeroed indices.
    """
    total_count = X.size(0)
    zero_count = int(total_count * (percentage / 100.0))

    # Random indices to be zeroed out
    zero_indices = torch.randperm(total_count)[:zero_count]

    X_zeroed = X.clone()
    doy_zeroed = doy.clone()

    # Set selected indices to zero
    X_zeroed[zero_indices, :] = 0
    doy_zeroed[zero_indices] = 0

    return X_zeroed, doy_zeroed

def apply_augmentation(X, doy, p, plotting):
    """
    Applies augmentation based on a random choice among:
    1. A single augmentation,
    2. Any two of them combined randomly but with controlled sequence,
    3. All three applied with controlled sequence.

    Args:
    X : torch.Tensor
        Multi-band time series data.
    doy : torch.Tensor
        Tensor containing the day of the year for each observation.
    p : float
        Probability of applying an augmentation.
    plotting : int or None
        Band to plot if plotting is enabled.

    Returns:
    tuple of (torch.Tensor, torch.Tensor)
        Augmented data and possibly adjusted DOY.
    """
    doy_aug = doy.clone()  # Start with a clone to not alter the original DOY
    X_aug = X.clone()  # Start with a clone to not alter the original X

    if torch.rand(1).item() < p:
        # Choose augmentation pattern with equal probability
        augmentation_patterns = ['single', 'double', 'triple']
        selected_pattern = np.random.choice(augmentation_patterns)

        if selected_pattern == 'single':
            # Apply one of the augmentations chosen randomly
            aug_type = np.random.choice(['scaling', 'day shifting', 'zero out'])
            if aug_type == 'scaling':
                X_aug = apply_scaling(X_aug, doy_aug, sigma=0.15)
                method = 'scaling'
            elif aug_type == 'day shifting':
                doy_aug = year_shifting(doy_aug, shift_range=16)
                method = 'day shifting'
            else:
                percentage_to_zero = np.random.randint(5, 71)
                X_aug, doy_aug = zero_out_data(X_aug, doy_aug, percentage=percentage_to_zero, )
                method = f'zero out {percentage_to_zero}%'

        elif selected_pattern == 'double':
            # Randomly select two augmentations and sort to ensure zero out happens last
            aug_types = np.random.choice(['scaling', 'day shifting', 'zero out'], size=2, replace=False)
            aug_types = sorted(aug_types, key=lambda x: x == 'zero out')  # This ensures zero out is applied last
            methods = []
            for aug in aug_types:
                if aug == 'scaling':
                    X_aug = apply_scaling(X_aug, doy_aug, sigma=0.15)
                    methods.append('scaling')
                elif aug == 'day shifting':
                    doy_aug = year_shifting(doy_aug, shift_range=16)
                    methods.append('day shifting')
                else:
                    percentage_to_zero = np.random.randint(5, 71)
                    X_aug, doy_aug = zero_out_data(X_aug, doy_aug, percentage=percentage_to_zero)
                    methods.append(f'zero out {percentage_to_zero}%')
            method = ' & '.join(methods)

        else:
            # Apply all three augmentations, ensuring zero out happens first
            percentage_to_zero = np.random.randint(5, 71)
            X_aug = apply_scaling(X_aug, doy_aug, sigma=0.15)
            doy_aug = year_shifting(doy_aug, shift_range=16)
            X_aug, doy_aug = zero_out_data(X_aug, doy_aug, percentage=percentage_to_zero)
            method = 'scaling & day shifting & zero out'

    else:
        method = 'none'  # No augmentation applied

    if plotting is not None:
        plot(X, X_aug, doy, doy_aug, method, band=plotting)

    return X_aug, doy_aug