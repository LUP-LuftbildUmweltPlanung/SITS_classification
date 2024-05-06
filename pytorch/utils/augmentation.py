from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
def apply_scaling(X, doy, sigma=0.2, sigma_change_freq=365):
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

def time_warp(X, sigma=0.05):
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


def plot(X, X_aug, doy, method, band):
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(doy, X.numpy()[:,band-1], label='Original X', marker='o')
    plt.plot(doy, X_aug.numpy()[:,band-1], label=f'Augmented X ({method}); Band {band}', marker='x')
    plt.xlabel('Day of Year (DOY)')
    plt.ylabel('Values')
    plt.title('Comparison of Original and Augmented Data')
    plt.legend()
    plt.grid(True)
    plt.show()
    time.sleep(1000)

def apply_augmentation(X, doy, p, plotting):
    if torch.rand(1).item() < p:
        if torch.rand(1).item() < 0.5:
            X_aug = apply_scaling(X,doy)
            method = 'scaling'
        else:
            X_aug = time_warp(X)
            method = 'time_warp'
    else:
        X_aug = X
        method = 'none'

    if plotting != None:
        plot(X, X_aug, doy, method, band=plotting)

    return X_aug, doy