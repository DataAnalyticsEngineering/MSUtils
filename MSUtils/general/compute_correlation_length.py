import numpy as np
from scipy.fft import fftn, ifftn
from scipy.optimize import curve_fit


def exponential_decay(x, l):
    return np.exp(-x / l)


def compute_correlation_length(img):
    """Return the aspect ratio, correlation lengths, and normalized autocorrelation."""
    mean = np.mean(img)
    spectrum = fftn(img)
    autocorr = ifftn(np.abs(spectrum) ** 2).real / img.size
    autocorr = (autocorr - mean**2) / (np.mean(img**2) - mean**2)

    correlation_lengths = np.empty(img.ndim)
    for axis, size in enumerate(img.shape):
        index = [0] * img.ndim
        index[axis] = slice(size // 2)
        values = autocorr[tuple(index)]
        correlation_lengths[axis] = curve_fit(
            exponential_decay,
            np.arange(values.size),
            values,
            p0=[10],
            bounds=(0, np.inf),
        )[0][0]

    aspect_ratio = np.floor(correlation_lengths / correlation_lengths.min()).astype(int)
    return aspect_ratio, correlation_lengths, autocorr
