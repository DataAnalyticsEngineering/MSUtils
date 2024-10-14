import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftn, ifftn
from scipy.optimize import curve_fit


def exponential_decay(x, l):
    return np.exp(-x / l)


def compute_correlation_length(img):
    """
    Compute the correlation length of an image using FFT and exponential decay fitting.

    Parameters:
    - img: The input image as a NumPy array.

    Returns:
    - cl: Correlation lengths for each dimension.
    - aspect_ratio: Aspect ratio derived from correlation lengths.
    - autocorr: The autocorrelation function of the image.
    """
    # Compute mean and variance for normalization
    c1 = np.mean(img)
    c2 = np.mean(img**2)

    # Perform FFT, get power spectrum, and compute autocorrelation
    f_img = fftn(img)
    power_spectrum = np.abs(f_img) ** 2
    autocorr = ifftn(power_spectrum).real / np.prod(img.shape)
    autocorr = (autocorr - c1**2) / (c2 - c1**2)

    # Initialize correlation lengths array
    cl = np.zeros(img.ndim)

    # Fit exponential decay to autocorrelation function slices
    for i_dim in range(img.ndim):
        slice_midpoint = slice(None, img.shape[i_dim] // 2)
        slice_others = [0 if dim != i_dim else slice_midpoint for dim in range(img.ndim)]
        y = autocorr[tuple(slice_others)].ravel()
        x = np.arange(y.size)
        params, _ = curve_fit(exponential_decay, x, y, p0=[10], bounds=(0, np.inf))
        cl[i_dim] = params[0]

    # Normalize correlation lengths and calculate aspect ratio
    normalized_cl = cl / np.min(cl)
    aspect_ratio = np.floor(normalized_cl).astype(int)

    return aspect_ratio, cl, autocorr


def plot_mid_planes(autocorr):
    """
    Visualize the mid-planes (XY, YZ, XZ) of the autocorrelation function for 3D data.
    For 2D data, just visualize the data itself.

    Parameters:
    - autocorr: The autocorrelation function as a NumPy array.
    """
    autocorr_shifted = np.fft.fftshift(autocorr)

    if autocorr.ndim == 3:
        nz, ny, nx = autocorr_shifted.shape
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
        planes = [
            autocorr_shifted[nz // 2, :, :],
            autocorr_shifted[:, ny // 2, :],
            autocorr_shifted[:, :, nx // 2],
        ]
        titles = ["XY Mid-Plane", "YZ Mid-Plane", "XZ Mid-Plane"]

        for ax, plane, title in zip(axes, planes, titles, strict=False):
            im = ax.imshow(plane, cmap="viridis")
            ax.set_title(title, fontsize=14)
            ax.axis("off")

        # Colorbar configuration
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
        fig.colorbar(im, cax=cbar_ax, orientation="horizontal")

    elif autocorr.ndim == 2:
        plt.figure(figsize=(6, 6), dpi=300)
        plt.imshow(autocorr_shifted, cmap="viridis")
        plt.title("2D Autocorrelation", fontsize=14)
        plt.axis("off")
        plt.colorbar(orientation="horizontal")

    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.2, wspace=0.3, hspace=0.3)
    plt.savefig("data/mid_planes.png", bbox_inches="tight")
    plt.show()


def visualize_correlation(cl, autocorr):
    """
    Visualize the autocorrelation function and exponential fits for each dimension.
    Handles both 2D and 3D autocorrelation data and plots only half of the autocorrelation line in each direction.

    Parameters:
    - cl: Correlation lengths for each dimension.
    - autocorr: The autocorrelation function as a NumPy array.
    """
    plot_mid_planes(autocorr)  # Handles both 2D and 3D

    # Ensure the autocorrelation function is centered
    autocorr_centered = np.fft.fftshift(autocorr)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    if autocorr.ndim == 3:
        directions = ["X", "Y", "Z"]  # The three directions for 3D data
        # Indices for slicing from the center to halfway in each direction
        indices = [
            (slice(None), autocorr.shape[1] // 2, autocorr.shape[2] // 2),
            (autocorr.shape[0] // 2, slice(None), autocorr.shape[2] // 2),
            (autocorr.shape[0] // 2, autocorr.shape[1] // 2, slice(None)),
        ]
    elif autocorr.ndim == 2:
        directions = ["X", "Y"]  # The two directions for 2D data
        indices = [
            (slice(None), autocorr.shape[1] // 2),
            (autocorr.shape[0] // 2, slice(None)),
        ]

    colors = cm.viridis(np.linspace(0, 1, len(directions)))
    markers = ["o", "s", "^", "d"][: len(directions)]

    for i_dim, direction, color, marker in zip(
        range(len(directions)), directions, colors, markers, strict=False
    ):
        line_full = autocorr_centered[indices[i_dim]].ravel()
        mid_point = len(line_full) // 2  # Find the center
        line = line_full[mid_point:]  # Take only the second half from the center
        x = np.arange(len(line))
        ax.plot(
            x,
            line,
            label=f"{direction} Autocorrelation",
            color=color,
            linestyle="-",
            marker=marker,
            markersize=5,
        )
        ax.plot(
            x,
            exponential_decay(x, cl[i_dim]),
            label=f"{direction} Exponential Fit",
            color=color,
            linestyle="--",
            linewidth=2,
        )

    ax.set_title("Autocorrelation and Exponential Fits", fontsize=24)
    ax.set_xlabel("Voxel Separation Distance", fontsize=24)
    ax.set_ylabel("Autocorrelation", fontsize=24)
    ax.legend(fontsize=20)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("data/correlation_lengths.png", bbox_inches="tight")
    plt.show()


def main():
    from MicrostructureImage import MicrostructureImage

    ms = MicrostructureImage(h5_filename="data/fibers1.h5", dset_name="/img")
    aspect_ratio, cl, autocorr = compute_correlation_length(ms.image)
    print(f"Correlation Lengths: {cl}")
    print(f"Aspect Ratio: {aspect_ratio}")

    visualize_correlation(cl, autocorr)


if __name__ == "__main__":
    main()
