import h5py
import torch
from torch.utils.data import Dataset


def downscale(image):
    """Downscale the image by a factor in all dimensions."""
    factor = 2
    if image.ndim == 2:
        return image[::factor, ::factor]
    elif image.ndim == 3:
        return image[::factor, ::factor, ::factor]
    else:
        raise ValueError("Image must be 2D or 3D.")
                
def compute_2pcf(images):
    """
    Compute the scaled 2-point correlation function (2PCF) for a batch of 2D or 3D images.

    Parameters:
    -----------
    images: torch.Tensor
        Input images tensor of shape (batch_size, ...), where '...' represents the spatial dimensions.

    Returns:
    --------
    pcf: torch.Tensor
        The scaled 2PCF snapshots, shape (num_features, batch_size).
    """
    batch_size = images.shape[0]
    num_elements = images[0].numel()  # Total number of pixels per image

    # Determine dimensions for FFT operations
    dims = tuple(range(1, images.dim()))

    # Compute FFT of images
    fft_images = torch.fft.fftn(images, dim=dims)

    # Compute power spectrum (autocorrelation in Fourier space)
    power_spectrum = fft_images.conj() * fft_images 

    # Compute autocorrelation (inverse FFT of power spectrum)
    autocorr = torch.fft.ifftn(power_spectrum, dim=dims).real / num_elements

    # Flatten the autocorrelation functions
    pcf = autocorr.view(batch_size, -1).T  # Shape: (num_pixels, batch_size)

    # Scale the snapshots with the volume fraction
    vol = torch.mean(pcf, dim=0).sqrt().unsqueeze(0)  # Shape: (1, batch_size)
    pcf = pcf - vol ** 2  # Zero mean
    pcf = pcf / (vol - vol**2) # Scale by the standard deviation
    return pcf

class H5ImageDataset3D(Dataset):
    def __init__(self, h5_path, group_name, transform=None, max_samples=None, dtype=torch.float32, start_index=0):
        self.h5_path = h5_path
        self.group_name = group_name
        self.transform = transform
        self.dtype = dtype

        self.h5_file = h5py.File(self.h5_path, 'r')
        self.dataset = self.h5_file[self.group_name]
        self.dataset_names = list(self.dataset.keys())[start_index:]
        if max_samples is not None:
            self.dataset_names = self.dataset_names[:max_samples]
        self.length = len(self.dataset_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dataset_name = self.dataset_names[idx]
        image = self.dataset[dataset_name]['image'][()]
        image = torch.from_numpy(image).type(self.dtype)
        if self.transform:
            image = self.transform(image)
        return image

    def __del__(self):
        self.h5_file.close()

class H5ImageDataset2D(Dataset):
    def __init__(self, h5_path, group_name, transform=None, max_samples=None, dtype=torch.float32):
        self.h5_path = h5_path
        self.group_name = group_name
        self.transform = transform
        self.dtype = dtype
        
        self.h5_file = h5py.File(self.h5_path, 'r')
        self.dataset = self.h5_file[f'{group_name}/image_data']
        self.length = min(len(self.dataset), max_samples) if max_samples else len(self.dataset)
        
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        image = self.dataset[idx]
        image = torch.from_numpy(image).type(self.dtype)
        if self.transform:
            image = self.transform(image)
        return image
    def __del__(self):
        self.h5_file.close()