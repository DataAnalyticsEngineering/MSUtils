import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from MSUtils.ReducedBasis.ReducedBasis import ReducedBasis


def downscale(image):
    """Downscale the image by a factor in all dimensions."""
    factor = 2
    if image.ndim == 2:
        return image[::factor, ::factor]
    elif image.ndim == 3:
        return image[::factor, ::factor, ::factor]
    else:
        raise ValueError("Image must be 2D or 3D.")

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


def main():
    # # Parameters for 3D microstructure data
    # h5_path = 'data/3d_microstructures.h5'
    # train_group = 'structures_train'
    # test_group = 'structures_test'
    # n_orig = 100          # Number of snapshots for initial basis computation
    # n_adjust = 50        # Number of snapshots for each incremental update
    # total_samples_train = 20000 # Total number of training samples
    # total_samples_test = 1000   # Total number of test samples
    # batch_size = 50
    
    # device = torch.device('cpu')
    # dtype = torch.float64
    # transform = downscale
    # projection_error_threshold = 0.005
    # truncation_limit = 0.01
    # low_rank = None
    
    # Parameters for 2D microstructure data
    h5_path = 'data/FNOCG_2D.h5'
    train_group = 'train_set'
    test_group = 'benchmark_set'
    n_orig = 200          # Number of snapshots for initial basis computation
    n_adjust = 100        # Number of snapshots for each incremental update
    total_samples_train = 30000 # Total number of training samples
    total_samples_test = 1500   # Total number of test samples
    batch_size = 100
    
    device = torch.device('cuda')
    dtype = torch.float64
    transform = None
    projection_error_threshold = 0.005
    truncation_limit = 0.01
    low_rank = None
    
    # Load datasets
    train_dataset = H5ImageDataset2D(
        h5_path, train_group, transform=transform, max_samples=total_samples_train, dtype=dtype
    )
    test_dataset = H5ImageDataset2D(
        h5_path, test_group, transform=transform, max_samples=total_samples_test, dtype=dtype
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the ReducedBasis object
    rb = ReducedBasis(truncation_limit=truncation_limit, device=device)

    # Initialize variables
    total_snapshots_processed = 0
    used_snapshot_indices = []
    initial_snapshots = []
    adjust_snapshots = []
    basis_computed = False  
    
    # Process training data
    for loader in [train_loader, test_loader]:
        for images in loader:
            pcf = compute_2pcf(images).to(device)
            batch_size_actual = images.size(0)
            snapshot_indices = np.arange(total_snapshots_processed, total_snapshots_processed + batch_size_actual)
            total_snapshots_processed += batch_size_actual

            if not basis_computed:
                # Collect snapshots for initial basis computation
                initial_snapshots.append(pcf)
                used_snapshot_indices.extend(snapshot_indices.tolist())

                if total_snapshots_processed >= n_orig:
                    # Compute the initial basis when we have collected at least n_orig snapshots
                    S = torch.cat(initial_snapshots, dim=1)
                    rb.compute_basis(S, low_rank)
                    print(f"Initial basis computed with {rb.B.shape[1]} modes.")
                    print("Reduced basis shape:", rb.B.shape)
                    initial_snapshots.clear()
                    basis_computed = True
            else:
                # Basis has been computed, proceed to compute projection errors and adjust the basis
                _, relative_errors = rb.projection_error(pcf)
                mask = (relative_errors > projection_error_threshold).cpu().numpy()
                if np.any(mask):
                    # Select snapshots exceeding the projection error threshold
                    selected_pcf = pcf[:, mask]
                    adjust_snapshots.append(selected_pcf)
                    selected_indices = snapshot_indices[mask]
                    used_snapshot_indices.extend(selected_indices.tolist())

                    # Adjust the basis when we have accumulated at least n_adjust snapshots
                    total_adjust_snapshots = sum(snap.shape[1] for snap in adjust_snapshots)
                    if total_adjust_snapshots >= n_adjust:
                        delta_S = torch.cat(adjust_snapshots, dim=1)
                        rb.adjust_basis(delta_S)
                        print(f"Basis adjusted. Total modes: {rb.B.shape[1]}")
                        adjust_snapshots.clear()


    # Final basis adjustment with remaining snapshots
    if adjust_snapshots:
        delta_S = torch.cat(adjust_snapshots, dim=1)
        rb.adjust_basis(delta_S)
        print(f"Basis adjusted. Total modes: {rb.B.shape[1]}")

    rb.save('data/reduced_basis.h5', group_name='test_1', compression_level=9)
    
    def evaluate_projection_errors(loader, dataset_type):
        relative_errors_all = []
        num_samples = 0

        with torch.no_grad():
            for images in loader:
                pcf = compute_2pcf(images).to(device)
                errors, relative_errors = rb.projection_error(pcf)
                relative_errors_all.append(relative_errors)
                num_samples += relative_errors.numel()

        relative_errors_all = torch.cat(relative_errors_all) * 100  # Convert to percentage
        mean_relative_error = torch.mean(relative_errors_all).item()
        max_relative_error = torch.max(relative_errors_all).item()
        min_relative_error = torch.min(relative_errors_all).item()
        print(f'{dataset_type} Mean relative projection error: {mean_relative_error}%')
        print(f'{dataset_type} Max relative projection error: {max_relative_error}%')
        print(f'{dataset_type} Min relative projection error: {min_relative_error}%')
        
        # Count the number of snapshots with projection error greater than the threshold
        num_errors_above_threshold = (relative_errors_all > (projection_error_threshold * 100)).sum().item()
        print(f'Number of {dataset_type} snapshots with projection error greater than {projection_error_threshold * 100}%: {num_errors_above_threshold} out of {num_samples}')
    
    evaluate_projection_errors(test_loader, "Test")
    print(f'Total snapshots used in reduced basis: {len(used_snapshot_indices)}')
    print("Shape of reduced basis:", rb.B.shape)

if __name__ == '__main__':
    main()
