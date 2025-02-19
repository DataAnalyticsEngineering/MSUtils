import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import numpy as np
import torch
from torch.utils.data import DataLoader

from MSUtils.ReducedBasis.data import H5ImageDataset3D, compute_2pcf, downscale
from MSUtils.ReducedBasis.ReducedBasis import ReducedBasis


def main():
    # Parameters for 3D microstructure data
    h5_path = 'data/3d_microstructures.h5'
    train_group = 'structures_train'
    test_group = 'structures_val'
    n_orig = 100          # Number of snapshots for initial basis computation
    n_adjust = 50        # Number of snapshots for each incremental update
    total_samples_train = 63000 # Total number of training samples
    total_samples_test = 13500   # Total number of test samples
    batch_size = 50
    
    device = torch.device('cpu')
    dtype = torch.float64
    transform = downscale
    projection_error_threshold = 0.025
    truncation_limit = 0.05
    low_rank = None
    
    # # Parameters for 2D microstructure data
    # h5_path = 'data/FNOCG_2D.h5'
    # train_group = 'train_set'
    # test_group = 'benchmark_set'
    # n_orig = 50          # Number of snapshots for initial basis computation
    # n_adjust = 50        # Number of snapshots for each incremental update
    # total_samples_train = 30000 # Total number of training samples
    # total_samples_test = 1500   # Total number of test samples
    # batch_size = 50
    
    # device = torch.device('cuda')
    # dtype = torch.float64
    # transform = None
    # projection_error_threshold = 0.005
    # truncation_limit = 0.01
    # low_rank = None
    
    # Load datasets
    train_dataset = H5ImageDataset3D(
        h5_path, train_group, transform=transform, max_samples=total_samples_train, dtype=dtype
    )
    test_dataset = H5ImageDataset3D(
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
    for loader in [test_loader, train_loader]:
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
