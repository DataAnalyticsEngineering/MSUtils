import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from MSUtils.ReducedBasis.ReducedBasis import ReducedBasis
from MSUtils.ReducedBasis.train import H5ImageDataset2D, compute_2pcf


def main():
    # Parameters for 2D microstructure data
    h5_path = 'data/FNOCG_2D.h5'
    train_group_name = 'train_set'
    test_group_name = 'benchmark_set'
    total_samples_train = 30000  # Total number of training samples
    total_samples_test = 1500    # Total number of test samples
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    transform = None
    
    # Load datasets
    batch_size = 250
    train_dataset = H5ImageDataset2D(
        h5_path, train_group_name, transform=transform, max_samples=total_samples_train, dtype=dtype
    )
    test_dataset = H5ImageDataset2D(
        h5_path, test_group_name, transform=transform, max_samples=total_samples_test, dtype=dtype
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Load the ReducedBasis object
    rb = ReducedBasis.load(filepath='data/reduced_basis.h5', group_name='test_1', device=device)
    print('Shape of the reduced basis:', rb.B.shape)
    print('Truncation limit:', rb.truncation_limit)    
    print('Shape of the Theta matrix:', rb.Theta.shape)
    
    # Determine the number of features (number of modes + 1 for volume fraction)
    num_modes = rb.B.shape[1]
    num_features = num_modes + 1  # +1 for volume fraction
    
    # Prepare the output HDF5 file
    # output_h5_path = '/home/keshav/Desktop/VoigtReussNN/data/feature_engineering_data.h5'
    output_h5_path = 'data/projected_basis.h5'
    def project_and_save(loader, group, total_samples):
        if 'feature_vector_rb_v1' in group:
            del group['feature_vector_rb_v1']
        features_ds = group.create_dataset(
            'feature_vector_rb_v1', shape=(total_samples, num_features), dtype='float64'
        )
        
        idx = 0
        for images in loader:
            images = images.to(device)
            batch_size_actual = images.size(0)
            vol_frac = images.view(images.size(0), -1).mean(dim=1).cpu().numpy()
            pcf = compute_2pcf(images).to(device)
            xi = rb.get_xi(pcf).T.cpu().numpy()
            features = np.hstack((vol_frac.reshape(-1, 1), xi))
            features_ds[idx:idx+batch_size_actual, :] = features
            idx += batch_size_actual
            del images, pcf, xi, vol_frac, features
            torch.cuda.empty_cache()
    
    with h5py.File(output_h5_path, 'a') as h5f:
        project_and_save(train_loader, h5f.require_group('train_set'), total_samples_train)
        project_and_save(test_loader, h5f.require_group('benchmark_set'), total_samples_test)
    
    print(f'Projected data saved to {output_h5_path}')

if __name__ == '__main__':
    main()