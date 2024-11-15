import h5py
import torch


class ReducedBasis:
    """
    A reduced basis object that can iteratively update the reduced basis
    given snapshot data using the 'svd' method.
    Designed to be general-purpose and applicable to any type of data.
    """

    def __init__(self, truncation_limit=0.05, device=None):
        """
        Initialize the ReducedBasis class with default parameters.
        Parameters:
        -----------
        truncation_limit: float, default 0.05
            The maximum allowed truncation error when computing or adjusting the basis.
        device: torch.device or str, default None
            The device ('cpu' or 'cuda') to use for computations.
            If None, will default to 'cpu'.
        """
        self.truncation_limit = truncation_limit
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.B = None       # The reduced basis matrix
        self.Theta = None   # Singular values

    def _compute_svd(self, matrix):
        """
        Compute the SVD of the input matrix.
        """
        matrix = matrix.to(self.device)
        U, Theta, Vh = torch.linalg.svd(matrix, full_matrices=False)
        return U, Theta, Vh

    def compute_basis(self, S):
        """ 
        Computes the reduced basis from the snapshot matrix S using SVD.
        """
        # Ensure S is on the correct device
        S = S.to(self.device)

        # Compute SVD
        U, Theta, Vh = self._compute_svd(S)

        # Compute the truncation limit
        cumulative_energy = torch.cumsum(Theta**2, dim=0)
        total_energy = cumulative_energy[-1]
        energy_ratio = cumulative_energy / total_energy
        N = torch.searchsorted(energy_ratio, 1 - self.truncation_limit, right=True).item() + 1

        # Truncate the reduced basis
        self.Theta = Theta[:N]
        self.B = U[:, :N]

        # Clean up memory
        del Vh, Theta, U
        torch.cuda.empty_cache()

    def adjust_basis(self, delta_S): 
        """
        Adjust the current basis with the snapshots matrix delta_S using the 'svd' method.
        """
        # Ensure delta_S is on the correct device
        delta_S = delta_S.to(self.device)

        # Project delta_S onto the existing basis
        xi = self.B.T @ delta_S

        # Compute the residual
        S_residual = delta_S - self.B @ xi

        # Compute SVD of the residual
        U_residual, Theta_residual, _ = self._compute_svd(S_residual)

        # Construct the combined matrix A
        # Combining existing singular values and projections with residual singular values
        Theta_diag = torch.diag(self.Theta)
        Theta_residual_diag = torch.diag(Theta_residual)
        zeros_bottom_left = torch.zeros(Theta_residual.shape[0], self.Theta.shape[0], device=self.device)

        A_top = torch.cat((Theta_diag, xi), dim=1)
        A_bottom = torch.cat((zeros_bottom_left, Theta_residual_diag), dim=1)
        A = torch.cat((A_top, A_bottom), dim=0)

        # Compute SVD of the combined matrix A
        U_A, Theta_new, _ = self._compute_svd(A)

        # Compute the truncation limit
        cumulative_energy = torch.cumsum(Theta_new**2, dim=0)
        total_energy = cumulative_energy[-1]
        energy_ratio = cumulative_energy / total_energy
        N_trunc = torch.searchsorted(1 - energy_ratio, self.truncation_limit, right=True).item() + 1

        # Update self.Theta and self.B
        self.Theta = Theta_new[:N_trunc]
        self.B = torch.cat((self.B, U_residual), dim=1) @ U_A[:, :N_trunc]

        # Clean up memory
        del xi, S_residual, U_residual, Theta_residual
        del Theta_diag, Theta_residual_diag, zeros_bottom_left, A_top, A_bottom, A, U_A, Theta_new
        torch.cuda.empty_cache()

    def get_xi(self, data, n_xi=None):
        """
        Compute the reduced coefficients of the snapshots
        using the current reduced basis.
        """
        data = data.to(self.device)
        n_xi = n_xi if n_xi is not None else self.B.shape[1]
        return self.B[:, :n_xi].T @ data

    def projection_error(self, data, n_xi=None):
        """
        Compute the projection error of the snapshots onto the reduced basis.

        Parameters:
        -----------
        data: torch.Tensor
            The snapshot data, with shape (num_features, num_snapshots).
        n_xi: int, optional
            Number of reduced basis modes to use. If None, use all available modes.

        Returns:
        --------
        errors: torch.Tensor
            Absolute projection errors for each snapshot, shape (num_snapshots,).
        relative_errors: torch.Tensor
            Relative projection errors for each snapshot, shape (num_snapshots,).
        """
        data = data.to(self.device)
        n_xi = n_xi if n_xi is not None else self.B.shape[1]

        xi = self.B[:, :n_xi].T @ data
        data_proj = self.B[:, :n_xi] @ xi
        residuals = data - data_proj
        errors = torch.linalg.norm(residuals, dim=0)
        data_norms = torch.linalg.norm(data, dim=0)
        relative_errors = errors / data_norms

        return errors.cpu(), relative_errors.cpu()


    def reorthogonalize_B(self, tolerance=1e-9):
        """
        Reorthogonalizes the reduced basis self.B 
        if B.T @ B deviates from the identity matrix beyond a certain threshold.
        """
        ortho = self.B.T @ self.B
        deviation = torch.sum(torch.abs(torch.diag(ortho) - 1.0))
        if deviation.item() > tolerance:
            print("Reorthogonalizing the basis matrix: deviation =", deviation.item())            
            Q, _ = torch.linalg.qr(self.B)
            self.B = Q
            return 1
        return 0
    
    def save(self, filepath, group_name, compression_level=6):
        """
        Save the ReducedBasis object to a specific group in a file with optional gzip compression.
        If the group already exists, it will be deleted before saving the new data.
        
        Parameters:
        -----------
        filepath: str
            The path to the file where the object will be saved.
        group_name: str
            The name of the group within the HDF5 file where the object will be saved.
        compression_level: int, default 6
            The gzip compression level (0-9) to use for saving the file.
        """

        # Open the file in append mode
        with h5py.File(filepath, 'a') as f:
            # Remove the group if it already exists
            if group_name in f:
                del f[group_name]
                print(f"Group '{group_name}' already exists. Deleting the existing group.")

            # Create the group and save the object to it
            group = f.create_group(group_name)
            group.create_dataset('truncation_limit', data=self.truncation_limit)
            group.create_dataset('B', data=self.B.cpu().numpy(), compression='gzip', compression_opts=compression_level)
            group.create_dataset('Theta', data=self.Theta.cpu().numpy(), compression='gzip', compression_opts=compression_level)

    @classmethod
    def load(cls, filepath, group_name, device=None):
        """
        Load a ReducedBasis object from a specific group in a file.
        
        Parameters:
        -----------
        filepath: str
            The path to the file where the object is saved.
        group_name: str
            The name of the group within the HDF5 file where the object is saved.
        device: torch.device or str, optional
            The device ('cpu' or 'cuda') to use for computations.
            If None, defaults to 'cpu'.
        
        Returns:
        --------
        ReducedBasis
            An instance of the ReducedBasis class.
        """
        with h5py.File(filepath, 'r') as f:
            group = f[group_name]
            truncation_limit = group['truncation_limit'][()]
            B = torch.tensor(group['B'][:])
            Theta = torch.tensor(group['Theta'][:])

        obj = cls(truncation_limit=truncation_limit, device=device)
        obj.B = B.to(obj.device)
        obj.Theta = Theta.to(obj.device)
        return obj