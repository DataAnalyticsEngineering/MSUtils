# %%
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from pyrecest.sampling.hyperspherical_sampler import LeopardiSampler as LeopardiSampler

torch.set_default_dtype(torch.double)

def DistanceMatrix(x):
    # D_ij = || x_i - x_j ||^2 = 2 - 2*x_i.x_j
    n=x.shape[0]
    L = torch.norm(x, dim=1)
    D = torch.sqrt(torch.relu(2.*(torch.ones(n,n) - x@x.T/ L[:,None] / L[None, :])))
    return D

class PointEnergy(torch.nn.Module):
    def __init__(self, dim=2, N=6):
        super(PointEnergy, self).__init__()
        sampler = LeopardiSampler(original_code_column_order=True)
        X, description = sampler.get_grid(N, dim)
        X = torch.tensor(X.copy())
        # Add a small perturbation to the points
        X += 1e-3*torch.randn(X.shape)
        X = X/((torch.norm(X, dim=1))[:, None])
        self.pts = torch.nn.Parameter(X)
        self.N = N
    
    def alt_force(self):
        F = torch.zeros(self.pts.shape)
        L = torch.norm(self.pts, dim=1)
        D = DistanceMatrix(self.pts)
        Dsafe = D + 1e-14
        factor = torch.log(Dsafe/2.)/Dsafe
        factor[range(self.N), range(self.N)] = 0.
        xxt = self.pts @ self.pts.T
        for i in range(self.N):
            for j in range(self.N):
                if( i != j ):
                    F[i, :] += - factor[i, j]/L[i]/L[j] * ( self.pts[j, :] - xxt[i,j] * self.pts[i, :]/L[i]**2 )
        return F/self.N**2

    def force(self):
        D = DistanceMatrix(self.pts)
        Dsafe = D + 1e-14
        factor = torch.log(Dsafe/2.)/Dsafe
        factor[range(self.N), range(self.N)] = 0.

        L = torch.norm(self.pts, dim=1)
        xxt = self.pts @ self.pts.T
        dW =    - torch.sum( (factor / L[:, None] / L[None, :])[:, :, None] * self.pts[None, :, :], dim = 1) \
                +  self.pts * (torch.sum(factor / L[None, :] /((L**3)[:, None])  * xxt, dim = 1)[:, None])
        
        return dW.ravel()/self.N**2

    def backproject(self):
        with torch.no_grad():
            L = torch.norm(self.pts.data, dim=1)
            self.pts.data /= L[:, None]

    def loss(self):
        D = DistanceMatrix(self.pts)
        Dsafe = D + 1e-14
        W = torch.mean( D*(torch.log(Dsafe/2.) - 1.) + 2.0 )
        return W

# initial points
model = PointEnergy(dim=5, N=512)

print(f"initial energy: {model.loss():16.10e}")
param = []
for m in (model,):
    param = param + list(m.parameters())

# -----------------------------------------------------------------------------
def PlotDistance(X, ax=None):
    D = DistanceMatrix(X)
    if( ax is None ):
        fig, ax = plt.subplots(1, 1)
    for d in D:
        d = torch.sort(d).values
        ax.plot(d, color='black', alpha=0.25)
# -----------------------------------------------------------------------------
optimizer = optim.LBFGS(param, lr=.25, max_iter=40, history_size=10,
                        tolerance_grad=1e-09, tolerance_change=1e-14)

# normalization factor
lambda_W = 1./model.loss().detach()
# -----------------------------------------------------------------------------
n_print=10
X0 = torch.clone(model.pts).detach()
for it in range(250):
    def closure():
        optimizer.zero_grad()  # Reset gradients
        model.backproject()
        loss = lambda_W * model.loss() - 1.
        loss.backward()  # Backpropagation|
        return loss
    
    optimizer.step(closure)  # Perform L-BFGS optimization step
    if(it % n_print == 0):
        print(f'it {it:05d} - rel. energy={closure().item():16.10e}, W={model.loss():16.10e}')
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(1, 2)
PlotDistance(X0, ax[0])
ax[0].set_title('initial')
PlotDistance(model.pts.detach(), ax[1])
ax[1].set_title('final')
