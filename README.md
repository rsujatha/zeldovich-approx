<h1>zeldovich-approximation</h1>
Graduate school project on the Zeldovich approximation or 1LPT,
      used to generate initial conditions for N-body simulations.
<table>
  <tr>
    <td>
      <img src="https://github.com/rsujatha/zeldovich-approx/blob/main/zeldovich100.gif?raw=true" alt="Demo animation" width="500"/>
    </td>
    <td>
      Animation shows evolution of matter under gravity from very small fluctuations to the formation of cosmic web filaments. Matter collapse is roughly categorised as 3d collapse (nodes), 3d collapse (sheets), 1d collapse (filaments) and no collapse (voids)
    </td>
  </tr>
</table>
[📄 Read the full report (PDF)](https://github.com/rsujatha/zeldovich-approx/blob/main/report_corrected.pdf)

```python
# lpt.py: Generate Zeldovich Approximation displacements

import numpy as np

def lpt_displacement(delta_k, kvec, growth_factor=1.0):
    """
    Compute 1LPT displacement field in Fourier space.

    Parameters
    ----------
    delta_k : ndarray (complex)
        Density contrast in Fourier space.
    kvec : ndarray, shape (...,3)
        Corresponding k-vectors for each mode.
    growth_factor : float, optional
        Scale factor for structure growth (default=1.0).

    Returns
    -------
    psi_k : ndarray (complex)
        Displacement field in Fourier space.
    """
    k2 = np.sum(kvec**2, axis=-1)                     # |k|^2
    with np.errstate(divide='ignore', invalid='ignore'):
        # Avoid division by zero at k=0 by masking
        psi_k = 1j * kvec * (delta_k / k2)[..., None]
    psi_k *= growth_factor                            # scale by growth factor
    return psi_k

def apply_displacement(psi, x_grid):
    """
    Move particles by displacement field in real space.

    Parameters
    ----------
    psi : ndarray, shape (N,3)
        Real-space displacement vectors.
    x_grid : ndarray, shape (N,3)
        Initial particle positions (e.g., grid).

    Returns
    -------
    x_new : ndarray, shape (N,3)
        Updated particle positions.
    """
    return x_grid + psi

if __name__ == "__main__":
    # Example usage block
    N = 64
    L = 100.0  # Box size
    # Prepare a k-space grid
    k_vals = np.fft.fftfreq(N, d=L/N) * 2*np.pi
    kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing="ij")
    kvec = np.stack((kx, ky, kz), axis=-1)

    # Sample Gaussian delta field in k-space
    delta_k = (np.random.normal(size=(N,N,N)) + 1j*np.random.normal(size=(N,N,N)))
    ps = np.abs(kvec)**-3  # example power spectrum shape
    delta_k *= np.sqrt(ps)

    psi_k = lpt_displacement(delta_k, kvec, growth_factor=0.8)

    # Convert to real-space displacement
    psi = np.fft.ifftn(psi_k, axes=(0,1,2))
    psi = np.real(psi)

    # Setup initial grid positions
    coords = np.linspace(0, L, N, endpoint=False)
    x_grid = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1)

    # Apply displacement
    x_new = apply_displacement(psi.reshape(-1,3), x_grid.reshape(-1,3))
    print("Sample moved position:", x_new[0])
