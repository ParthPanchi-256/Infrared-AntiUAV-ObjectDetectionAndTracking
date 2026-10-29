"""t-SVD image sparsification (PISTTN Sec. II-B, Eqs. 1-4)."""
import torch


def tsvd_sparsify(clip: torch.Tensor, rank: int = 1, thresh: float = 20.0):
    """clip: (T, H, W) float tensor of raw 8-bit intensities.

    Returns (coords, values): coords (N, 3) int [t, y, x] of voxels with |S'| >= thresh,
    values (N,) the sparse residual S' at those voxels.
    """
    T, H, W = clip.shape
    y = clip.reshape(T, H * W).T                       # Y' in R^{(HW) x T}
    u, s, vh = torch.linalg.svd(y, full_matrices=False)
    low = (u[:, :rank] * s[:rank]) @ vh[:rank]         # rank-r background L'
    sparse = (y - low).T.reshape(T, H, W)              # S' = Y' - L'
    mask = sparse.abs() >= thresh
    coords = mask.nonzero()
    return coords, sparse[mask]
