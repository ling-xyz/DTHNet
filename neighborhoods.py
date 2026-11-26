import numpy as np
from scipy.spatial import cKDTree


def build_kd_neighborhoods(xyz, k):

    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not 1 <= k < len(xyz):
        raise ValueError("Expected xyz with shape (N, 3) and 1 <= k < N")
    _, candidates = cKDTree(xyz).query(xyz, k=k + 1)
    neighbors = np.stack([row[row != i][:k] for i, row in enumerate(candidates)])
    depth = k.bit_length()
    trees = np.full((len(xyz), (1 << (depth + 1)) - 1), -1, dtype=np.int64)
    trees[:, 0] = k

    def fill(points, indices, slots, position):
        if not len(indices):
            return
        axis = np.ptp(points[indices], axis=0).argmax()
        ordered = indices[np.argsort(points[indices, axis], kind="stable")]
        mid = len(ordered) // 2
        slots[position] = ordered[mid]
        fill(points, ordered[:mid], slots, 2 * position + 1)
        fill(points, ordered[mid + 1:], slots, 2 * position + 2)

    for i, indices in enumerate(neighbors):
        points = xyz[indices]
        axis = np.ptp(points, axis=0).argmax()
        left = points[:, axis] < xyz[i, axis]
        fill(points, np.flatnonzero(left), trees[i], 1)
        fill(points, np.flatnonzero(~left), trees[i], 2)
    return neighbors, trees
