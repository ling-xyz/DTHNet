import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree
from sklearn.cluster import DBSCAN

def cluster(points, eps, min_samples):
    if points.shape[0] < min_samples:
        return np.full(points.shape[0], -1, dtype=np.int64)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(points).astype(np.int64)


def hull_volume(points):
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 4:
        return np.nan
    try:
        hull = ConvexHull(points)
    except QhullError:
        return np.nan
    origin = points[hull.vertices].mean(axis=0)
    total = 0.0
    for simplex in hull.simplices:
        a = points[simplex[0]] - origin
        b = points[simplex[1]] - origin
        c = points[simplex[2]] - origin
        total += abs(np.dot(a, np.cross(b, c))) / 6.0
    return float(total)


def confusion_matrix(pred, label, num_classes=2):
    valid = (label >= 0) & (label < num_classes)
    flat = label[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64)
    return np.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def iou_from_confusion(matrix):
    tp = np.diag(matrix).astype(np.float64)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    denom = tp + fp + fn
    return np.divide(tp, denom, out=np.full_like(tp, np.nan), where=denom > 0)


def boundary_f1(pred, label, xyz, tau, num_classes=2):
    result = [np.nan, np.nan]
    if xyz.shape[0] < 2:
        return result
    masks = [label == c for c in range(num_classes)]
    if not all(mask.any() for mask in masks):
        return result
    trees = [cKDTree(xyz[mask]) for mask in masks]
    distances = []
    for c in range(num_classes):
        d, _ = trees[c].query(xyz, k=1)
        distances.append(d)
    band = np.zeros(label.shape[0], dtype=bool)
    for c in range(num_classes):
        other = 1 - c
        band |= masks[c] & (distances[other] <= tau)
    for c in range(num_classes):
        positive = masks[c] & band
        negative = (~masks[c]) & band
        tp = np.sum((pred == c) & positive)
        fp = np.sum((pred == c) & negative)
        fn = np.sum((pred != c) & positive)
        denom = 2 * tp + fp + fn
        if denom > 0:
            result[c] = 2.0 * tp / denom
    return result


def match_instances(matrix, threshold=0.5):
    matches = []
    if matrix.size == 0:
        return matches
    order = np.argsort(matrix.ravel())[::-1]
    used_gt = set()
    used_pred = set()
    for flat in order:
        i, j = np.unravel_index(flat, matrix.shape)
        if matrix[i, j] < threshold:
            break
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i)
        used_pred.add(j)
        matches.append((i, j))
    return matches


def r2_rmse(estimate, reference):
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if estimate.size == 0:
        return np.nan, np.nan
    residual = estimate - reference
    total = reference - reference.mean()
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum(total ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(r2), float(np.sqrt(np.mean(residual ** 2)))
