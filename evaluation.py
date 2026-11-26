from collections import Counter

import numpy as np

from metrics import cluster, hull_volume, match_instances, r2_rmse


def stage_of(record):
    parts = record["id"].split("_")
    return str(record.get("stage", parts[1] if len(parts) > 1 else "NA"))


def read_volume_references(path, sheet=None, columns=None):

    from openpyxl import load_workbook

    names = {"block_id": "block_id", "instance_id": "instance_id", "plant_id": "plant_id",
             "reference_m3": "reference_m3", "length_mm": "length_mm",
             "width_mm": "width_mm", "height_mm": "height_mm"}
    names.update(columns or {})
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = (workbook[sheet] if sheet is not None else workbook.active).values
        header = [str(cell).strip() for cell in next(rows, ())]
        measured_volume = names["reference_m3"] in header
        values = ("reference_m3",) if measured_volume else ("length_mm", "width_mm", "height_mm")
        keys = ("block_id", "instance_id", "plant_id", *values)
        indices = {key: header.index(names[key]) for key in keys if names[key] in header}
        if len(indices) != len(keys):
            missing = ", ".join(names[key] for key in keys if key not in indices)
            raise ValueError(f"Missing Excel columns: {missing}")

        references = {}
        for line, row in enumerate(rows, start=2):
            if all(cell is None for cell in row):
                continue
            cells = {key: row[index] for key, index in indices.items()}
            try:
                block = str(cells["block_id"]).strip() if cells["block_id"] is not None else ""
                plant = str(cells["plant_id"]).strip() if cells["plant_id"] is not None else ""
                instance = int(cells["instance_id"])
                measurements = np.array([float(cells[key]) for key in values])
                if not block or not plant or instance < 0 or float(cells["instance_id"]) != instance:
                    raise ValueError("invalid block, plant or instance identifier")
                if not np.all(np.isfinite(measurements) & (measurements > 0)):
                    raise ValueError("measurements must be positive and finite")
                reference = measurements[0] if measured_volume else np.pi / 6 * np.prod(measurements / 1000)
                if not np.isfinite(reference) or reference <= 0:
                    raise ValueError("invalid reference volume")
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"Invalid Excel row {line}: {error}") from error
            key = (block, instance)
            if key in references:
                raise ValueError(f"Duplicate Excel reference at row {line}: {key}")
            references[key] = (plant, float(reference))
        if not references:
            raise ValueError("Excel workbook contains no reference measurements")
        return references
    finally:
        workbook.close()


def _volume_stats(pairs):
    estimate = np.array([row["estimate"] for row in pairs])
    reference = np.array([row["reference"] for row in pairs])
    r2, rmse = r2_rmse(estimate, reference)
    slope, intercept = (np.polyfit(reference, estimate, 1)
                        if reference.size > 1 and np.ptp(reference) > 0 else (np.nan, np.nan))
    return {"r2": r2, "rmse": rmse, "rmse_1e-4m3": rmse * 1e4, "n": len(pairs),
            "slope": float(slope), "intercept": float(intercept)}


def summarize_instances(records, cfg):
    settings = cfg["instance"]
    volume = cfg.get("volume", {})
    excel = volume.get("excel")
    refs = read_volume_references(excel, volume.get("sheet"), volume.get("columns")) if excel else {}

    overlap_fraction = float(volume.get("overlap_fraction", 0.10))
    if not 0 < overlap_fraction <= 0.5:
        raise ValueError("volume.overlap_fraction must be in (0, 0.5]")
    total_gt = total_pred = total_matches = 0
    coverage_sum = weighted_sum = size_sum = 0
    per_block, predicted_volumes, audit = [], [], []
    block_stages = {}
    stage_plants = set()
    matched_keys = set()

    for record in records:
        block = record["id"]
        stage = stage_of(record)
        if block in block_stages:
            raise ValueError(f"Duplicate prediction block: {block}")
        block_stages[block] = stage
        points = np.asarray(record["xyz_raw"], dtype=np.float64)
        labels = np.asarray(record["label"])
        prediction = np.asarray(record["pred"])
        manual = record.get("instance_ids")
        if manual is None:
            raise ValueError(f"{block}: manual instance_ids are required")
        manual = np.asarray(manual)
        if manual.shape != labels.shape or manual.dtype.kind not in "iu" or np.any(manual[labels == 0] < 0):
            raise ValueError(f"{block}: each labeled head point needs a valid manual instance ID")

        gt = np.where(labels == 0, manual, -1)
        is_head = (prediction == 0) & np.isin(labels, (0, 1))
        pred = np.full(labels.shape, -1, dtype=np.int64)
        pred[is_head] = cluster(points[is_head], float(settings["eps"]), int(settings["min_samples"]))
        pred_ids, pred_sizes = np.unique(pred[pred >= 0], return_counts=True)
        keep = pred_sizes >= int(settings.get("min_points", 1))
        pred[~np.isin(pred, pred_ids[keep])] = -1
        pred_ids, pred_sizes = pred_ids[keep], pred_sizes[keep]
        gt_ids, gt_sizes = np.unique(gt[gt >= 0], return_counts=True)
        intersections = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
        common = (gt >= 0) & (pred >= 0)
        np.add.at(intersections, (np.searchsorted(gt_ids, gt[common]),
                                  np.searchsorted(pred_ids, pred[common])), 1)
        iou = intersections / (gt_sizes[:, None] + pred_sizes - intersections)
        matches = dict(match_instances(iou))
        coverage = iou.max(axis=1) if len(pred_ids) else np.zeros(len(gt_ids))
        total_gt += len(gt_ids)
        total_pred += len(pred_ids)
        total_matches += len(matches)
        coverage_sum += coverage.sum()
        weighted_sum += coverage @ gt_sizes
        size_sum += gt_sizes.sum()
        per_block.append({"block": block, "gt_instances": len(gt_ids),
                          "pred_instances": len(pred_ids), "matches": len(matches)})

        estimates = {}
        for pred_id, count in zip(pred_ids, pred_sizes):
            estimate = hull_volume(points[pred == pred_id])
            estimates[pred_id] = estimate
            if np.isfinite(estimate):
                predicted_volumes.append({"block": block, "stage": stage, "instance": int(pred_id),
                                          "points": int(count), "volume_m3": estimate,
                                          "volume_cm3": estimate * 1e6})



        is_split = (intersections >= overlap_fraction * gt_sizes[:, None]).sum(axis=1) > 1
        is_merged = (intersections >= overlap_fraction * pred_sizes).sum(axis=0) > 1
        for i, gt_id in enumerate(gt_ids):
            key = (block, int(gt_id))
            if key not in refs:
                continue
            matched_keys.add(key)
            plant, reference = refs[key]
            if (stage, plant) in stage_plants:
                raise ValueError(f"Duplicate measured plant in stage {stage}: {plant}")
            stage_plants.add((stage, plant))
            row = {"block": block, "stage": stage, "gt": int(gt_id),
                   "plant_id": plant, "reference": reference}
            j = matches.get(i)
            if j is not None:
                row.update(pred=int(pred_ids[j]), iou=float(iou[i, j]))
            merged = np.any(is_merged & (intersections[i] >= overlap_fraction * pred_sizes))
            reasons = []
            if is_split[i]:
                reasons.append("split")
            if merged:
                reasons.append("merged")
            if not reasons and j is None:
                reasons.append("no_match" if intersections[i].any() else "missed")
            if not reasons:
                estimate = estimates[pred_ids[j]]
                if np.isfinite(estimate):
                    row["estimate"] = estimate
                else:
                    reasons.append("invalid_hull")
            row["status"] = "+".join(reasons) if reasons else "matched"
            audit.append(row)

    for key in sorted(refs.keys() - matched_keys):
        block, gt = key
        plant, reference = refs[key]
        stage = block_stages.get(block, stage_of({"id": block}))
        if (stage, plant) in stage_plants:
            raise ValueError(f"Duplicate measured plant in stage {stage}: {plant}")
        stage_plants.add((stage, plant))
        status = "missing_gt" if block in block_stages else "missing_block"
        audit.append({"block": block, "stage": stage, "gt": gt, "plant_id": plant,
                      "reference": reference, "status": status})

    pairs = [row for row in audit if row["status"] == "matched"]
    result = {
        "mPrec": total_matches / total_pred if total_pred else np.nan,
        "mRec": total_matches / total_gt if total_gt else np.nan,
        "mCov": float(coverage_sum / total_gt) if total_gt else np.nan,
        "mWCov": float(weighted_sum / size_sum) if size_sum else np.nan,
        "matches": total_matches,
        "per_block": per_block,
        "predicted_volumes": predicted_volumes,
        "volume_pairs": pairs,
        "volume_audit": audit,
        "volume": _volume_stats(pairs),
    }
    if pairs:
        result["volume"]["mean_reference_1e-4m3"] = float(np.mean([row["reference"] for row in pairs]) * 1e4)
        result["volume"]["mean_estimate_1e-4m3"] = float(np.mean([row["estimate"] for row in pairs]) * 1e4)
    result["volume"]["status_counts"] = dict(Counter(row["status"] for row in audit))
    result["volume"]["per_stage"] = {}
    for stage in sorted({row["stage"] for row in audit}):
        stats = _volume_stats([row for row in pairs if row["stage"] == stage])
        stats["status_counts"] = dict(Counter(row["status"] for row in audit if row["stage"] == stage))
        result["volume"]["per_stage"][stage] = stats
    return result
