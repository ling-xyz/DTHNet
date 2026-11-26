from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from metrics import boundary_f1, confusion_matrix, iou_from_confusion
from evaluation import summarize_instances, stage_of
from model import triplet_loss

KEYS = [
    "xyz",
    "xyz_raw",
    "rgb",
    "labels",
    "knn0",
    "knn1",
    "knn2",
    "kd0",
    "kd1",
    "kd2",
    "fps1",
    "fps2",
    "fps3",
    "interp1_idx",
    "interp1_w",
    "interp2_idx",
    "interp2_w",
    "interp3_idx",
    "interp3_w",
]


class CabbageBlocks(Dataset):
    def __init__(self, processed, ids):
        self.root = Path(processed)
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        block_id = self.ids[index]
        item = {"id": block_id}
        with np.load(self.root / f"{block_id}.npz") as data:
            for key in KEYS:
                if key == "xyz_raw":
                    dtype = torch.float64
                elif key in ("xyz", "rgb", "interp1_w", "interp2_w", "interp3_w"):
                    dtype = torch.float32
                else:
                    dtype = torch.long
                item[key] = torch.from_numpy(data[key]).to(dtype)
            if "instance_ids" in data:
                annotation = data["instance_ids"]
                if annotation.shape != data["labels"].shape or annotation.dtype.kind not in "iu":
                    raise ValueError(f"{block_id}: instance_ids must be an integer array aligned with labels")
                item["instance_ids"] = torch.from_numpy(annotation).long()
        return item


def collate_blocks(blocks):
    batch = {key: torch.stack([block[key] for block in blocks]) for key in KEYS}
    batch["ids"] = [block["id"] for block in blocks]
    batch["instance_ids"] = [block.get("instance_ids") for block in blocks]
    return batch


def move_batch(batch, device):
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=True)
    return batch


def amp_context(mode):
    if mode == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if mode == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_epoch(model, loader, optimizer, device, cfg, scaler=None):
    model.train()
    amp = cfg["train"]["amp"]
    accum = int(cfg["train"]["accum"])
    clip = float(cfg["train"].get("clip", 0.0))
    ce_weight = float(cfg["loss"]["ce"])
    tri_weight = float(cfg["loss"]["triplet"])
    margin = float(cfg["loss"]["margin"])
    anchor_limit = cfg["loss"].get("anchors")
    anchors = int(anchor_limit) if anchor_limit is not None else None
    class_weights = cfg["loss"].get("class_weights")
    weight = torch.tensor(class_weights, device=device, dtype=torch.float32) if class_weights else None
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "ce": 0.0, "triplet": 0.0, "steps": 0}
    batches = len(loader)
    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        with amp_context(amp):
            logits, hidden = model(batch)
        labels = batch["labels"]
        if not ((labels >= 0) & (labels < 2)).any():
            ce = logits.float().sum() * 0.0
        else:
            ce = F.cross_entropy(
                logits.float().reshape(-1, 2), labels.reshape(-1), weight=weight, ignore_index=-1
            )
        tri = triplet_loss(hidden, batch["labels"], margin, anchors)
        loss = ce_weight * ce + tri_weight * tri
        group_size = min(accum, batches - (step // accum) * accum)
        scaled = loss / group_size
        if scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()
        if (step + 1) % accum == 0 or step + 1 == batches:
            if scaler is not None:
                scaler.unscale_(optimizer)
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        totals["loss"] += float(loss.detach())
        totals["ce"] += float(ce.detach())
        totals["triplet"] += float(tri.detach())
        totals["steps"] += 1
    steps = max(totals["steps"], 1)
    return {key: totals[key] / steps for key in ("loss", "ce", "triplet")}


@torch.no_grad()
def predict_records(model, loader, device, amp):
    model.eval()
    records = []
    for batch in loader:
        batch = move_batch(batch, device)
        with amp_context(amp):
            logits, _ = model(batch)
        prediction = logits.argmax(dim=-1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        xyz = batch["xyz"].cpu().numpy()
        xyz_raw = batch["xyz_raw"].cpu().numpy()
        annotations = batch.get("instance_ids", [None] * len(batch["ids"]))
        for i, block_id in enumerate(batch["ids"]):
            records.append(
                {
                    "id": block_id,
                    "instance_ids": None if annotations[i] is None else annotations[i].cpu().numpy(),
                    "pred": prediction[i],
                    "label": labels[i],
                    "xyz": xyz[i],
                    "xyz_raw": xyz_raw[i],
                    "logits": logits[i].float().cpu().numpy(),
                }
            )
    return records


def summarize_semantic(records, tau=0.02):
    total = np.zeros((2, 2), dtype=np.int64)
    per_block = []
    stage_values = defaultdict(list)
    for record in records:
        matrix = confusion_matrix(record["pred"], record["label"], 2)
        total += matrix
        iou = iou_from_confusion(matrix)
        head_f1, outer_f1 = boundary_f1(record["pred"], record["label"], record["xyz_raw"], tau)
        stage = stage_of(record)
        stage_values[stage].append((iou, head_f1, outer_f1))
        per_block.append(
            {
                "block": record["id"],
                "head_iou": float(iou[0]),
                "outer_iou": float(iou[1]),
                "miou": float(np.nanmean(iou)),
                "h_f1_2": float(head_f1),
                "o_f1_2": float(outer_f1),
                "oa": float(np.trace(matrix) / max(matrix.sum(), 1)),
            }
        )
    iou = iou_from_confusion(total)
    result = {
        "head_iou": float(iou[0]),
        "outer_iou": float(iou[1]),
        "miou": float(np.nanmean(iou)),
        "oa": float(np.trace(total) / max(total.sum(), 1)),
        "h_f1_2": float(np.nanmean([row["h_f1_2"] for row in per_block])),
        "o_f1_2": float(np.nanmean([row["o_f1_2"] for row in per_block])),
        "blocks": len(records),
        "per_block": per_block,
        "per_stage": {},
    }
    for stage, values in sorted(stage_values.items()):
        ious = np.stack([value[0] for value in values], axis=0)
        result["per_stage"][stage] = {
            "head_iou": float(np.nanmean(ious[:, 0])),
            "outer_iou": float(np.nanmean(ious[:, 1])),
            "miou": float(np.nanmean(ious)),
            "h_f1_2": float(np.nanmean([value[1] for value in values])),
            "o_f1_2": float(np.nanmean([value[2] for value in values])),
            "blocks": len(values),
        }
    return result

