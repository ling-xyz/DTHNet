import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def tree_depth(k):
    return int(np.ceil(np.log2(max(k + 1, 2))))


def slot_count(k):
    return (1 << (tree_depth(k) + 1)) - 1


def level_slots(k):
    depth = tree_depth(k)
    return [list(range((1 << d) - 1, (1 << (d + 1)) - 1)) for d in range(depth + 1)]



EPS = 1e-8


def expmap0(z, c):
    sqrt_c = c ** 0.5
    norm = z.norm(dim=-1, keepdim=True).clamp_min(EPS)
    scale = torch.tanh(sqrt_c * norm) / (sqrt_c * norm)
    return scale * z


def logmap0(e, c):
    sqrt_c = c ** 0.5
    norm = e.norm(dim=-1, keepdim=True).clamp_max(1.0 / sqrt_c - 1e-5).clamp_min(EPS)
    scale = torch.atanh((sqrt_c * norm).clamp_max(1.0 - 1e-7)) / (sqrt_c * norm)
    return scale * e


def poincare_dist(e1, e2, c):
    sqrt_c = c ** 0.5
    n1 = e1.norm(dim=-1, keepdim=True).pow(2).clamp_max(1.0 / c - 1e-5)
    n2 = e2.norm(dim=-1, keepdim=True).pow(2).clamp_max(1.0 / c - 1e-5)
    diff = (e1 - e2).pow(2).sum(dim=-1, keepdim=True)
    denom = (1.0 - c * n1) * (1.0 - c * n2)
    arg = 1.0 + 2.0 * c * diff / denom.clamp_min(EPS)
    return torch.acosh(arg.clamp_min(1.0 + 1e-7)) / sqrt_c


def rank_slots(similarity, k):
    order = torch.argsort(similarity, dim=1, descending=True, stable=True)
    n = order.shape[0]
    total = slot_count(k)
    slots = torch.full((n, total), -1, dtype=torch.long, device=order.device)
    slots[:, 0] = k
    slots[:, 1 : k + 1] = order
    return slots


def gather_nodes(self_feat, neighbor_feat, slots):
    pool = torch.cat([neighbor_feat, self_feat.unsqueeze(1)], dim=1)
    valid = slots >= 0
    index = slots.clamp_min(0).unsqueeze(-1).expand(-1, -1, pool.shape[-1])
    nodes = torch.gather(pool, 1, index)
    return nodes * valid.unsqueeze(-1).to(nodes.dtype), valid


class TreeGCN(nn.Module):
    def __init__(self, in_dim, dim, heads, k, curvature):
        super().__init__()
        self.dim = dim
        self.k = k
        self.depth = tree_depth(k)
        self.total = slot_count(k)
        self.levels = level_slots(k)
        self.curvature = curvature
        self.w_in = nn.Linear(in_dim, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.w_g = nn.Linear(dim, dim)
        self.w_agg = nn.Linear(dim, dim, bias=False)
        self.ln_agg = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln_root = nn.LayerNorm(dim)
        self.register_buffer("slot_index", torch.arange(self.total), persistent=False)
        parent = ((torch.arange(self.total) - 1) // 2).clamp_min(0)
        self.register_buffer("parent", parent, persistent=False)
        self.register_buffer("non_root", (torch.arange(self.total) > 0), persistent=False)

    def forward(self, node_feat, valid):
        m = node_feat.shape[0]
        device = node_feat.device
        mask = valid.unsqueeze(-1)
        z = self.w_in(node_feat) * mask
        e = expmap0(z.float(), self.curvature) * mask
        parent = self.parent.unsqueeze(0).expand(m, -1)
        child_valid = valid & self.non_root.unsqueeze(0)
        parent_e = torch.gather(e, 1, parent.unsqueeze(-1).expand(-1, -1, self.dim))
        distance = poincare_dist(parent_e, e, self.curvature).squeeze(-1)
        score = torch.where(child_valid, -distance, torch.full_like(distance, -1e9))
        pmax = torch.full_like(score, -1e9).scatter_reduce(1, parent, score, reduce="amax", include_self=True)
        exp_score = torch.exp(score - pmax.gather(1, parent)) * child_valid.to(score.dtype)
        psum = torch.zeros_like(score).scatter_add_(1, parent, exp_score)
        weight = exp_score / psum.gather(1, parent).clamp_min(1e-8)
        weight = weight * child_valid.to(weight.dtype)
        z_tan = logmap0(e, self.curvature).to(z.dtype) * mask
        weight = weight.to(z.dtype)

        u = torch.zeros_like(z_tan)
        level_features = []
        for d in range(1, self.depth + 1):
            idx = torch.tensor(self.levels[d], device=device)
            x = z_tan.index_select(1, idx)
            level_valid = valid.index_select(1, idx)
            order = torch.argsort((~level_valid).to(torch.int8), dim=1, stable=True)
            x = torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, self.dim))
            level_valid = torch.gather(level_valid, 1, order)
            counts = level_valid.sum(dim=1)
            length = int(counts.max().item()) if counts.numel() > 0 else 0
            if length == 0:
                level_features.append(torch.zeros(m, self.dim, device=device, dtype=z_tan.dtype))
                continue
            packed = pack_padded_sequence(
                x[:, :length], counts.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.gru(packed)
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=length)
            out = out.to(z_tan.dtype)
            last = out[torch.arange(m, device=device), counts.clamp_min(1) - 1]
            last = torch.where((counts > 0).unsqueeze(-1), last, torch.zeros_like(last))
            level_features.append(torch.sigmoid(self.w_g(last)) * last)
            updated = torch.zeros(m, len(idx), self.dim, device=device, dtype=z_tan.dtype)
            updated.scatter_(
                1,
                order[:, :length].unsqueeze(-1).expand(-1, -1, self.dim),
                out * level_valid[:, :length].unsqueeze(-1),
            )
            u = u.index_copy(1, idx, updated)

        root_mask = (self.slot_index == 0).view(1, -1, 1)
        z_in = torch.where(root_mask, z_tan, u)
        r = z_in
        for d in range(self.depth, -1, -1):
            idx = torch.tensor(self.levels[d], device=device)
            zs = z_in.index_select(1, idx)
            if d < self.depth:
                c1 = 2 * idx + 1
                c2 = 2 * idx + 2
                child_valid = valid.index_select(1, c1) | valid.index_select(1, c2)
                child_sum = (
                    weight.index_select(1, c1).unsqueeze(-1) * r.index_select(1, c1)
                    + weight.index_select(1, c2).unsqueeze(-1) * r.index_select(1, c2)
                )
                agg = self.ln_agg(F.relu(zs + self.w_agg(child_sum)))
                rs = torch.where(child_valid.unsqueeze(-1), agg, zs)
            else:
                rs = zs
            r = r.index_copy(1, idx, rs)

        root = r[:, 0]
        levels = torch.stack(level_features, dim=1)
        kv = torch.cat([root.unsqueeze(1), levels], dim=1)
        padding = torch.stack([~valid[:, slots].any(dim=1) for slots in self.levels], dim=1)
        attn, _ = self.attn(root.unsqueeze(1), kv, kv, key_padding_mask=padding, need_weights=False)
        return self.ln_root(F.relu(attn[:, 0]))


class DTHA(nn.Module):
    def __init__(self, in_dim, out_dim, tree_dim, heads, k, curvature, chunk):
        super().__init__()
        self.k = k
        self.chunk = chunk
        self.tree_dim = tree_dim
        self.kd_tree = TreeGCN(in_dim, tree_dim, heads, k, curvature)
        self.rank_tree = TreeGCN(in_dim, tree_dim, heads, k, curvature)
        self.refine_kd = nn.Linear(tree_dim, tree_dim)
        self.refine_rank = nn.Linear(tree_dim, tree_dim)
        self.gate = nn.Linear(2 * tree_dim, 1, bias=False)
        self.proj = nn.Linear(tree_dim, out_dim) if tree_dim != out_dim else None

    def _neighbors(self, full_feat, knn):
        return full_feat[knn]

    def _tree_output(self, feat, knn, kd, full_feat):
        neighbor = self._neighbors(full_feat, knn)
        node_kd, valid_kd = gather_nodes(feat, neighbor, kd)
        out_kd = self.kd_tree(node_kd, valid_kd)
        similarity = (feat.unsqueeze(1) * neighbor).sum(dim=-1) / (
            feat.norm(dim=-1, keepdim=True) * neighbor.norm(dim=-1) + 1e-8
        )
        slots = rank_slots(similarity, self.k)
        node_rank, valid_rank = gather_nodes(feat, neighbor, slots)
        out_rank = self.rank_tree(node_rank, valid_rank)
        return out_kd, out_rank

    def forward(self, feat, knn, kd):
        b, n, c = feat.shape
        flat = feat.reshape(b * n, c)
        offsets = torch.arange(b, device=feat.device).view(b, 1, 1) * n
        flat_knn = (knn + offsets).reshape(b * n, knn.shape[-1])
        flat_kd = kd.reshape(b * n, kd.shape[-1])
        flat_kd = F.pad(flat_kd, (0, slot_count(self.k) - flat_kd.shape[-1]), value=-1)
        outputs = []
        for start in range(0, b * n, self.chunk):
            end = min(start + self.chunk, b * n)
            z_kd, z_rank = self._tree_output(flat[start:end], flat_knn[start:end], flat_kd[start:end], flat)
            g_kd = z_kd + F.relu(self.refine_kd(z_kd))
            g_rank = z_rank + F.relu(self.refine_rank(z_rank))
            gate = torch.sigmoid(self.gate(torch.cat([g_kd, g_rank], dim=-1)))
            out = gate * g_kd + (1.0 - gate) * g_rank
            if self.proj is not None:
                out = self.proj(out)
            outputs.append(out)
        return torch.cat(outputs, dim=0).view(b, n, -1)


def gather_points(feat, index):
    return torch.gather(feat, 1, index.unsqueeze(-1).expand(-1, -1, feat.shape[-1]))


def interpolate(feat, index, weight):
    b, ns, c = feat.shape
    _, n, k = index.shape
    flat = feat.reshape(b, ns * c)
    offset = (index.unsqueeze(-1) * c + torch.arange(c, device=feat.device).view(1, 1, 1, c)).reshape(b, n * k * c)
    values = torch.gather(flat, 1, offset).reshape(b, n, k, c)
    return (values * weight.unsqueeze(-1)).sum(dim=2)


class DTHNet(nn.Module):
    def __init__(self, widths, tree_dim, heads, k, curvature, chunk, num_classes, use_rgb):
        super().__init__()
        self.use_rgb = use_rgb
        in_dim = 6 if use_rgb else 3
        w0, w1, w2, w3, w4 = widths
        self.dtha1 = DTHA(in_dim, w0, tree_dim, heads, k, curvature, chunk)
        self.dtha2 = DTHA(w0, w1, tree_dim, heads, k, curvature, chunk)
        self.dthap1 = DTHA(w1, w2, tree_dim, heads, k, curvature, chunk)
        self.dthap2 = DTHA(w2, w3, tree_dim, heads, k, curvature, chunk)
        self.dthap3 = DTHA(w3, w4, tree_dim, heads, k, curvature, chunk)
        self.global_proj = nn.Linear(w4, 1024)
        fused = w1 + w2 + w3 + w4 + 1024
        self.decoder = nn.Sequential(
            nn.Linear(fused, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, batch):
        xyz = batch["xyz"]
        feat = torch.cat([xyz, batch["rgb"]], dim=-1) if self.use_rgb else xyz
        f0 = self.dtha1(feat, batch["knn0"], batch["kd0"])
        f1 = self.dtha2(f0, batch["knn0"], batch["kd0"])
        g1 = self.dthap1(f1, batch["knn0"], batch["kd0"])
        f2 = gather_points(g1, batch["fps1"])
        g2 = self.dthap2(f2, batch["knn1"], batch["kd1"])
        f3 = gather_points(g2, batch["fps2"])
        g3 = self.dthap3(f3, batch["knn2"], batch["kd2"])
        f4 = gather_points(g3, batch["fps3"])
        global_feat = self.global_proj(f4).max(dim=1).values
        u1 = interpolate(f2, batch["interp1_idx"], batch["interp1_w"])
        u2 = interpolate(f3, batch["interp2_idx"], batch["interp2_w"])
        u3 = interpolate(f4, batch["interp3_idx"], batch["interp3_w"])
        global_feat = global_feat.unsqueeze(1).expand(-1, xyz.shape[1], -1)
        fused = torch.cat([f1, u1, u2, u3, global_feat], dim=-1)
        hidden = F.relu(self.decoder[0](fused))
        hidden = F.relu(self.decoder[2](hidden))
        logits = self.decoder[4](hidden)
        return logits, hidden


def triplet_loss(feat, labels, margin, max_anchors):
    f = feat.reshape(-1, feat.shape[-1]).float()
    y = labels.reshape(-1)
    valid = (y >= 0) & (y < 2)
    index = valid.nonzero(as_tuple=False).squeeze(1)
    if index.numel() < 3:
        return f.sum() * 0.0
    if max_anchors is not None and index.numel() > max_anchors:
        perm = torch.randperm(index.numel(), device=index.device)[:max_anchors]
        index = index[perm]
    anchor_label = y[index]
    eligible = torch.zeros_like(anchor_label, dtype=torch.bool)
    positive = torch.empty_like(index)
    negative = torch.empty_like(index)
    for class_id in (0, 1):
        same = (y == class_id).nonzero(as_tuple=True)[0]
        other = (y == 1 - class_id).nonzero(as_tuple=True)[0]
        if same.numel() < 2 or other.numel() == 0:
            continue
        mask = anchor_label == class_id
        own_position = torch.searchsorted(same, index[mask])
        draw = torch.randint(same.numel() - 1, own_position.shape, device=f.device)
        positive[mask] = same[draw + (draw >= own_position)]
        negative[mask] = other[torch.randint(other.numel(), draw.shape, device=f.device)]
        eligible |= mask
    if not eligible.any():
        return f.sum() * 0.0
    anchor = f[index[eligible]]
    pos_dist = (anchor - f[positive[eligible]]).square().sum(dim=-1)
    neg_dist = (anchor - f[negative[eligible]]).square().sum(dim=-1)
    return torch.relu(pos_dist - neg_dist + margin).mean()
