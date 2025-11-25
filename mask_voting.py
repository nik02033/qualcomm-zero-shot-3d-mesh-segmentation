#!/usr/bin/env python3
# GPU-accelerated version (PyTorch + Kornia) with the same logic as your original.
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import argparse

try:
    import kornia as K
    _HAS_KORNIA = True
except Exception:
    _HAS_KORNIA = False

# ---------- Primary strict rule (defaults; can be overridden via CLI) ----------
COVERAGE_REQ = 0.85               # fraction of mask that must lie inside box
AREA_LOW, AREA_HIGH = 0.85, 2.0   # ratio = box_area / mask_area allowed range

# ---------- Fallback 1: single-mask edge-hug (defaults; CLI override) ----------
HUG_EDGE_EPS_FRAC = 0.02                  # 2% of side length near edge
HUG_EDGE_EPS_MIN_PX = 3
HUG_SINGLE_MIN, HUG_SINGLE_MAX = 0.30, 1.00  # ratio = mask_in_box_area / box_area
INSIDE_TOL_FRAC = 0.05                    # <=5% of mask pixels may lie outside bbox

# ---------- Fallback 2: pair merge & edge-hug (bbox-first, last resort) ----------
MERGE_KERNEL = (3, 3)
MERGE_DILATE_ITER = 1
HUG_PAIR_MIN, HUG_PAIR_MAX = 0.30, 1.50   # ratio = (m1|m2)_in_box_area / box_area

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Vote FastSAM masks onto DINO boxes with GPU acceleration."
    )
    p.add_argument(
        "--renders_dir",
        type=str,
        required=True,
        help='Directory containing input render PNGs (e.g. "renders", matched by view_*.png).',
    )
    p.add_argument(
        "--dets_dir",
        type=str,
        required=True,
        help='Directory containing DINO dets_*.npy (e.g. "dino_dets").',
    )
    p.add_argument(
        "--ann_dir",
        type=str,
        required=True,
        help='Directory to save annotated images with matched masks.',
    )
    p.add_argument(
        "--labeled_dir",
        type=str,
        required=True,
        help='Directory to save labeled mask npy files (masks_*.npy).',
    )

    # ---- tunable matching parameters (with current values as defaults) ----
    p.add_argument(
        "--coverage_req",
        type=float,
        default=COVERAGE_REQ,
        help=f"Strict rule: minimum fraction of mask that must lie inside a bbox "
             f"(default: {COVERAGE_REQ}).",
    )
    p.add_argument(
        "--area_low",
        type=float,
        default=AREA_LOW,
        help=f"Strict rule: lower bound for box_area/mask_area (default: {AREA_LOW}).",
    )
    p.add_argument(
        "--area_high",
        type=float,
        default=AREA_HIGH,
        help=f"Strict rule: upper bound for box_area/mask_area (default: {AREA_HIGH}).",
    )
    p.add_argument(
        "--hug_single_min",
        type=float,
        default=HUG_SINGLE_MIN,
        help=f"Single-mask edge-hug fallback: min mask_in_box_area/box_area "
             f"(default: {HUG_SINGLE_MIN}).",
    )
    p.add_argument(
        "--hug_single_max",
        type=float,
        default=HUG_SINGLE_MAX,
        help=f"Single-mask edge-hug fallback: max mask_in_box_area/box_area "
             f"(default: {HUG_SINGLE_MAX}).",
    )
    p.add_argument(
        "--inside_tol_frac",
        type=float,
        default=INSIDE_TOL_FRAC,
        help=f"Single-mask edge-hug fallback: max fraction of mask allowed outside "
             f"the bbox (default: {INSIDE_TOL_FRAC}).",
    )

    return p.parse_args()


# -------------------- Tensor helpers --------------------
def _to_bin_t(mask_np: np.ndarray, device=DEVICE):
    """(H,W) uint8/0-1 -> (1,1,H,W) float32 {0,1} on device."""
    m = np.asarray(mask_np)
    if m.ndim > 2:
        m = np.squeeze(m)
    m = (m > 0).astype(np.uint8)
    t = torch.from_numpy(m).to(device=device, dtype=torch.float32)
    return t.unsqueeze(0).unsqueeze(0)


def _resize_nearest_t(t: torch.Tensor, H: int, W: int):
    """t: (1,1,h,w) -> (1,1,H,W)"""
    return F.interpolate(t, size=(H, W), mode="nearest")


def _sum_t(t: torch.Tensor) -> float:
    return float(t.sum().item())


def _crop_box(clamped_xyxy, H, W):
    x1, y1, x2, y2 = map(int, clamped_xyxy)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W - 1, x2); y2 = min(H - 1, y2)
    return x1, y1, x2, y2


def _bbox_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)


def _mask_area_in_box_t(mask_t: torch.Tensor, box) -> int:
    """mask_t: (1,1,H,W) binary float."""
    H, W = mask_t.shape[-2:]
    x1, y1, x2, y2 = _crop_box(box, H, W)
    if x2 < x1 or y2 < y1:
        return 0
    return int(mask_t[0, 0, y1:y2+1, x1:x2+1].sum().item())


def _mask_in_box_coverage_t(mask_t: torch.Tensor, box) -> float:
    """fraction of mask pixels that lie inside the bbox."""
    total = _sum_t(mask_t)
    if total <= 0.0:
        return 0.0
    inside = _mask_area_in_box_t(mask_t, box)
    return float(inside) / total


def _mask_bounds_t(mask_t: torch.Tensor):
    """Return tight bbox (x1,y1,x2,y2) of positive pixels. None if empty."""
    nz = torch.nonzero(mask_t[0, 0] > 0, as_tuple=False)
    if nz.numel() == 0:
        return None
    ys = nz[:, 0]; xs = nz[:, 1]
    x1 = int(xs.min().item()); x2 = int(xs.max().item())
    y1 = int(ys.min().item()); y2 = int(ys.max().item())
    return x1, y1, x2, y2


def _mask_fully_inside_box_t(mask_t: torch.Tensor, box) -> bool:
    """True if all positive pixels of mask lie inside the bbox (inclusive)."""
    bb = _mask_bounds_t(mask_t)
    if bb is None:
        return False
    mx1, my1, mx2, my2 = bb
    x1, y1, x2, y2 = map(int, box)
    return (mx1 >= x1) and (mx2 <= x2) and (my1 >= y1) and (my2 <= y2)


def _mask_mostly_inside_box_t(mask_t: torch.Tensor, box, tol_frac=INSIDE_TOL_FRAC) -> bool:
    """True if the fraction of mask pixels outside the bbox is <= tol_frac."""
    H, W = mask_t.shape[-2:]
    total = _sum_t(mask_t)
    if total <= 0.0:
        return False
    x1, y1, x2, y2 = _crop_box(box, H, W)
    inside = mask_t[0, 0, y1:y2+1, x1:x2+1].sum()
    outside = total - float(inside.item())
    return (outside / total) <= tol_frac


def _mask_hugs_opposite_edges_t(mask_t: torch.Tensor, box) -> bool:
    """Check if nonzero pixels inside the bbox touch both opposite edges in X or Y (with eps)."""
    H, W = mask_t.shape[-2:]
    x1, y1, x2, y2 = _crop_box(box, H, W)
    if x2 < x1 or y2 < y1:
        return False
    roi = mask_t[0, 0, y1:y2+1, x1:x2+1]  # (h,w)
    nz = torch.nonzero(roi > 0, as_tuple=False)
    if nz.numel() == 0:
        return False

    h = (y2 - y1 + 1)
    w = (x2 - x1 + 1)
    eps_x = max(HUG_EDGE_EPS_MIN_PX, HUG_EDGE_EPS_FRAC * w)
    eps_y = max(HUG_EDGE_EPS_MIN_PX, HUG_EDGE_EPS_FRAC * h)

    ys = nz[:, 0].float()
    xs = nz[:, 1].float()
    dx_left   = xs.min().item() - 0.0
    dx_right  = (w - 1) - xs.max().item()
    dy_top    = ys.min().item() - 0.0
    dy_bottom = (h - 1) - ys.max().item()

    hugs_x = (dx_left <= eps_x) and (dx_right <= eps_x)
    hugs_y = (dy_top  <= eps_y) and (dy_bottom <= eps_y)
    return bool(hugs_x or hugs_y)


def _dilated_overlap_t(m1: torch.Tensor, m2: torch.Tensor) -> bool:
    """Binary dilation via max-pool; test overlap after dilation."""
    kx, ky = MERGE_KERNEL
    assert kx == ky, "non-square kernels not supported in this fast path"
    pad = kx // 2
    x = m1
    y = m2
    for _ in range(MERGE_DILATE_ITER):
        x = F.max_pool2d(x, kernel_size=kx, stride=1, padding=pad)
        y = F.max_pool2d(y, kernel_size=kx, stride=1, padding=pad)
    return bool((x > 0).logical_and(y > 0).any().item())


# -------------------- I/O + conversion --------------------
def load_fastsam_masks(root: Path, stem: str, H: int, W: int, device=DEVICE):
    """Load masks_*.npy and return list of (1,1,H,W) float32 tensors on device with {0,1}."""
    p = (root / "masks" / f"masks_{stem}.npy")
    if not p.exists():
        return []
    arr = np.load(p, allow_pickle=True)
    masks_np = []
    if arr.dtype == object:
        it = arr
    elif arr.ndim == 3:
        it = [arr[i] for i in range(arr.shape[0])]
    else:
        it = [arr]
    for m in it:
        t = _to_bin_t(m, device=device)
        if t.shape[-2:] != (H, W):
            t = _resize_nearest_t(t, H, W)
        # Ensure strict binary {0,1}
        t = (t > 0.5).to(dtype=torch.float32)
        masks_np.append(t)
    return masks_np


def _addweighted_gpu(img_bgr: np.ndarray, mask_any_t: torch.Tensor, alpha=0.45) -> np.ndarray:
    """
    Overlay: green where mask_any>0, and draw red edges (Canny or Sobel fallback).
    All heavy ops on GPU, return BGR uint8 on CPU.
    """
    H, W = img_bgr.shape[:2]
    # to float tensor [0,1] on device
    img_t = torch.from_numpy(img_bgr).to(device=DEVICE, dtype=torch.float32) / 255.0  # (H,W,3) BGR
    img_t = img_t.permute(2, 0, 1).contiguous().unsqueeze(0)  # (1,3,H,W)

    # color layer (green)
    color = torch.zeros_like(img_t)
    color[:, 1, :, :] = (mask_any_t > 0).float()

    out = (1.0 - alpha) * img_t + alpha * color

    # edges (use Kornia Canny if available; otherwise Sobel fallback )
    if _HAS_KORNIA:
        # Kornia expects RGB float [0,1]
        rgb = img_t[:, [2, 1, 0], :, :]
        edges, _ = K.filters.canny(rgb, low_threshold=0.1, high_threshold=0.2)
        edges = (edges > 0.0).float()          # (1,1,H,W)
    else:
        # --- Sobel fallback on GPU ---
        # img_t is BGR in [0,1], shape (1,3,H,W)
        gray = (
            0.114 * img_t[:, 0:1, :, :] +   # B
            0.587 * img_t[:, 1:2, :, :] +   # G
            0.299 * img_t[:, 2:3, :, :]     # R
        )  # (1,1,H,W)

        # 3x3 Sobel kernels with correct conv2d shape: (out_ch=1, in_ch=1, kH=3, kW=3)
        sobel_x = torch.tensor(
            [[[[-1., 0., 1.],
               [-2., 0., 2.],
               [-1., 0., 1.]]]],
            device=DEVICE, dtype=img_t.dtype
        )
        sobel_y = torch.tensor(
            [[[[-1., -2., -1.],
               [ 0.,  0.,  0.],
               [ 1.,  2.,  1.]]]],
            device=DEVICE, dtype=img_t.dtype
        )

        gx = F.conv2d(gray, sobel_x, padding=1)  # (1,1,H,W)
        gy = F.conv2d(gray, sobel_y, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy)

        # Pick a reasonable threshold for edges (grayscale is [0,1])
        edges = (mag > 0.2).float()             # (1,1,H,W)

    # paint edges red
    out[:, 2, :, :] = torch.where(
        edges[:, 0] > 0,
        torch.tensor(1.0, device=DEVICE, dtype=out.dtype),
        out[:, 2, :, :],
    )
    out[:, 1, :, :] = torch.where(
        edges[:, 0] > 0,
        torch.tensor(0.0, device=DEVICE, dtype=out.dtype),
        out[:, 1, :, :],
    )
    out[:, 0, :, :] = torch.where(
        edges[:, 0] > 0,
        torch.tensor(0.0, device=DEVICE, dtype=out.dtype),
        out[:, 0, :, :],
    )

    out = (out.clamp(0, 1) * 255.0).byte().squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
    return out


def overlay_masks(img_bgr, masks_t, alpha=0.45):
    """masks_t: list of (1,1,H,W) float32 tensors on device."""
    if not masks_t:
        return img_bgr
    # union of masks
    union = torch.zeros(
        (1, 1, img_bgr.shape[0], img_bgr.shape[1]),
        dtype=torch.float32,
        device=DEVICE,
    )
    for t in masks_t:
        union = torch.logical_or(union > 0, t > 0).float()
    return _addweighted_gpu(img_bgr, union, alpha=alpha)


def main():
    global COVERAGE_REQ, AREA_LOW, AREA_HIGH, HUG_SINGLE_MIN, HUG_SINGLE_MAX, INSIDE_TOL_FRAC

    ROOT = Path(__file__).resolve().parent
    args = parse_args()

    # Override global thresholds from CLI (defaults ensure current behavior if not specified)
    COVERAGE_REQ   = float(args.coverage_req)
    AREA_LOW       = float(args.area_low)
    AREA_HIGH      = float(args.area_high)
    HUG_SINGLE_MIN = float(args.hug_single_min)
    HUG_SINGLE_MAX = float(args.hug_single_max)
    INSIDE_TOL_FRAC = float(args.inside_tol_frac)

    def resolve_dir(path_str: str) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else (ROOT / p)

    RENDERS = resolve_dir(args.renders_dir)
    OUT_DIR = resolve_dir(args.dets_dir)
    ANN_DIR = resolve_dir(args.ann_dir)
    LABELED_DIR = resolve_dir(args.labeled_dir)

    ANN_DIR.mkdir(exist_ok=True, parents=True)
    LABELED_DIR.mkdir(exist_ok=True, parents=True)

    print(f"[INFO] Renders dir:   {RENDERS}")
    print(f"[INFO] Dets dir:      {OUT_DIR}")
    print(f"[INFO] Annotated dir: {ANN_DIR}")
    print(f"[INFO] Labeled dir:   {LABELED_DIR}")
    print(f"[INFO] coverage_req={COVERAGE_REQ}, area_low={AREA_LOW}, area_high={AREA_HIGH}, "
          f"hug_single_min={HUG_SINGLE_MIN}, hug_single_max={HUG_SINGLE_MAX}, "
          f"inside_tol_frac={INSIDE_TOL_FRAC}")

    images = sorted(RENDERS.glob("view_*.png"))
    if not images:
        print(f"[ERR] No images in {RENDERS}/view_*.png")
        return

    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] unreadable {img_path}")
            continue
        H, W = img_bgr.shape[:2]
        stem = img_path.stem

        dets_path = OUT_DIR / f"dets_{stem}.npy"
        if not dets_path.exists():
            print(f"[WARN] missing dets: {dets_path}, skipping")
            continue
        dets = np.load(dets_path, allow_pickle=True).tolist()

        masks_t = load_fastsam_masks(ROOT, stem, H, W, device=DEVICE)
        if not masks_t:
            print(f"[INFO] no masks for {stem}, skipping voting")
            continue

        used_mask = set()
        labeled = []
        matched_masks_t = []

        for det in dets:
            box = det["bbox"]; box_area = float(_bbox_area(box))
            label = det.get("label", "unknown"); score = float(det.get("score", 0.0))

            # 1) STRICT ORIGINAL RULE
            best = None
            best_key = (1e9, 1e9, -1.0)
            for i, m in enumerate(masks_t):
                if i in used_mask:
                    continue
                mask_area = _sum_t(m)
                if mask_area <= 0.0:
                    continue
                cov = _mask_in_box_coverage_t(m, box)
                if cov < COVERAGE_REQ:
                    continue
                ratio = box_area / float(mask_area)
                if (ratio < AREA_LOW) or (ratio > AREA_HIGH):
                    continue
                key = (abs(ratio - 1.0), box_area, -score)
                if key < best_key:
                    best_key = key; best = (i, m)

            if best is not None:
                i, mb = best
                if label != "unknown":
                    np_mask = (mb[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                    labeled.append(
                        {"mask": np_mask, "label": label, "matched_box": box, "score": score}
                    )
                    matched_masks_t.append(mb); used_mask.add(i)
                continue

            # 2) SINGLE-MASK EDGE-HUG FALLBACK (with "mostly-inside" tolerance)
            hug_pick = None
            for i, m in enumerate(masks_t):
                if i in used_mask:
                    continue
                if not _mask_mostly_inside_box_t(m, box, tol_frac=INSIDE_TOL_FRAC):
                    continue
                area_in_box = float(_mask_area_in_box_t(m, box))
                if area_in_box <= 0:
                    continue
                ratio = area_in_box / box_area
                if (ratio < HUG_SINGLE_MIN) or (ratio > HUG_SINGLE_MAX):
                    continue
                if not _mask_hugs_opposite_edges_t(m, box):
                    continue
                hug_pick = (i, m); break

            if hug_pick is not None:
                i, mb = hug_pick
                if label != "unknown":
                    np_mask = (mb[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                    labeled.append(
                        {"mask": np_mask, "label": label, "matched_box": box, "score": score}
                    )
                    matched_masks_t.append(mb); used_mask.add(i)
                continue  # do not try pair-merge if single-mask works

            # 3) PAIR-MERGE EDGE-HUG FALLBACK (last resort)
            candidate_ids = [
                i for i, m in enumerate(masks_t)
                if (i not in used_mask) and _mask_fully_inside_box_t(m, box)
            ]

            best_pair = None
            best_pair_key = (1e9, -1.0)
            for a in range(len(candidate_ids)):
                for b in range(a + 1, len(candidate_ids)):
                    ia, ib = candidate_ids[a], candidate_ids[b]
                    m1, m2 = masks_t[ia], masks_t[ib]
                    if not _dilated_overlap_t(m1, m2):  # adjacency test
                        continue
                    comb = torch.logical_or(m1 > 0, m2 > 0).float()
                    if not _mask_hugs_opposite_edges_t(comb, box):
                        continue
                    area_in_box = float(_mask_area_in_box_t(comb, box))
                    if area_in_box <= 0:
                        continue
                    ratio = area_in_box / box_area
                    if (ratio < HUG_PAIR_MIN) or (ratio > HUG_PAIR_MAX):
                        continue
                    key = (abs(ratio - 1.0), area_in_box)
                    if key < best_pair_key:
                        best_pair_key = key
                        best_pair = (ia, ib, comb)

            if best_pair is not None:
                ia, ib, comb = best_pair
                if label != "unknown":
                    np_mask = (comb[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                    labeled.append(
                        {"mask": np_mask, "label": label, "matched_box": box, "score": score}
                    )
                    matched_masks_t.append(comb)
                    used_mask.add(ia); used_mask.add(ib)

        out_labeled = LABELED_DIR / f"masks_{stem}.npy"
        out_labeled.parent.mkdir(exist_ok=True, parents=True)
        np.save(out_labeled, np.array(labeled, dtype=object))
        print(f"[OK] labeled masks -> {out_labeled} (saved={len(labeled)})")

        annotated = overlay_masks(img_bgr, matched_masks_t, alpha=0.45)
        ANN_DIR.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(ANN_DIR / f"{stem}_matched_masks.jpg"), annotated)


if __name__ == "__main__":
    main()
