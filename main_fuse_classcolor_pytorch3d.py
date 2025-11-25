#!/usr/bin/env python3
# main_fuse_classcolor_pytorch3d.py
# Class voting using PyTorch3D rasterizer.
# Pipeline:
#   - Load mesh with PyTorch3D
#   - Build cameras from camera_params.json (R/T + vertical FoV or intrinsics K)
#   - Rasterize mesh to get per-pixel face ids (pix_to_face)
#   - For each view, accumulate class votes over faces using masks_view_XX.npy
#   - Reduce votes to a single class per face via "paint in class order"
#   - Save per-face class strings (face_class_strings.npy)
#   - Optionally write a colored OBJ and print a color legend

import os
import re
import json
import argparse
from typing import Optional, Tuple, List, Dict

import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
import cv2
import torch

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PerspectiveCameras,  
    RasterizationSettings,
    MeshRasterizer,
)

# ---------- Utilities ----------


def find_view_index_from_any(name: str) -> int:
    """
    Extract the integer view index from a filename like 'view_07.png' or 'masks_view_07.npy'.

    Expected pattern: 'view_<digits>'.
    Raises ValueError if no such pattern exists.
    """
    m = re.search(r"view_(\d+)", name)
    if not m:
        raise ValueError(f"Cannot parse view index from filename: {name}")
    return int(m.group(1))


def _natural_key(s: str):
    """
    Generate a key for "natural" sorting of strings with embedded numbers.

    Example: view_2, view_10 → [2, ''], [10, ''] so 2 < 10 numerically.
    """
    parts = re.findall(r'\d+|\D+', str(s))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _expand_split_RT_format(cam_params: Dict) -> Optional[List[Dict]]:
    """
    Handle a compact camera JSON format where R and T are top-level lists:

        {
          "R": [...], "T": [...],
          "intrinsics": {fx, fy, cx, cy} or fov_deg_vertical / fov
        }

    Returns a list of per-view dicts:
        [{"R":..., "T":..., "use_fov":bool, "fov_deg":float|None, "K":3x3 or None}, ...]

    If the format doesn't match, returns None (caller tries other formats).
    """
    if not isinstance(cam_params, dict):
        return None

    if "R" in cam_params and "T" in cam_params:
        Rs, Ts = cam_params["R"], cam_params["T"]
        # Require parallel lists for R and T
        if isinstance(Rs, list) and isinstance(Ts, list) and len(Rs) == len(Ts):
            intr = cam_params.get("intrinsics", {})
            fx, fy = intr.get("fx", None), intr.get("fy", None)
            cx, cy = intr.get("cx", None), intr.get("cy", None)

            use_fov = False
            fov_deg = None
            K = None

            # Prefer explicit intrinsics if all present
            if all(v is not None for v in (fx, fy, cx, cy)):
                K = np.array(
                    [
                        [fx, 0.0, cx],
                        [0.0, fy, cy],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )
            else:
                # Fallback to FOV if provided
                fov_deg = cam_params.get("fov_deg_vertical", cam_params.get("fov", None))
                if fov_deg is not None:
                    use_fov = True
                    fov_deg = float(fov_deg)

            out = []
            for i in range(len(Rs)):
                out.append(
                    dict(
                        R=np.asarray(Rs[i], dtype=np.float32),
                        T=np.asarray(Ts[i], dtype=np.float32),
                        use_fov=use_fov,
                        fov_deg=fov_deg if use_fov else None,
                        K=None if use_fov else K,
                    )
                )
            return out

    return None


def _extract_cam_entries(cam_params) -> List[Dict]:
    """
    Normalize camera_params JSON to a list of per-view dicts.

    Supported shapes:
      - Expanded R/T list format (handled by _expand_split_RT_format)
      - Top-level list of camera dicts
      - Dict with 'cameras' / 'views' / 'frames' as a list
      - Dict-of-dicts keyed by arbitrary names (sorted by natural key)
    """
    # 1) Try the compact R/T split format first
    expanded = _expand_split_RT_format(cam_params)
    if expanded is not None:
        return expanded

    # 2) If it's already a list, assume it's a list of camera dicts
    if isinstance(cam_params, list):
        return cam_params

    # 3) If dict, look for common container keys or nested dicts
    if isinstance(cam_params, dict):
        # Common keys used in various pipelines
        for k in ("cameras", "views", "frames"):
            v = cam_params.get(k, None)
            if isinstance(v, list):
                return v

        # Fallback: treat values that are dicts as separate camera entries
        items = [(k, v) for k, v in cam_params.items() if isinstance(v, dict)]
        items.sort(key=lambda kv: _natural_key(kv[0]))
        if items:
            return [v for _, v in items]

    raise TypeError(f"Unsupported camera_params JSON type/shape: {type(cam_params).__name__}")


def build_cameras_from_json(cam_params, image_size_for_fov: int) -> List[Dict]:
    """
    Parse camera JSON into a list of standard camera dicts.

    Returns list of dicts with fields:
        {
          "R": 3x3 float32,
          "T": 3-vector float32,
          "use_fov": bool,
          "fov_deg": float | None,
          "K": 3x3 float32 | None
        }

    If K is absent, we fall back to a per-view vertical FoV.
    """
    entries = _extract_cam_entries(cam_params)
    cams: List[Dict] = []

    for idx, c in enumerate(entries):
        # Rotation and translation; accept different key variants
        R = np.asarray(c.get("R", c.get("rotation")), dtype=np.float32)
        T = np.asarray(c.get("T", c.get("translation")), dtype=np.float32)

        # Normalize singleton batch dimensions if present
        if R.ndim == 3 and R.shape[0] == 1:
            R = R[0]
        if T.ndim == 2 and T.shape[0] == 1:
            T = T[0]

        K = None
        use_fov = False
        fov_deg: Optional[float] = None

        if "intrinsics" in c and isinstance(c["intrinsics"], (list, tuple, np.ndarray)):
            K = np.asarray(c["intrinsics"], dtype=np.float32)
        elif "K" in c and c["K"] is not None:
            K = np.asarray(c["K"], dtype=np.float32)
        elif "intrinsics" in c and isinstance(c["intrinsics"], dict):
            intr = c["intrinsics"]
            fx, fy, cx, cy = [intr.get(k, None) for k in ("fx", "fy", "cx", "cy")]
            if all(v is not None for v in (fx, fy, cx, cy)):
                K = np.array(
                    [
                        [fx, 0.0, cx],
                        [0.0, fy, cy],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )

        # If K is missing, fall back to FoV; default to 60 degrees if absent
        if K is None:
            fov_deg = float(c.get("fov_deg_vertical", c.get("fov", 60.0)))
            use_fov = True

        cams.append(dict(R=R, T=T, use_fov=use_fov, fov_deg=fov_deg, K=K))

    return cams


def derive_fx_fy_from_fov(image_h: int, image_w: int, fov_deg: float) -> Tuple[float, float]:
    """
    Derive focal lengths (fx, fy) from vertical FoV assuming square-ish FOV.

    If the image is not square, we take the max dimension to avoid
    accidentally cropping diagonals.
    """
    H_for_fov = image_h if image_h == image_w else max(image_h, image_w)
    fy = (H_for_fov * 0.5) / np.tan(np.deg2rad(fov_deg * 0.5))
    fx = fy
    return float(fx), float(fy)


def safe_erode_mask(mask_uint8: np.ndarray, px: int) -> np.ndarray:
    """
    Erode a binary mask by 'px' pixels using a 3x3 elliptical kernel, but never
    erode it completely away.

    If 'px' <= 0 or the mask is empty, returns the original binary mask.
    This is used to shrink mask borders before voting to reduce bleeding
    over neighboring geometry.
    """
    m = (mask_uint8 > 0).astype(np.uint8)
    if px <= 0 or m.sum() == 0:
        return m

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cur = m
    for _ in range(px):
        nxt = cv2.erode(cur, kernel, iterations=1)
        # If the next erosion would delete everything, stop early
        if nxt.sum() == 0:
            return cur
        cur = nxt
    return cur


# ---------- OBJ writer ----------


def save_colored_obj_compat(
    verts: np.ndarray,
    faces: np.ndarray,
    verts_rgb: np.ndarray,
    out_path: str,
):
    """
    Write an OBJ file with per-vertex colors appended to each 'v' line:

        v x y z r g b

    Colors are assumed to be in [0,1]. Faces are indexed as (i,j,k) zero-based
    and are converted to 1-based indices for OBJ.
    """
    with open(out_path, "w") as f:
        f.write("# OBJ with per-vertex colors\n")
        for (x, y, z), (r, g, b) in zip(verts, verts_rgb):
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
        for (i, j, k) in faces:
            f.write(f"f {i+1} {j+1} {k+1}\n")
    print(f"[INFO] Wrote colored mesh: {out_path}")


# ---------- Named-color helper ----------

# A compact set of named colors (RGB in 0..255) for legend labeling.
_NAMED_COLORS = {
    # grays / white / black
    "black": (0, 0, 0),
    "dim gray": (105, 105, 105),
    "gray": (128, 128, 128),
    "silver": (192, 192, 192),
    "gainsboro": (220, 220, 220),
    "white": (255, 255, 255),

    # reds / pinks
    "dark red": (139, 0, 0),
    "firebrick": (178, 34, 34),
    "crimson": (220, 20, 60),
    "red": (255, 0, 0),
    "tomato": (255, 99, 71),
    "orangered": (255, 69, 0),
    "salmon": (250, 128, 114),
    "coral": (255, 127, 80),
    "deeppink": (255, 20, 147),
    "hot pink": (255, 105, 180),
    "pink": (255, 192, 203),
    "light pink": (255, 182, 193),
    "fuchsia": (255, 0, 255),
    "magenta": (255, 0, 255),
    "medium violet red": (199, 21, 133),
    "pale violet red": (219, 112, 147),
    "cerise": (222, 49, 99),

    # oranges / browns
    "dark orange": (255, 140, 0),
    "orange": (255, 165, 0),
    "gold": (255, 215, 0),
    "goldenrod": (218, 165, 32),
    "dark goldenrod": (184, 134, 11),
    "peru": (205, 133, 63),
    "sienna": (160, 82, 45),
    "chocolate": (210, 105, 30),
    "sandy brown": (244, 164, 96),
    "burlywood": (222, 184, 135),
    "tan": (210, 180, 140),
    "wheat": (245, 222, 179),
    "peach puff": (255, 218, 185),

    # yellows / yellow-greens
    "yellow": (255, 255, 0),
    "khaki": (240, 230, 140),
    "dark khaki": (189, 183, 107),
    "chartreuse": (127, 255, 0),
    "lawn green": (124, 252, 0),
    "lime": (0, 255, 0),
    "lime green": (50, 205, 50),
    "yellowgreen": (154, 205, 50),
    "olive": (128, 128, 0),
    "olivedrab": (107, 142, 35),

    # greens / teals
    "green": (0, 128, 0),
    "forest green": (34, 139, 34),
    "seagreen": (46, 139, 87),
    "medium seagreen": (60, 179, 113),
    "spring green": (0, 255, 127),
    "medium spring green": (0, 250, 154),
    "light sea green": (32, 178, 170),
    "teal": (0, 128, 128),
    "dark cyan": (0, 139, 139),

    # cyans / aquas
    "aqua": (0, 255, 255),
    "cyan": (0, 255, 255),
    "turquoise": (64, 224, 208),
    "medium turquoise": (72, 209, 204),
    "cadet blue": (95, 158, 160),
    "aquamarine": (127, 255, 212),

    # blues
    "navy": (0, 0, 128),
    "blue": (0, 0, 255),
    "medium blue": (0, 0, 205),
    "royal blue": (65, 105, 225),
    "slate blue": (106, 90, 205),
    "indigo": (75, 0, 130),
    "dodger blue": (30, 144, 255),
    "deepsky blue": (0, 191, 255),
    "sky blue": (135, 206, 235),
    "light sky blue": (135, 206, 250),
    "steel blue": (70, 130, 180),
    "cornflower blue": (100, 149, 237),

    # purples / violets
    "purple": (128, 0, 128),
    "rebecca purple": (102, 51, 153),
    "dark violet": (148, 0, 211),
    "violet": (238, 130, 238),
    "plum": (221, 160, 221),
    "thistle": (216, 191, 216),
    "orchid": (218, 112, 214),

    # lavenders
    "lavender": (230, 230, 250),
    "medium slate blue": (123, 104, 238),

    # extras
    "maroon": (128, 0, 0),
    "brown": (165, 42, 42),
    "rosybrown": (188, 143, 143),
}

# Precompute normalized RGB and names for fast nearest-color lookup
_named_names = list(_NAMED_COLORS.keys())
_named_rgb = np.array(
    [np.array(v, dtype=np.float32) / 255.0 for v in _NAMED_COLORS.values()],
    dtype=np.float32,
)


def _nearest_color_name(rgb_01: np.ndarray) -> str:
    """
    Return the nearest named color for an RGB triplet in [0,1] via L2 distance
    to the small named color set above.
    """
    v = np.asarray(rgb_01, dtype=np.float32).reshape(1, 3)
    d2 = np.sum((_named_rgb - v) ** 2, axis=1)
    idx = int(np.argmin(d2))
    return _named_names[idx]


# ---------- Palette ----------


def _make_distinct_palette(n: int) -> np.ndarray:
    """
    Create n distinct RGB colors in [0,1] using HSV spacing with golden ratio
    stepping on hue.

    This is deterministic and doesn't require external libraries, and gives
    reasonably well-separated colors up to hundreds of classes.
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    phi = (1 + 5**0.5) / 2
    hues = [(i / phi) % 1.0 for i in range(n)]
    sats = np.linspace(0.65, 0.95, num=n)  # vary saturation slightly
    vals = np.linspace(0.85, 1.0, num=n)   # vary value slightly

    def hsv2rgb(h, s, v):
        """
        Minimal HSV → RGB conversion used for palette generation.
        h, s, v in [0,1].
        """
        h6 = h * 6.0
        i = int(h6) % 6
        f = h6 - int(h6)
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        return r, g, b

    rgb = np.zeros((n, 3), dtype=np.float32)
    for i, (h, s, v) in enumerate(zip(hues, sats, vals)):
        rgb[i] = hsv2rgb(h, float(s), float(v))
    return rgb


# ---------- SUM Parts fixed palette (0–255 RGB, will normalize to [0,1]) ----------

# Fixed colors per SUM-Parts class name (used when --sumparts is enabled).
SUMPARTS_CLASS_COLORS = {
    "trees":              (2, 255, 0),
    "water":              (2, 255, 255),
    "car":                (255, 0, 255),
    "boat":               (0, 0, 153),
    "roof":               (85, 85, 127),
    "building":           (255, 255, 5),
    "chimney":            (255, 51, 50),
    "window":             (255, 255, 5),
    "door":               (255, 255, 5),
    "grass":              (2, 255, 0),
    "road":               (170, 86, 1),
    "road marking":       (170, 86, 1),
    "bridge":             (170, 86, 1),
    "dormer":             (106, 0, 159),
    "balcony":            (50, 126, 151),
    "roof_installations": (77, 0, 77),
}


# ---------- Main ----------

def main():
    # CLI argument parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True, help="Path to mesh (.obj)")
    ap.add_argument("--renders", required=True, help="Dir with view_XX.png and camera_params.json")
    ap.add_argument("--masks_dir", required=True, help="Dir with labeled masks (masks_view_XX.npy)")
    ap.add_argument("--image_size", type=int, default=1024, help="Fallback raster size (if RGB not found)")
    ap.add_argument("--out_labels", default="face_class_strings.npy",
                    help="Output path for per-face class strings (.npy, length=F)")
    ap.add_argument("--out_colored_obj", default="",
                    help="Output path for colored OBJ (per-vertex RGB). Empty disables output.")
    ap.add_argument(
        "--erode_px",
        type=int,
        default=2,
        help="Shrink mask borders before voting (reduces bleeding across geometry boundaries).",
    )
    ap.add_argument(
        "--order_metric",
        choices=["votes", "faces"],
        default="votes",
        help=(
            "Class painting order metric: "
            "'votes' = total pixel votes per class; "
            "'faces' = number of faces with any support."
        ),
    )
    ap.add_argument(
        "--sumparts",
        action="store_true",
        help="Use fixed SUM Parts class colors for coloring/legend instead of random palette.",
    )
    args = ap.parse_args()

    # Select CUDA if available; otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Torch device: {device.type}")
    print("[INFO] Rasterizer: PyTorch3D (bin_size=0 naive)")

    # ----------------------------------------------------------------------
    # 1) Load mesh
    # ----------------------------------------------------------------------
    meshes = load_objs_as_meshes([args.obj], device=device)
    verts = meshes.verts_packed().detach().cpu().numpy()
    faces = meshes.faces_packed().detach().cpu().numpy()
    V, F = verts.shape[0], faces.shape[0]
    print(f"[INFO] Mesh: V={V}, F={F}")

    # ----------------------------------------------------------------------
    # 2) Load camera parameters and infer image size
    # ----------------------------------------------------------------------
    cam_json = os.path.join(args.renders, "camera_params.json")
    with open(cam_json, "r") as f:
        cam_params = json.load(f)

    # Collect mask files for all views (masks_view_XX.npy)
    mask_files = sorted(
        [
            f
            for f in os.listdir(args.masks_dir)
            if f.startswith("masks_view_") and f.endswith(".npy")
        ],
        key=_natural_key,
    )
    if not mask_files:
        raise FileNotFoundError(f"No masks_view_XX.npy found in {args.masks_dir}")

    # Infer image size from the first view's RGB if available
    first_view_idx = find_view_index_from_any(mask_files[0])
    rgb_path0 = os.path.join(args.renders, f"view_{first_view_idx:02d}.png")
    if os.path.isfile(rgb_path0):
        rgb0 = imageio.imread(rgb_path0)
        H0, W0 = rgb0.shape[:2]
        image_h, image_w = H0, W0
    else:
        image_h = image_w = args.image_size

    # Build camera specs (R/T + either FoV or K) for each view
    cameras_spec = build_cameras_from_json(cam_params, max(image_h, image_w))

    # ----------------------------------------------------------------------
    # 3) Configure rasterizer
    # ----------------------------------------------------------------------
    try:
        rast_settings = RasterizationSettings(
            image_size=(image_h, image_w),
            faces_per_pixel=1,
            blur_radius=0.0,
            cull_backfaces=False,
            bin_size=0,            # naive mode: single bin
            max_faces_per_bin=0,   # allow unlimited faces per bin
        )
    except TypeError:
        rast_settings = RasterizationSettings(
            image_size=(image_h, image_w),
            faces_per_pixel=1,
            blur_radius=0.0,
            cull_backfaces=False,
            bin_size=0,
        )

    # Mapping from class name -> row index in class_votes
    class_to_idx: Dict[str, int] = {}
    # Reverse mapping (index -> class name)
    idx_to_class: List[str] = []
    # Accumulator for votes: shape (C, F). Each row is one class, each column a face.
    class_votes = None

    # ----------------------------------------------------------------------
    # 4) For each view: rasterize and accumulate votes per face
    # ----------------------------------------------------------------------
    for mask_file in tqdm(
        mask_files,
        desc="[INFO] Backprojecting (PyTorch3D rasterizer + GPU votes)",
    ):
        # Derive view index from filename (e.g. masks_view_03.npy -> 3)
        view_idx = find_view_index_from_any(mask_file)
        if view_idx >= len(cameras_spec):
            # Skip if camera_params.json doesn't have a matching entry
            continue

        cam = cameras_spec[view_idx]
        R = torch.from_numpy(cam["R"]).float().to(device)[None, ...]
        T = torch.from_numpy(cam["T"]).float().to(device)[None, ...]

        # Build FoV-based camera for PyTorch3D.
        # If K is present, approximate vertical FoV from fy and image height.
        if cam["use_fov"]:
            cameras = FoVPerspectiveCameras(
                device=device,
                R=R,
                T=T,
                fov=float(cam["fov_deg"]),
            )
        else:
            K = cam["K"].astype(np.float32)
            fx, fy = float(K[0, 0]), float(K[1, 1])
            fov_v = float(
                np.rad2deg(
                    2.0 * np.arctan((image_h * 0.5) / max(fy, 1e-6))
                )
            )
            cameras = FoVPerspectiveCameras(
                device=device,
                R=R,
                T=T,
                fov=fov_v,
            )

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=rast_settings,
        )

        # Rasterize once for this view: get pix_to_face (per-pixel face id)
        with torch.no_grad():
            frags = rasterizer(meshes)
        pix_to_face = frags.pix_to_face[0]          # (H, W, faces_per_pixel=1)
        face_ids = pix_to_face[..., 0].contiguous() # (H, W), -1 denotes "no face"

        # Load labeled masks for this view. Each element is a dict:
        #   { "mask": HxW uint8/bool, "label": str, "matched_box": [x1,y1,x2,y2], "score": float }
        arr = np.load(os.path.join(args.masks_dir, mask_file), allow_pickle=True)
        labeled = [x.item() if hasattr(x, "item") else x for x in arr]

        # Valid pixels are those that hit some face (exclude background)
        valid_t = (face_ids >= 0)

        # Iterate over each labeled region for this view and accumulate votes
        for rec in labeled:
            m = np.asarray(rec.get("mask", None))
            lbl = (rec.get("label") or "unknown").strip().lower()
            if lbl == "unknown" or m is None:
                # Skip unlabeled or invalid records
                continue

            # Binarize mask and safely erode to tighten borders
            m_u8 = (m.astype(np.uint8) > 0).astype(np.uint8)
            mask_bin = safe_erode_mask(m_u8, int(args.erode_px))
            mask_t = torch.from_numpy(mask_bin.astype(bool)).to(device)

            # Pixels that both hit a face and are inside this class mask
            sel_t = valid_t & mask_t
            if not torch.any(sel_t):
                continue

            # Ensure the class has a row in class_votes
            if lbl not in class_to_idx:
                class_to_idx[lbl] = len(idx_to_class)
                idx_to_class.append(lbl)
                new_votes = torch.zeros((1, F), dtype=torch.int64, device=device)
                class_votes = (
                    new_votes
                    if class_votes is None
                    else torch.cat([class_votes, new_votes], dim=0)
                )
            ci = class_to_idx[lbl]

            # For all selected pixels, collect face indices and increment class_votes
            f_sel = face_ids[sel_t].to(torch.int64)  # face indices in [0, F)
            class_votes[ci] += torch.bincount(f_sel, minlength=F)

    # ----------------------------------------------------------------------
    # 5) Reduce votes to a single class per face via "paint in class order"
    # ----------------------------------------------------------------------
    # Default: all faces start as "unknown" (no class)
    face_class_idx = np.full(F, -1, dtype=np.int32)

    order = []
    class_totals = None
    if class_votes is not None:
        # Summarize each class's support over faces according to chosen metric
        if args.order_metric == "faces":
            # Number of faces with any support
            class_totals = (class_votes > 0).sum(dim=1)
        else:
            # Total vote count (default)
            class_totals = class_votes.sum(dim=1)

        # Sort classes by descending total support
        order = torch.argsort(class_totals, descending=True)

        # Start with all faces unlabeled, then "paint" classes in order
        face_class_idx_t = torch.full(
            (F,),
            -1,
            dtype=torch.int64,
            device=class_votes.device,
        )
        for ci_t in order:
            ci = int(ci_t.item())
            has_support = class_votes[ci] > 0
            # Overwrite previous class for faces where this class has votes
            face_class_idx_t[has_support] = ci
        face_class_idx = face_class_idx_t.detach().cpu().numpy()

    # Convert per-face class indices to strings (or "unknown")
    face_class_strings = np.array(
        [idx_to_class[i] if i >= 0 else "unknown" for i in face_class_idx],
        dtype=object,
    )
    np.save(args.out_labels, face_class_strings)
    print(f"[OK] Saved per-face classes (paint-order reduction): {args.out_labels}")

    # ----------------------------------------------------------------------
    # 6) Optional: colored OBJ + legend
    # ----------------------------------------------------------------------
    if args.out_colored_obj:
        C = len(idx_to_class)

        # Choose palette:
        # - if --sumparts: use fixed SUM-Parts colors per class name
        # - otherwise: generate a distinct palette procedurally
        if args.sumparts:
            palette = np.zeros((C, 3), dtype=np.float32)
            for ci, cname in enumerate(idx_to_class):
                key = cname.lower()
                if key in SUMPARTS_CLASS_COLORS:
                    # Normalize RGB from [0,255] to [0,1]
                    palette[ci] = (
                        np.array(SUMPARTS_CLASS_COLORS[key], dtype=np.float32) / 255.0
                    )
                else:
                    # Fallback for non-SUM-Parts classes
                    palette[ci] = np.array([0.6, 0.6, 0.6], dtype=np.float32)
                    print(f"[WARN] No SUM Parts color for class '{cname}', using gray.")
        else:
            # Random-ish but deterministic palette
            palette = _make_distinct_palette(C)  # (C,3) in [0,1]

        unknown_rgb = np.array([0.6, 0.6, 0.6], dtype=np.float32)

        # Assign face colors from palette or unknown gray
        face_colors = np.zeros((F, 3), dtype=np.float32)
        for fi in range(F):
            ci = face_class_idx[fi]
            face_colors[fi] = unknown_rgb if ci == -1 else palette[ci]

        # Compute per-vertex colors as mean of incident face colors
        V = verts.shape[0]
        verts_rgb = np.zeros((V, 3), dtype=np.float32)
        counts = np.zeros((V,), dtype=np.float32)
        for k in range(3):
            v_ids = faces[:, k]
            np.add.at(verts_rgb, v_ids, face_colors)
            np.add.at(counts, v_ids, 1.0)
        counts = np.maximum(counts, 1.0)[:, None]  # avoid division by zero
        verts_rgb = verts_rgb / counts

        save_colored_obj_compat(verts, faces, verts_rgb, args.out_colored_obj)

        # Print legend: classes ordered by paint-order metric with approximate named colors
        print("\n[INFO] Class → Color legend (paint-order):")
        if class_totals is not None and len(order) > 0:
            totals_cpu = class_totals.detach().cpu().numpy()
            for rank, ci_t in enumerate(order.tolist(), 1):
                ci = int(ci_t)
                cname = idx_to_class[ci]
                total = float(totals_cpu[ci])
                rgb = palette[ci]
                cname_color = _nearest_color_name(rgb)
                print(
                    f"  {rank:2d}. {cname:15s} votes={int(total)} "
                    f"color={np.array2string(rgb, precision=6)}  ≈ {cname_color}"
                )
        else:
            # Fallback if class_totals/order weren't computed for some reason
            for ci, cname in enumerate(idx_to_class):
                rgb = palette[ci]
                cname_color = _nearest_color_name(rgb)
                print(
                    f"  {ci:2d}. {cname:15s} "
                    f"color={np.array2string(rgb, precision=6)}  ≈ {cname_color}"
                )
        print("  unknown         → [0.6, 0.6, 0.6]  ≈ gray")


if __name__ == "__main__":
    main()
