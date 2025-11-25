#!/usr/bin/env python3
# main_sam.py
#
# FastSAM over renders/view_*.png
#
# For each input render:
#   - Runs FastSAM "everything" segmentation
#   - Saves masks as:
#       masks/masks_view_XX.npy
#   - Saves debug images as:
#       mask_debug/view_XX_masks_only.png     (binary mask of all segments)
#       mask_debug/view_XX_masks_overlay.png  (green overlay on original RGB)

from pathlib import Path
import sys
import argparse
import numpy as np
import cv2
import torch

# ---- Config ----
DEVICE = "cuda"            # Preferred device; falls back to CPU if CUDA is unavailable
IMG_GLOB = "view_*.png"    # Pattern for input render filenames


def ensure_uint8_mask(m):
    """
    Convert an arbitrary mask array to a uint8 binary mask in {0, 1}.

    Args:
        m: Any array-like mask (boolean, float, etc).

    Returns:
        np.uint8 mask of the same spatial shape, with 1 for foreground, 0 for background.
    """
    m = np.asarray(m)
    return (m > 0).astype(np.uint8)


def save_mask_debug(img_bgr, masks_list, out_base: Path):
    """
    Save simple debug visualizations of the segmentation result.

    Outputs:
      - <out_base>_masks_only.png: all mask pixels collapsed into a single binary map.
      - <out_base>_masks_overlay.png: overlay of all masks on top of the input RGB.

    Args:
        img_bgr: Original BGR image as loaded by OpenCV (H, W, 3).
        masks_list: List of per-instance uint8 masks (H, W), values in {0, 1}.
        out_base: Path *without* extension; the basename is reused for both outputs.
    """
    H, W = img_bgr.shape[:2]

    # Create a label map where each pixel stores the index of the mask that covers it.
    # If multiple masks overlap, later masks overwrite earlier ones.
    label_map = np.zeros((H, W), dtype=np.int32)
    for i, m in enumerate(masks_list):
        label_map[m > 0] = i + 1  # +1 so background stays 0

    # (1) Binary masks-only view (white where any mask is present)
    only = (label_map > 0).astype(np.uint8) * 255
    cv2.imwrite(str(out_base.parent / f"{out_base.name}_masks_only.png"), only)

    # (2) Overlay view: original RGB with a semi-transparent green overlay where masks exist
    ov = img_bgr.copy()
    nz = (label_map > 0)
    if np.any(nz):
        alpha = 0.45
        color = np.zeros_like(ov)
        color[nz] = (40, 220, 80)  # bright-ish green overlay
        ov = (
            ov.astype(np.float32) * (1.0 - alpha)
            + color.astype(np.float32) * alpha
        ).astype(np.uint8)

    cv2.imwrite(str(out_base.parent / f"{out_base.name}_masks_overlay.png"), ov)


def main():
    """
    Entry point for running FastSAM on a directory of renders.

    Pipeline:
      1. Parse CLI arguments and resolve directories.
      2. Load FastSAM (weights from ./FastSAM/weights/FastSAM.pt).
      3. For each view_*.png:
         - Run FastSAM to get "everything" masks.
         - Extract masks either from results.masks.data or from annotations.
         - Save masks_<view>.npy and debug visualizations.
    """
    ROOT = Path(__file__).resolve().parent

    # ----- CLI (all required) -----
    ap = argparse.ArgumentParser(description="Run FastSAM over view_*.png renders.")
    ap.add_argument(
        "--renders_dir",
        type=str,
        required=True,
        help="Directory containing input render PNGs (e.g., view_*.png).",
    )
    ap.add_argument(
        "--masks_dir",
        type=str,
        required=True,
        help="Directory to save .npy masks (masks_view_XX.npy).",
    )
    ap.add_argument(
        "--debug_dir",
        type=str,
        required=True,
        help="Directory to save debug PNGs (overlay / masks-only).",
    )
    args = ap.parse_args()

    def resolve_dir(path_str: str) -> Path:
        """
        Resolve a possibly relative directory path against this script's root.

        Args:
            path_str: Raw path string from CLI.

        Returns:
            Absolute Path pointing to the intended directory.
        """
        p = Path(path_str)
        return p if p.is_absolute() else (ROOT / p)

    # Resolve all directories relative to this script
    RENDERS = resolve_dir(args.renders_dir)
    MASKS_DIR = resolve_dir(args.masks_dir)
    DEBUG_DIR = resolve_dir(args.debug_dir)

    # Ensure output directories exist
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Renders dir: {RENDERS}")
    print(f"[INFO] Masks dir:   {MASKS_DIR}")
    print(f"[INFO] Debug dir:   {DEBUG_DIR}")

    # Absolute path to FastSAM weights; we don't rely on chdir()
    FASTSAM_WEIGHTS = (ROOT / "FastSAM" / "weights" / "FastSAM.pt").resolve()
    if not FASTSAM_WEIGHTS.exists():
        raise FileNotFoundError(f"FastSAM weights not found: {FASTSAM_WEIGHTS}")

    # Import FastSAM from its submodule without changing current working directory
    sys.path.insert(0, str((ROOT / "FastSAM").resolve()))
    from fastsam import FastSAM, FastSAMPrompt  # type: ignore

    # Pick device: prefer CUDA, but fall back to CPU if unavailable
    dev = "cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"

    # Initialize FastSAM model
    fsam = FastSAM(str(FASTSAM_WEIGHTS))

    # Inference parameters:
    #   - retina_masks: high-resolution masks
    #   - imgsz: inference resolution
    #   - conf: confidence threshold
    #   - iou: IoU threshold for NMS / post-processing
    infer_kwargs = dict(device=dev, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

    # Collect all input renders matching the glob pattern
    images = sorted(RENDERS.glob(IMG_GLOB))
    if not images:
        print(f"[ERR] No images like {IMG_GLOB} in {RENDERS}")
        return

    # Process all renders
    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] Unreadable image: {img_path}")
            continue
        H, W = img_bgr.shape[:2]

        # ----------------------------
        # 1) Run FastSAM segmentation
        # ----------------------------
        results = fsam(str(img_path), **infer_kwargs)
        prompt = FastSAMPrompt(str(img_path), results, device=dev)
        anns = prompt.everything_prompt()  # get segmentation for "everything" in the image

        masks_list = []

        # ----------------------------
        # 2A) Preferred path: ultralytics-like results.masks.data
        # ----------------------------
        try:
            # Some FastSAM variants wrap results in a list-like structure
            res0 = results[0] if isinstance(results, (list, tuple)) else results
            data = getattr(getattr(res0, "masks", None), "data", None)

            if data is not None:
                # data is typically (N, H, W) or similar tensor on device
                mnp = data.detach().cpu().numpy()

                for i in range(mnp.shape[0]):
                    mi = ensure_uint8_mask(mnp[i])
                    # Resize to original image size if needed
                    if mi.shape != (H, W):
                        mi = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST)
                    masks_list.append(mi)
        except Exception as e:
            # If structure isn't as expected, we fall back to annotation-based parsing
            print(f"[WARN] masks.data read failed: {e}")

        # ----------------------------
        # 2B) Fallback path: parse masks from annotations (anns)
        # ----------------------------
        if not masks_list and anns is not None:
            # anns can be a list of dictionaries or a single dict-like annotation
            for a in (anns if isinstance(anns, (list, tuple)) else [anns]):
                if isinstance(a, dict):
                    # Some implementations use 'mask', others 'segmentation'
                    mm = a.get("mask") if a.get("mask") is not None else a.get("segmentation")
                    if mm is not None:
                        mm = ensure_uint8_mask(mm)
                        if mm.shape != (H, W):
                            mm = cv2.resize(mm, (W, H), interpolation=cv2.INTER_NEAREST)
                        masks_list.append(mm)

        # If no masks were found at all, skip this image
        if not masks_list:
            print(f"[WARN] No masks for {img_path.name}")
            continue

        # ----------------------------
        # 3) Save masks as .npy
        # ----------------------------
        out_npy = MASKS_DIR / f"masks_{img_path.stem}.npy"
        try:
            # Preferred: dense (M, H, W) uint8 array
            np.save(out_npy, np.stack(masks_list, axis=0).astype(np.uint8))
        except Exception:
            # Fallback: object array if shapes are inconsistent
            np.save(out_npy, np.array(masks_list, dtype=object))
        print(f"[OK] masks -> {out_npy} (M={len(masks_list)})")

        # ----------------------------
        # 4) Save debug visualizations
        # ----------------------------
        save_mask_debug(img_bgr, masks_list, DEBUG_DIR / img_path.stem)


if __name__ == "__main__":
    main()
