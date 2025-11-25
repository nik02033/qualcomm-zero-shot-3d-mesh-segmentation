#!/usr/bin/env python3
# gdino_infer.py
#
# GroundingDINO inference over renders/view_*.png.
#
# Two backends:
#   - gdino1  : local GroundingDINO v1 from Grounded-SAM-2 (batched prompts)
#   - gdino15 : GroundingDINO 1.5 Cloud API (DDS), no batching
#
# Saves per-view detections as dino_dets/dets_view_XX.npy
# Also saves a simple debug image drawing the saved detections:
#   dino_dets/annotated/view_XX_from_saved.jpg
#
# Classes are loaded from custom_classes.txt (one class name per line).
# To reduce hallucinations for the local model, classes are processed in
# batches of up to MAX_CLASSES_PER_PROMPT per forward pass, and detections
# from all batches are combined for each view.

from pathlib import Path
import sys, re, os
import argparse
import numpy as np
import cv2
import torch

# ---------------- config ----------------
# Default thresholds (only used as CLI defaults; not mutated at runtime)
DEFAULT_BOX_THR   = 0.2   # used by local GDINO (and also bbox_threshold for gdino15)
DEFAULT_TEXT_THR  = 0.25
DEFAULT_MIN_SCORE = 0.15

USE_CPU   = False

# Max number of classes per GroundingDINO prompt (only for local gdino1)
MAX_CLASSES_PER_PROMPT = 6

# GDINO 1.5 cloud model + IOU threshold (for NMS in cloud)
GDINO15_MODEL   = os.environ.get("GDINO15_MODEL", "GroundingDino-1.5-Pro")
GDINO15_IOU_THR = 0.8

IMG_GLOB = "view_*.png"

# ---------------- paths & imports ----------------

# Directory where this script lives
ROOT = Path(__file__).resolve().parent

# Grounded-SAM-2 root:
#   - default: <this_script_dir>/Grounded-SAM-2
#   - or override via env GROUNDED_SAM2_ROOT
GSAM2_ROOT = Path(
    os.environ.get("GROUNDED_SAM2_ROOT", ROOT / "Grounded-SAM-2")
).expanduser().resolve()

if not GSAM2_ROOT.is_dir():
    raise FileNotFoundError(
        f"Could not find Grounded-SAM-2 root at {GSAM2_ROOT}.\n"
        "Either create a 'Grounded-SAM-2' directory next to gdino_infer.py,\n"
        "or set GROUNDED_SAM2_ROOT=/path/to/Grounded-SAM-2."
    )

GDINO_REPO = GSAM2_ROOT / "grounding_dino"
if not GDINO_REPO.is_dir():
    raise FileNotFoundError(
        f"Could not find 'grounding_dino' directory under {GSAM2_ROOT}.\n"
        "Your Grounded-SAM-2 checkout should contain a 'grounding_dino' folder."
    )

# Make sure Python can see:
#   - GSAM2_ROOT so that `grounding_dino` is a top-level package
#   - GDINO_REPO so that `groundingdino` (inner package) is also visible
sys.path.insert(0, str(GSAM2_ROOT))
sys.path.insert(0, str(GDINO_REPO))

# Try to import the local GroundingDINO v1 inference helpers from the repo
try:
    # Repo-style import (matches how util/inference.py imports submodules)
    from grounding_dino.groundingdino.util.inference import (
        load_model,
        load_image,
        predict,
    )
except ImportError:
    # Fallback if someone installed it differently
    try:
        from groundingdino.util.inference import load_model, load_image, predict
    except ImportError as e:
        raise RuntimeError(
            "Failed to import GroundingDINO from Grounded-SAM-2.\n"
            "Make sure the repo is laid out like:\n"
            "  Grounded-SAM-2/grounding_dino/groundingdino/...\n"
            "and that GROUNDED_SAM2_ROOT is set correctly if needed."
        ) from e


# ---------------- CLI ----------------

def parse_args():
    p = argparse.ArgumentParser(
        description="GroundingDINO inference over view_*.png"
    )
    p.add_argument(
        "--backend",
        choices=["gdino1", "gdino15"],
        default="gdino1",
        help="Backend to use: 'gdino1' (local checkpoint, batched prompts) "
             "or 'gdino15' (DDS Cloud API, all classes at once).",
    )
    p.add_argument(
        "--renders_dir",
        type=str,
        required=True,
        help="Directory containing input render PNGs (matched by view_*.png).",
    )
    p.add_argument(
        "--dets_dir",
        type=str,
        required=True,
        help="Directory to save detection .npy files.",
    )
    p.add_argument(
        "--ann_dir",
        type=str,
        required=True,
        help="Directory to save annotated debug images.",
    )
    # Thresholds are now pure CLI options with defaults
    p.add_argument(
        "--box_thr",
        type=float,
        default=DEFAULT_BOX_THR,
        help=f"Box confidence threshold for detections (default: {DEFAULT_BOX_THR}). "
             "Used both for local GDINO and as bbox_threshold for gdino15.",
    )
    p.add_argument(
        "--text_thr",
        type=float,
        default=DEFAULT_TEXT_THR,
        help=f"Text score threshold for local GroundingDINO (default: {DEFAULT_TEXT_THR}).",
    )
    p.add_argument(
        "--min_score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"Final minimum score for saving detections (default: {DEFAULT_MIN_SCORE}).",
    )
    return p.parse_args()


# ---------------- helpers ----------------

def load_classes(root: Path):
    """
    Load class list from custom_classes.txt (required).

    File format:
        custom_classes.txt
            building
            car
            tree
            ...

    Each non-empty line = one class name.
    """
    f = root / "custom_classes.txt"
    if not f.is_file():
        raise FileNotFoundError(
            f"custom_classes.txt not found at {f}. "
            "Please create it with one class name per line."
        )

    lines = [ln.strip() for ln in f.read_text().splitlines()]
    # Keep only non-empty lines
    classes = [ln for ln in lines if ln]
    if not classes:
        raise ValueError(
            f"custom_classes.txt at {f} is empty. "
            "Add one class name per line."
        )

    # Normalize to lowercase for prompting + matching, and deduplicate
    seen, out = set(), []
    for c in classes:
        c_low = c.lower()
        if c_low not in seen:
            out.append(c_low)
            seen.add(c_low)
    return out


def make_prompt_local(classes):
    """
    Prompt style for local GDINO v1.

    GroundingDINO v1 expects something like:
        "a . b . c ."
    """
    return " . ".join(classes) + " ."


def make_prompt_cloud(classes):
    """
    Prompt style for GDINO 1.5 Cloud.

    DDS examples use "a. b. c." style. We'll mirror that:
        "a. b. c."
    """
    return ". ".join(classes) + "."


def iter_batches(seq, batch_size):
    """Yield consecutive batches of size <= batch_size from seq."""
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def clamp_sort_box(b, H, W):
    x1, y1, x2, y2 = [float(v) for v in b]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0, min(int(round(x1)), W - 1))
    x2 = max(0, min(int(round(x2)), W - 1))
    y1 = max(0, min(int(round(y1)), H - 1))
    y2 = max(0, min(int(round(y2)), H - 1))
    return [x1, y1, x2, y2]


def to_xyxy_pixels(b, H, W):
    """
    Convert GroundingDINO box -> pixel xyxy.
    Handles XYXY (normalized or pixels) and CXCYWH (normalized or pixels).
    """
    bb = np.array(b, dtype=float).tolist()
    if len(bb) != 4:
        raise ValueError(f"Unexpected bbox length: {len(bb)} for {b}")
    x1, y1, x2, y2 = bb
    vmin, vmax = min(bb), max(bb)

    # Looks like xyxy
    if (x2 >= x1) and (y2 >= y1):
        if 0.0 <= vmin <= 1.0 and 0.0 <= vmax <= 1.0:
            x1 *= W
            y1 *= H
            x2 *= W
            y2 *= H
        return clamp_sort_box([x1, y1, x2, y2], H, W)

    # Otherwise treat as cxcywh
    cx, cy, w, h = bb
    if 0.0 <= vmin <= 1.0 and 0.0 <= vmax <= 1.0:
        cx *= W
        cy *= H
        w *= W
        h *= H
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return clamp_sort_box([x1, y1, x2, y2], H, W)


def _put_label(img, x, y, text, color):
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th_text), _ = cv2.getTextSize(text, font, fs, th)
    cv2.rectangle(img, (x, y - th_text - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), font, fs, (255, 255, 255), th, cv2.LINE_AA)


def draw_boxes(img_bgr, dets, color=(255, 0, 255), show_label=True):
    out = img_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if show_label:
            lbl = d.get("label", "unknown")
            sc = d.get("score", None)
            txt = f"{lbl}" + (f" {sc:.2f}" if isinstance(sc, (float, int)) else "")
            _put_label(out, x1, max(0, y1), txt, color)
    return out


def _resolve_gdino_config() -> str:
    """
    Find the GroundingDINO SwinT OGC config bundled in Grounded-SAM-2.
    We search under the `grounding_dino` directory.
    """
    candidates = list(GDINO_REPO.rglob("GroundingDINO_SwinT_OGC.py"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find GroundingDINO_SwinT_OGC.py under {GDINO_REPO}.\n"
            "Make sure the GroundingDINO code is present in the Grounded-SAM-2 repo."
        )
    return str(candidates[0])


def _ensure_weights() -> str:
    """
    Return a local path to the GroundingDINO weights.

    Priority:
      1) env GROUNDINGDINO_WEIGHTS or GROUNDING_DINO_WEIGHTS
      2) <GSAM2_ROOT>/gdino_checkpoints/groundingdino_swint_ogc.pth
    """
    # Environment overrides
    for env_name in ("GROUNDINGDINO_WEIGHTS", "GROUNDING_DINO_WEIGHTS"):
        env_path = os.environ.get(env_name)
        if env_path:
            p = Path(env_path).expanduser().resolve()
            if not p.is_file():
                raise FileNotFoundError(
                    f"{env_name} is set but file not found: {p}"
                )
            return str(p)

    # Default: checkpoint directory inside Grounded-SAM-2
    ckpt_dir = GSAM2_ROOT / "gdino_checkpoints"
    default_ckpt = ckpt_dir / "groundingdino_swint_ogc.pth"
    if default_ckpt.is_file():
        return str(default_ckpt.resolve())

    raise FileNotFoundError(
        "Could not find GroundingDINO weights.\n"
        f"Tried: {default_ckpt}\n"
        "Run (inside Grounded-SAM-2):\n"
        "  cd gdino_checkpoints && bash download_ckpts.sh\n"
        "or set GROUNDINGDINO_WEIGHTS to an existing .pth file."
    )


# -------- shared canonicalization --------

def make_canonicalizer(classes):
    """Return a canonicalize(phrase) function bound to a given classes list."""
    def _tokenize(s: str):
        return re.findall(r"[a-z0-9]+", s.lower())

    def canonicalize(phrase: str):
        p = phrase.strip().lower()
        if p in classes:
            return p
        ptoks = _tokenize(p)
        if not ptoks:
            return "unknown"
        matches = []
        for cls in classes:
            ct = _tokenize(cls)
            L = len(ct)
            for i in range(0, len(ptoks) - L + 1):
                if ptoks[i:i + L] == ct:
                    matches.append((cls, L))
                    break
        if not matches:
            return "unknown"
        matches.sort(key=lambda t: -t[1])  # longest match first
        return matches[0][0]
    return canonicalize


# ---------------- local GDINO v1 pipeline ----------------

def run_gdino1(classes, images, OUT_DIR, ANN_DIR,
               box_thr: float, text_thr: float, min_score: float):
    print("[INFO] Using backend: gdino1 (local checkpoint, batched prompts)")

    GDINO_CFG = _resolve_gdino_config()
    GDINO_WTS = _ensure_weights()

    device = "cpu" if (USE_CPU or not torch.cuda.is_available()) else "cuda"
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] GroundingDINO cfg: {GDINO_CFG}")
    print(f"[INFO] GroundingDINO weights: {GDINO_WTS}")
    print(f"[INFO] box_thr={box_thr}, text_thr={text_thr}, min_score={min_score}")

    model = load_model(GDINO_CFG, GDINO_WTS, device=device).eval()
    canonicalize = make_canonicalizer(classes)

    for img_path in images:
        print(f"[INFO] Processing {img_path.name}")
        image_source, image_tensor = load_image(str(img_path))
        if isinstance(image_source, np.ndarray):
            H, W = image_source.shape[:2]
            img_bgr = (
                image_source
                if image_source.ndim == 3
                else cv2.cvtColor(image_source, cv2.COLOR_GRAY2BGR)
            )
        else:
            W, H = image_source.size
            img_bgr = np.array(image_source)
            if img_bgr.ndim == 2:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        # Collect detections from all class batches for this image
        dets = []

        batches = list(iter_batches(classes, MAX_CLASSES_PER_PROMPT))
        print(f"[INFO]  - {len(batches)} class batches for this view")

        for batch_idx, class_batch in enumerate(batches):
            prompt = make_prompt_local(class_batch)
            print(
                f"[INFO]    Batch {batch_idx + 1}/{len(batches)} "
                f"with {len(class_batch)} classes: {class_batch}"
            )

            with torch.no_grad():
                boxes_raw, logits_raw, phrases_raw = predict(
                    model=model,
                    image=image_tensor,
                    caption=prompt,
                    box_threshold=box_thr,
                    text_threshold=text_thr,
                    device=device,
                )

            # Convert to pixel xyxy + canonicalize labels
            for b, s, p in zip(boxes_raw, logits_raw, phrases_raw):
                try:
                    score = (
                        float(s)
                        if isinstance(s, (float, int))
                        else float(getattr(s, "item", lambda: s)())
                    )
                except Exception:
                    score = float(s)
                if score < min_score:
                    continue
                bbox = to_xyxy_pixels(b, H, W)
                label = canonicalize(str(p))
                # drop unknowns completely
                if label == "unknown":
                    continue
                dets.append(
                    {
                        "bbox": bbox,
                        "label": label,
                        "score": score,
                        "raw_phrase": str(p).strip().lower(),
                    }
                )

        out_npy = OUT_DIR / f"dets_{img_path.stem}.npy"
        np.save(out_npy, np.array(dets, dtype=object))
        print(f"[OK] boxes -> {out_npy} (N={len(dets)})")

        # Simple debug image from combined dets
        try:
            dets_saved = np.load(out_npy, allow_pickle=True).tolist()
            img_from_saved = draw_boxes(img_bgr, dets_saved, color=(255, 0, 255))
            cv2.imwrite(str(ANN_DIR / f"{img_path.stem}_from_saved.jpg"), img_from_saved)
        except Exception as e:
            print(f"[WARN] failed to draw from_saved for {img_path.name}: {e}")


# ---------------- GDINO 1.5 cloud pipeline ----------------

def run_gdino15(classes, images, OUT_DIR, ANN_DIR,
                box_thr: float, min_score: float):
    print("[INFO] Using backend: gdino15 (DDS Cloud API, all classes at once)")

    # Lazy import so local-only users don't need the SDK
    try:
        from dds_cloudapi_sdk import Config, Client
        from dds_cloudapi_sdk.tasks.v2_task import V2Task
    except ImportError as e:
        raise RuntimeError(
            "dds_cloudapi_sdk is not installed. Install it with:\n"
            "  pip install dds-cloudapi-sdk\n"
        ) from e

    token = (os.environ.get("DDS_API_TOKEN") or "").strip()
    if not token:
        raise SystemExit(
            "[ERR] DDS_API_TOKEN is not set.\n"
            "Export your API token, e.g.:\n"
            "  export DDS_API_TOKEN='your_token_here'\n"
            "and then run:\n"
            "  python gdino_infer.py --backend gdino15 ..."
        )

    client = Client(Config(token))
    text_prompt = make_prompt_cloud(classes)
    print(f"[INFO] GDINO 1.5 model: {GDINO15_MODEL}")
    print(f"[INFO] Text prompt: {text_prompt}")
    print(f"[INFO] box_thr={box_thr}, min_score={min_score}, iou_thr={GDINO15_IOU_THR}")

    canonicalize = make_canonicalizer(classes)

    def run_cloud_on_path(img_fp: str):
        image_url = client.upload_file(img_fp)
        task = V2Task(
            api_path="/v2/task/grounding_dino/detection",
            api_body={
                "model": GDINO15_MODEL,
                "image": image_url,
                "prompt": {"type": "text", "text": text_prompt},
                "targets": ["bbox"],
                "bbox_threshold": float(box_thr),
                "iou_threshold": float(GDINO15_IOU_THR),
            },
        )
        client.run_task(task)
        return task.result.get("objects", [])

    for img_path in images:
        print(f"[INFO] Processing {img_path.name}")
        img_path_str = str(img_path)

        # Load image to get shape & for annotation
        src_bgr = cv2.imread(img_path_str)
        if src_bgr is None:
            print(f"[WARN] Could not read image {img_path_str}, skipping.")
            continue
        H, W = src_bgr.shape[:2]

        # Call DDS cloud once per image (all classes at once, no batching)
        objects = run_cloud_on_path(img_path_str)

        dets = []
        for obj in objects:
            box = obj.get("bbox")
            if box is None:
                continue
            cls_name = str(obj.get("category", "")).lower().strip()
            score = float(obj.get("score", 0.0))
            if score < min_score:
                continue

            bbox = to_xyxy_pixels(box, H, W)
            label = canonicalize(cls_name)
            # drop unknowns completely
            if label == "unknown":
                continue
            dets.append(
                {
                    "bbox": bbox,
                    "label": label,
                    "score": score,
                    "raw_phrase": cls_name,
                }
            )

        out_npy = OUT_DIR / f"dets_{img_path.stem}.npy"
        np.save(out_npy, np.array(dets, dtype=object))
        print(f"[OK] boxes -> {out_npy} (N={len(dets)})")

        # Simple debug image from combined dets
        try:
            img_from_saved = draw_boxes(src_bgr, dets, color=(255, 0, 255))
            cv2.imwrite(str(ANN_DIR / f"{img_path.stem}_from_saved.jpg"), img_from_saved)
        except Exception as e:
            print(f"[WARN] failed to draw annotated image for {img_path.name}: {e}")


# ---------------- main ----------------

def main():
    args = parse_args()

    def resolve_dir(path_str: str) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else (ROOT / p)

    RENDERS = resolve_dir(args.renders_dir)
    OUT_DIR = resolve_dir(args.dets_dir)
    ANN_DIR = resolve_dir(args.ann_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ANN_DIR.mkdir(parents=True, exist_ok=True)

    classes = load_classes(ROOT)
    print(f"[INFO] Loaded {len(classes)} classes from custom_classes.txt")
    print(f"[INFO] Classes: {classes}")
    print(f"[INFO] Renders dir: {RENDERS}")
    print(f"[INFO] Dets dir:    {OUT_DIR}")
    print(f"[INFO] Ann dir:     {ANN_DIR}")
    print(f"[INFO] CLI thresholds: box_thr={args.box_thr}, text_thr={args.text_thr}, min_score={args.min_score}")

    images = sorted(RENTERS.glob(IMG_GLOB)) 
    if not images:
        print(f"[ERR] No images found in {RENTERS}/{IMG_GLOB}")
        return

    if args.backend == "gdino15":
        run_gdino15(classes, images, OUT_DIR, ANN_DIR,
                    box_thr=args.box_thr,
                    min_score=args.min_score)
    else:
        run_gdino1(classes, images, OUT_DIR, ANN_DIR,
                   box_thr=args.box_thr,
                   text_thr=args.text_thr,
                   min_score=args.min_score)


if __name__ == "__main__":
    main()
