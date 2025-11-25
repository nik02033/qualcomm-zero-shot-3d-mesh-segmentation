#!/usr/bin/env python3
# grounded_sam2_gd15_batch.py
#
# Batch pipeline:
#   - Read a set of rendered views (view_*.png)
#   - Run DDS Cloud GroundingDINO (1.5 / 1.6 etc.) to get text-conditioned boxes
#   - Run local SAM2 on those boxes to get segmentation masks
#   - Save:
#       * Annotated JPGs with boxes + masks + labels
#       * Per-image masks_*.npy files (mask + label + box + score)
#       * Optional COCO-style JSON per image (RLE masks + bbox + score)
#
# This script is designed to be called from the larger Qualcomm 3D pipeline,
# but can also be run standalone.

import os
import cv2
import json
import glob
import torch
import argparse
import tempfile
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from PIL import Image

API_TOKEN = ""  # <-- put your token here (or export DDS_API_TOKEN)

# TEXT_PROMPT is filled at runtime from either SUMPARTS_PROMPT or custom_classes.txt
TEXT_PROMPT = ""

# Hardcoded SumParts prompt (matches SUM Parts classes / colors used downstream)
SUMPARTS_PROMPT = (
    "trees. water. car. boat. roof. building. chimney. window. door. grass. road. "
    "road marking. bridge. dormer. balcony. roof_installations"
)

# Input image naming convention (must match renderer output)
IMG_GLOB = "view_*.png"

# SAM2 checkpoint and config (local model)
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Future hook: sliced / tiled inference for huge images (currently unused)
WITH_SLICE_INFERENCE = False

# ---------------------------------------------------------------------------
# DDS Cloud API (Grounding DINO)
# ---------------------------------------------------------------------------
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task

# ---------------------------------------------------------------------------
# SAM2 (local segmentation model)
# ---------------------------------------------------------------------------
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def parse_args():
    """
    Parse command-line arguments for batch Grounded-SAM2 processing.

    Core inputs:
      --images_dir : directory containing view_*.png
      --annot_dir  : directory for annotated visualization JPGs
      --npy_dir    : directory for masks_*.npy outputs
      --json_dir   : optional directory for per-image JSON (COCO-style RLEs)

    Semantic controls:
      --sumparts        : use hardcoded SUMPARTS_PROMPT instead of custom_classes.txt
      --grounding_model : DDS GroundingDINO model identifier
      --box_threshold   : bounding box confidence threshold
      --iou_threshold   : IoU threshold used for box NMS in DDS
    """
    p = argparse.ArgumentParser(
        description="Batch Grounded-SAM (DDS GroundingDINO + SAM2) over images named view_*.png"
    )
    p.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing view_*.png",
    )
    p.add_argument(
        "--annot_dir",
        type=str,
        required=True,
        help="Directory to save annotated JPGs",
    )
    p.add_argument(
        "--npy_dir",
        type=str,
        required=True,
        help="Directory to save per-image masks_*.npy",
    )
    p.add_argument(
        "--json_dir",
        type=str,
        default=None,
        help="(Optional) Directory to save per-image JSON files",
    )
    p.add_argument(
        "--sumparts",
        dest="sum_parts",
        action="store_true",
        help="Use hardcoded SumParts TEXT_PROMPT instead of ../custom_classes.txt",
    )
    p.add_argument(
        "--grounding_model",
        type=str,
        default="GroundingDino-1.5-Pro",
        help="DDS Grounding DINO model name "
             "(e.g. 'GroundingDino-1.5-Pro', 'GroundingDino-1.6-Pro').",
    )
    p.add_argument(
        "--box_threshold",
        type=float,
        default=0.2,
        help="Box confidence threshold for Grounding DINO (bbox_threshold).",
    )
    p.add_argument(
        "--iou_threshold",
        type=float,
        default=0.8,
        help="IoU threshold for NMS in Grounding DINO (iou_threshold).",
    )

    return p.parse_args()


def single_mask_to_rle(mask: np.ndarray):
    """
    Convert a 2D binary mask (H, W) to a COCO RLE dict with UTF-8 counts.

    The RLE is Fortran-ordered as required by pycocotools.
    """
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _flatten_scores(scores, n_limit=None):
    """
    Make SAM2 scores JSON-safe python floats.

    Handles:
      - torch.Tensor on CPU/GPU
      - numpy arrays of any shape (including scalar, (N,1), etc.)
      - plain python scalars / lists

    Returns:
      list[float] with length <= n_limit (if n_limit is provided).
    """
    # If it's a tensor (SAM2 / PyTorch), move to CPU numpy
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    elif hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()

    # Normalize to a numpy array
    arr = np.array(scores)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.squeeze()
    # Ensure 1D list of floats
    arr = np.ravel(arr).astype(np.float32).tolist()

    # Optionally truncate
    if n_limit is not None:
        arr = arr[:n_limit]
    return arr


def _save_annotated(img_bgr, boxes_xyxy, masks_bool, labels, det_confs, out_path: Path):
    """
    Draw bounding boxes, class labels, and masks on top of the input BGR image,
    then write a JPG visualization to 'out_path'.

    If there are no detections, the original image is written unchanged.
    """
    if len(boxes_xyxy) == 0:
        cv2.imwrite(str(out_path), img_bgr)
        return

    # Detections object packs boxes + masks + class ids for supervision annotators
    class_ids = np.arange(len(labels))
    dets = sv.Detections(xyxy=boxes_xyxy, mask=masks_bool, class_id=class_ids)
    label_texts = [f"{c} {float(s):.2f}" for c, s in zip(labels, det_confs)]

    img = img_bgr.copy()
    img = sv.BoxAnnotator().annotate(scene=img, detections=dets)
    img = sv.LabelAnnotator().annotate(scene=img, detections=dets, labels=label_texts)
    img = sv.MaskAnnotator().annotate(scene=img, detections=dets)

    cv2.imwrite(str(out_path), img)


def _save_npy(stem, boxes_xyxy, masks_bool, labels, det_confs, out_dir: Path):
    """
    Save per-image mask information to masks_<stem>.npy in out_dir.

    Stored format: numpy array (dtype=object), each element is a dict:
      {
        "mask":        HxW uint8 (0/1),
        "label":       str,
        "matched_box": [x1, y1, x2, y2] (int),
        "score":       float (GroundingDINO detection confidence)
      }
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_list = []
    n = len(boxes_xyxy)

    for i in range(n):
        x1, y1, x2, y2 = [int(round(float(v))) for v in boxes_xyxy[i]]
        out_list.append(
            {
                "mask": masks_bool[i].astype(np.uint8),
                "label": str(labels[i]),
                "matched_box": [x1, y1, x2, y2],
                "score": float(det_confs[i]),
            }
        )

    np.save(out_dir / f"masks_{stem}.npy", np.array(out_list, dtype=object))


def _save_json(stem, img_path, boxes_xyxy, masks_bool, labels, sam2_scores, img_wh, out_dir: Path):
    """
    Optionally save a per-image JSON file with COCO-style RLE masks.

    JSON structure:
      {
        "image_path": str,
        "annotations": [
          {
            "class_name": str,
            "bbox": [x1, y1, x2, y2],
            "segmentation": RLE dict,
            "score": float(SAM2 score)
          }, ...
        ],
        "box_format": "xyxy",
        "img_width": W,
        "img_height": H
      }
    """
    if out_dir is None:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    W, H = img_wh
    # Convert boolean masks to RLE
    mask_rles = [single_mask_to_rle(mask.astype(np.uint8)) for mask in masks_bool]
    ann = []
    for cls, box, rle, s in zip(labels, boxes_xyxy.tolist(), mask_rles, sam2_scores):
        ann.append(
            {
                "class_name": cls.strip(),
                "bbox": [int(round(float(x))) for x in box],
                "segmentation": rle,
                "score": float(s),
            }
        )

    results = {
        "image_path": str(img_path),
        "annotations": ann,
        "box_format": "xyxy",
        "img_width": W,
        "img_height": H,
    }
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump(results, f, indent=4)


def main():
    """
    Entry point:
      - Resolve TEXT_PROMPT
      - Set up DDS Cloud client + SAM2 predictor
      - Loop over all view_*.png, run detection + segmentation
      - Save annotated images, masks_*.npy, optional JSON per view
    """
    global TEXT_PROMPT

    args = parse_args()

    images_dir = Path(args.images_dir)
    annot_dir = Path(args.annot_dir)
    npy_dir = Path(args.npy_dir)
    json_dir = Path(args.json_dir) if args.json_dir else None

    # ----------------------------------------------------------------------
    # 1) Build the text prompt (class vocabulary)
    # ----------------------------------------------------------------------
    #   - If --sumparts is enabled, use the fixed SUMPARTS_PROMPT string.
    #   - Otherwise, read class names line-by-line from ../custom_classes.txt
    #     and join them into "class1. class2. class3." style text.
    if args.sum_parts:
        TEXT_PROMPT = SUMPARTS_PROMPT
    else:
        custom_path = Path(__file__).resolve().parent.parent / "custom_classes.txt"
        if not custom_path.is_file():
            raise SystemExit(f"[ERR] custom_classes.txt not found at {custom_path}")
        with open(custom_path, "r") as f:
            # Each non-empty line is one class
            custom_classes = [line.strip() for line in f.readlines() if line.strip()]
        if not custom_classes:
            raise SystemExit(f"[ERR] No classes found in {custom_path}")
        # Build TEXT_PROMPT in "class1. class2. class3." format
        TEXT_PROMPT = ". ".join(custom_classes) + "."

    # ----------------------------------------------------------------------
    # 2) Initialize DDS Cloud client (GroundingDINO) and SAM2
    # ----------------------------------------------------------------------
    # Token resolution:
    #   - Prefer API_TOKEN constant if set
    #   - Fallback to DDS_API_TOKEN environment variable
    token = (API_TOKEN or os.environ.get("DDS_API_TOKEN") or "").strip()
    if not token:
        raise SystemExit("[ERR] API token missing. Set API_TOKEN in this script or export DDS_API_TOKEN.")

    # DDS client handles GroundingDINO detection requests
    client = Client(Config(token))

    # Build SAM2 once (same style as single-image version)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # ----------------------------------------------------------------------
    # 3) Class name <-> ID mappings for text prompt classes
    # ----------------------------------------------------------------------
    # Text prompt is dot-separated; each segment is a semantic class
    classes = [x.strip().lower() for x in TEXT_PROMPT.split('.') if x]
    class_name_to_id = {name: i for i, name in enumerate(classes)}
    class_id_to_name = {i: name for name, i in class_name_to_id.items()}

    # ----------------------------------------------------------------------
    # 4) Collect all input image paths matching view_*.png
    # ----------------------------------------------------------------------
    img_paths = sorted(glob.glob(str(images_dir / IMG_GLOB)))
    if not img_paths:
        raise SystemExit(f"[ERR] No images matching {images_dir}/{IMG_GLOB}")

    print(f"[INFO] Found {len(img_paths)} images in {images_dir}")
    print(f"[INFO] Using TEXT_PROMPT: {TEXT_PROMPT}")
    print(f"[INFO] Grounding model: {args.grounding_model}")
    print(f"[INFO] BOX threshold:  {args.box_threshold}")
    print(f"[INFO] IOU threshold:  {args.iou_threshold}")

    # ----------------------------------------------------------------------
    # Helper: one DDS GroundingDINO call per image path
    # ----------------------------------------------------------------------
    def run_cloud_on_path(img_fp: str):
        """
        Upload a local image file to DDS Cloud, then run a single
        GroundingDINO detection task using the current TEXT_PROMPT and
        configured thresholds.

        Returns:
            task.result["objects"] (list of detection dicts), or [] if none.
        """
        # 1) Upload file and get a DDS image URL
        image_url = client.upload_file(img_fp)

        # 2) Prepare detection task
        task = V2Task(
            api_path="/v2/task/grounding_dino/detection",
            api_body={
                "model": args.grounding_model,  # configurable from CLI
                "image": image_url,
                "prompt": {"type": "text", "text": TEXT_PROMPT},
                "targets": ["bbox"],
                "bbox_threshold": float(args.box_threshold),
                "iou_threshold": float(args.iou_threshold),
            },
        )

        # 3) Execute and wait for completion
        client.run_task(task)
        return task.result.get("objects", [])

    # ----------------------------------------------------------------------
    # 5) Main loop over all views
    # ----------------------------------------------------------------------
    for img_path_str in img_paths:
        img_path = Path(img_path_str)
        stem = img_path.stem
        print(f"[RUN] {stem}")

        # --------------------------------------------------------------
        # 5a) DDS detection step (via temp JPEG)
        # --------------------------------------------------------------
        # Some APIs prefer JPEG; we re-encode the PNG to a temp .jpg file.
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
            tmpname = tmpfile.name

        cv2.imwrite(tmpname, cv2.imread(str(img_path)))
        try:
            objects = run_cloud_on_path(tmpname)
        finally:
            # Best-effort cleanup of temp file
            try:
                os.remove(tmpname)
            except Exception:
                pass

        # Collect boxes + scores + class names, filtering out unknown classes
        input_boxes = []
        det_confs = []
        class_names = []
        class_ids = []
        for obj in objects:
            box = obj.get("bbox")
            if box is None:
                # Skip degenerate objects without bbox
                continue

            cls_name = str(obj.get("category", "")).lower().strip()
            if cls_name not in class_name_to_id:
                # Ignore detections for classes not in our TEXT_PROMPT
                continue

            input_boxes.append(box)
            det_confs.append(float(obj.get("score", 0.0)))
            class_names.append(cls_name)
            class_ids.append(class_name_to_id[cls_name])

        input_boxes = np.array(input_boxes, dtype=np.float32).reshape(-1, 4)
        det_confs = np.array(det_confs, dtype=np.float32)
        class_ids = np.array(class_ids, dtype=np.int32)

        # Load original BGR image for visualization + size info
        src_bgr = cv2.imread(str(img_path))
        H, W = src_bgr.shape[:2]

        # --------------------------------------------------------------
        # 5b) Handle case: no detections for this view
        # --------------------------------------------------------------
        if input_boxes.size == 0:
            # Still produce "empty" artifacts so downstream code has a file
            annot_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(annot_dir / f"{stem}_annotated.jpg"), src_bgr)

            npy_dir.mkdir(parents=True, exist_ok=True)
            np.save(npy_dir / f"masks_{stem}.npy", np.array([], dtype=object))

            if json_dir is not None:
                json_dir.mkdir(parents=True, exist_ok=True)
                with open(json_dir / f"{stem}.json", "w") as f:
                    json.dump(
                        {
                            "image_path": str(img_path),
                            "annotations": [],
                            "box_format": "xyxy",
                            "img_width": W,
                            "img_height": H,
                        },
                        f,
                        indent=4,
                    )
            print(f"[OK] {stem}: no detections")
            continue

        # --------------------------------------------------------------
        # 5c) SAM2 segmentation step
        # --------------------------------------------------------------
        # Enable autocast for speed and memory savings; on modern GPUs also
        # enable TF32 for additional acceleration.
        torch.autocast(
            device_type=("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=torch.bfloat16,
        ).__enter__()
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # SAM2 expects RGB numpy array; load with PIL for consistency
        pil_img = Image.open(str(img_path))
        image_rgb = np.array(pil_img.convert("RGB"))
        sam2_predictor.set_image(image_rgb)

        # One mask per box (multimask_output=False)
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # SAM2 may return (N,1,H,W); squeeze channel dimension if present
        if masks.ndim == 4:
            masks = masks.squeeze(1)  # (N,1,H,W) -> (N,H,W)

        # Convert to boolean mask array for downstream logic
        masks_bool = (masks > 0)

        # Flatten SAM2 scores to a simple list of python floats for JSON safety
        sam2_scores = _flatten_scores(scores, n_limit=len(input_boxes))

        # --------------------------------------------------------------
        # 5d) Save artifacts for this view
        # --------------------------------------------------------------
        # 1) Annotated JPG with boxes + labels + masks
        _save_annotated(
            img_bgr=src_bgr,
            boxes_xyxy=input_boxes,
            masks_bool=masks_bool,
            labels=class_names,
            det_confs=det_confs,
            out_path=Path(args.annot_dir) / f"{stem}_annotated.jpg",
        )

        # 2) masks_<stem>.npy (mask + class + box + score)
        _save_npy(
            stem=stem,
            boxes_xyxy=input_boxes,
            masks_bool=masks_bool,
            labels=class_names,
            det_confs=det_confs,
            out_dir=Path(args.npy_dir),
        )

        # 3) Optional JSON with RLE masks
        _save_json(
            stem=stem,
            img_path=img_path,
            boxes_xyxy=input_boxes,
            masks_bool=masks_bool,
            labels=[c.strip() for c in class_names],
            sam2_scores=sam2_scores,
            img_wh=(W, H),
            out_dir=Path(args.json_dir) if args.json_dir else None,
        )

        print(
            f"[OK] {stem}: wrote "
            f"{Path(args.annot_dir) / f'{stem}_annotated.jpg'}  and  "
            f"{Path(args.npy_dir) / f'masks_{stem}.npy'}"
        )

    print("[DONE] Batch complete.")


if __name__ == "__main__":
    main()
