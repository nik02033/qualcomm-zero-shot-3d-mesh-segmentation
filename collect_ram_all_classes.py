#!/usr/bin/env python3
"""
Fast, batched RAM++ tag collection using batch_inference.py utilities (absolute paths).

- Loads model ONCE (CUDA)
- Batches images from --image-dir (view_*.png|jpg|jpeg)
- Applies official RAM thresholds (closed-set)
- Writes UNION of tags to --out, ONE PER LINE (no prefix)
- Excludes any tags present in a newline-separated negatives file via --neg

Usage:
python collect_ram_all_classes.py \
  --ra-dir /path/to/recognize-anything \
  --image-dir /path/to/renders \
  --checkpoint /path/to/pretrained/ram_plus_swin_large_14m.pth \
  --out /path/to/custom_classes.txt \
  --neg /path/to/negatives.txt \
  --batch-size 16 --num-workers 4 --gpu-id 0 --device cuda
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Iterable

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torch import Tensor

# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra-dir", type=Path, required=True,
                    help="Path to recognize-anything repo (where batch_inference.py lives)")
    ap.add_argument("--image-dir", type=Path, required=True,
                    help="Directory containing view_*.png|jpg|jpeg")
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="Path to ram_plus_swin_large_14m.pth")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output file for UNION of tags (one per line)")
    ap.add_argument("--neg", type=Path, default=None,
                    help="Path to newline-separated negatives file; tags here are excluded from --out")
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--gpu-id", default="0")
    ap.add_argument("--device", choices=["cuda"], default="cuda",
                    help="RAM++ forward uses CUDA paths; CPU disabled here.")
    ap.add_argument("--keep-blacklisted", action="store_true",
                    help="Keep tokens that are normally dropped by the internal blacklist.")
    return ap.parse_args()

# -------------------- Small helpers ---------------------
def _natural_key(p: Path):
    m = re.search(r"view_(\d+)", p.stem)
    return int(m.group(1)) if m else p.stem

def find_images(img_dir: Path) -> List[Path]:
    imgs: List[Path] = []
    for ext in (".png", ".jpg", ".jpeg"):
        imgs += list(img_dir.glob(f"view_*{ext}"))
    return sorted(set(imgs), key=_natural_key)

class RenderDataset(Dataset):
    def __init__(self, files: List[Path], transform_fn):
        self.files = files
        self.transform = transform_fn
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx: int) -> Tensor:
        p = self.files[idx]
        try:
            img = Image.open(p).convert("RGB")
        except (OSError, FileNotFoundError, UnidentifiedImageError):
            img = Image.new("RGB", (10, 10), 0)
            print(f"[WARN] Error loading image: {p}")
        return self.transform(img)

# --- token cleaning ---
TOKEN_BLACKLIST = {
    "screenshot", "photo", "image", "images", "processing", "demo",
    "cpu", "cuda", "model", "pretrained", "deprecated", "please", "import",
    "warning", "info", "success", "futurewarning)"
}
VALID_TOKEN_RE = re.compile(r"^[a-z][a-z0-9\-\s/]+[a-z0-9]$")

def _normalize_token(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip().lower()
    t = t.strip("[]()\"'")
    return t

def clean_tokens(tags: Iterable[str], keep_blacklisted: bool) -> List[str]:
    out = []
    for raw in tags:
        t = _normalize_token(raw)
        if not t or len(t) > 60:
            continue
        if not keep_blacklisted and t in TOKEN_BLACKLIST:
            continue
        if VALID_TOKEN_RE.match(t):
            out.append(t)
    return sorted(set(out))

def read_negatives_file(path: Path) -> List[str]:
    if not path:
        return []
    if not path.is_file():
        print(f"[WARN] Negatives file missing: {path} (skipping)")
        return []
    negs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = _normalize_token(line)
            if t:
                negs.append(t)
    return sorted(set(negs))

# -------------------- Main --------------------
def main():
    args = parse_args()

    # Validate paths
    if not args.ra_dir.is_dir():
        sys.stderr.write(f"[ERR] RAM dir missing: {args.ra_dir}\n")
        sys.exit(1)
    if not args.image_dir.is_dir():
        sys.stderr.write(f"[ERR] images dir missing: {args.image_dir}\n")
        sys.exit(1)
    if not args.checkpoint.is_file():
        sys.stderr.write(f"[ERR] checkpoint not found: {args.checkpoint}\n")
        sys.exit(1)
    if not torch.cuda.is_available():
        sys.stderr.write("[ERR] CUDA not available but --device cuda was requested.\n")
        sys.exit(1)

    # Make RAM repo importable
    sys.path.insert(0, str(args.ra_dir))

    # Imports from RAM repo AFTER sys.path insert
    from batch_inference import (  # type: ignore
        load_ram_plus, load_thresholds, forward_ram_plus
    )
    from ram import get_transform  # type: ignore
    import batch_inference as BI    # to set its module-level 'device'

    # Device / env
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    BI.device = "cuda"  # ensure the imported module uses CUDA

    print(f"[INFO] Using device: CUDA")

    files = find_images(args.image_dir)
    if not files:
        sys.stderr.write(f"[ERR] No images named view_*.png|jpg|jpeg in {args.image_dir}\n")
        sys.exit(1)
    print(f"[INFO] Found {len(files)} images. Loading model once …")

    t0_all = time.perf_counter()

    # Build transform
    transform_fn = get_transform(args.input_size)

    # Tag list + thresholds (absolute paths)
    taglist_file = args.ra_dir / "ram" / "data" / "ram_tag_list.txt"
    thre_file    = args.ra_dir / "ram" / "data" / "ram_tag_list_threshold.txt"
    if not taglist_file.is_file():
        sys.stderr.write(f"[ERR] Missing tag list: {taglist_file}\n")
        sys.exit(1)
    if not thre_file.is_file():
        sys.stderr.write(f"[ERR] Missing threshold file: {thre_file}\n")
        sys.exit(1)

    with taglist_file.open("r", encoding="utf-8") as f:
        taglist = [line.strip() for line in f]
    class_idxs = list(range(len(taglist)))

    thresholds = load_thresholds(
        threshold=None,
        threshold_file=str(thre_file),
        model_type="ram_plus",
        open_set=False,
        class_idxs=class_idxs,
        num_classes=len(taglist),
    )

    # Load model ONCE
    t0 = time.perf_counter()
    model = load_ram_plus(
        backbone="swin_l",
        checkpoint=str(args.checkpoint),
        input_size=args.input_size,
        taglist=taglist,
        tag_des=None,
        open_set=False,
        class_idxs=class_idxs
    )
    t1 = time.perf_counter()
    print(f"[TIMING] Model load + to(cuda): {t1 - t0:.2f}s")

    # DataLoader
    ds = RenderDataset(files, transform_fn)
    dl = DataLoader(
        ds, shuffle=False, drop_last=False, pin_memory=True,
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Batched inference -> logits
    logits = torch.empty(len(files), len(taglist))
    pos = 0
    with torch.inference_mode():
        for batch in dl:
            s0 = time.perf_counter()
            out = forward_ram_plus(model, batch)  # [B, num_classes] on CUDA
            bs = batch.shape[0]
            logits[pos:pos+bs, :] = out.cpu()
            pos += bs
            s1 = time.perf_counter()
            print(f"[OK] batch of {bs}: {(s1 - s0):.2f}s")

    # Union with thresholds
    raw_union = set()
    for scores in logits.tolist():
        for i, s in enumerate(scores):
            if s >= thresholds[i]:
                raw_union.add(taglist[i])

    # Clean tokens
    cleaned_union = clean_tokens(raw_union, keep_blacklisted=args.keep_blacklisted)

    # Load negatives (newline-separated file) and exclude
    negatives = set(read_negatives_file(args.neg)) if args.neg else set()
    final_tags = sorted([t for t in cleaned_union if t not in negatives])

    # Write output ONE PER LINE
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write("\n".join(final_tags) + "\n")
    print(f"[DONE] Wrote {len(final_tags)} tags (one per line) → {args.out}")

    if negatives:
        print(f"[INFO] Negatives excluded ({len(negatives)}): {', '.join(sorted(negatives))}")

    t_all = time.perf_counter() - t0_all
    print(f"[TIMING] Total elapsed: {t_all:.2f}s")

if __name__ == "__main__":
    main()
