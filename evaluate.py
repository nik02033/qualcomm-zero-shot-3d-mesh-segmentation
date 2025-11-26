#!/usr/bin/env python3
"""
evaluate.py
Evaluation with automatic label remapping from RAM++ to SUM-Parts.

Workflow:
1. Load ground truth PLY (SUM-Parts IDs 0-12)
2. Load predictions NPY (arbitrary IDs from main_fuse.py: 0,1,2,3...)
3. Read masks/view_XX/label.json to see which pred ID = which RAM++ label
4. Apply label_mapping.json to convert RAM++ → SUM-Parts IDs
5. Compare remapped predictions vs ground truth

Usage:
    python evaluate.py \
      --gt_ply ground_truth.ply \
      --pred_npy face_labels.npy \
      --label_map label_mapping.json \
      --masks_dir masks \
      --output_json results.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict
import numpy as np

# Import fixed schema
from sumparts_config import (
    SUMPARTS_CLASSES,
    SUMPARTS_NAME_TO_ID,
    NUM_FACE_CLASSES,
    DEFAULT_IGNORE_CLASS
)


def load_face_labels(ply_path: str) -> np.ndarray:
    """Load face labels from PLY file."""
    labels = []
    in_faces = False
    vertex_count = 0
    
    with open(ply_path, 'r') as f:
        for line in f:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[2])
            elif line.strip() == "end_header":
                in_faces = True
                continue
            
            if not in_faces:
                continue
            
            if vertex_count > 0:
                vertex_count -= 1
                continue
            
            parts = line.strip().split()
            if not parts:
                continue
            
            try:
                label = int(parts[-2])
                labels.append(label)
            except (ValueError, IndexError):
                continue
    
    return np.array(labels, dtype=np.int32)


def build_pred_to_ram_mapping(masks_dir: str) -> Dict[int, str]:
    """
    Build mapping from prediction IDs to RAM++ labels.
    Reads all masks/view_XX/label.json files to reconstruct the global label map.
    
    Returns:
        {pred_id: ram_label} - e.g., {0: 'background', 1: 'water', 2: 'bridge', ...}
    """
    masks_path = Path(masks_dir)
    view_dirs = sorted(masks_path.glob("view_*"))
    
    pred_to_ram = {}
    
    # Collect all RAM labels and their values from all views
    for view_dir in view_dirs:
        label_json = view_dir / "label.json"
        if not label_json.exists():
            continue
        
        with open(label_json, 'r') as f:
            data = json.load(f)
            mask_meta = data.get("mask", [])
            
            for entry in mask_meta:
                value = entry.get("value", -1)
                ram_label = entry.get("label", "")
                
                if value >= 0 and ram_label:
                    # Store lowercase for consistent matching
                    pred_to_ram[value] = ram_label.lower()
    
    return pred_to_ram


def build_ram_to_sumparts_mapping(label_map_path: str) -> Dict[str, int]:
    """
    Load RAM++ → SUM-Parts mapping from label_mapping.json.
    
    Returns:
        {ram_label: sumparts_id} - e.g., {'building': 3, 'water': 4, ...}
    """
    if not Path(label_map_path).exists():
        raise FileNotFoundError(f"Label mapping not found: {label_map_path}")
    
    with open(label_map_path, 'r') as f:
        data = json.load(f)
    
    mappings = data.get("mappings", {})
    
    ram_to_sumparts = {}
    for ram_label, sumparts_name in mappings.items():
        sumparts_id = SUMPARTS_NAME_TO_ID.get(sumparts_name.lower(), -1)
        if sumparts_id >= 0:
            ram_to_sumparts[ram_label.lower()] = sumparts_id
        else:
            print(f"[WARN] Unknown SUM-Parts class '{sumparts_name}' in mapping for '{ram_label}'")
    
    return ram_to_sumparts


def remap_predictions(pred_labels: np.ndarray,
                     pred_to_ram: Dict[int, str],
                     ram_to_sumparts: Dict[str, int],
                     verbose: bool = True) -> np.ndarray:
    """
    Remap prediction IDs (or RAM++ label strings) to SUM-Parts IDs.

    - If pred_labels is integer-typed: use pred_to_ram (built from masks_dir).
    - If pred_labels is strings/object: assume they are RAM++ labels directly.
    """

    # ------------------------------------------------------------------
    # Case A: predictions are strings / RAM++ labels directly
    # ------------------------------------------------------------------
    if pred_labels.dtype.kind in {"U", "S", "O"}:
        if verbose:
            print("\n[REMAPPING]")
            print("[INFO] Detected string predictions; treating them as RAM++ labels")
        
        # Normalize predictions to lowercase + strip
        norm_pred = np.array(
            [str(x).lower().strip() for x in pred_labels],
            dtype=object
        )

        unique_labels = sorted(set(norm_pred.tolist()))
        if verbose:
            print("\n[INFO] Unique RAM++ labels in predictions:")
            for lab in unique_labels:
                print(f"  '{lab}'")

            print("\n[INFO] RAM++ Label → SUM-Parts ID (from label_mapping.json):")
            for ram_label in sorted(ram_to_sumparts.keys()):
                sp_id = ram_to_sumparts[ram_label]
                sp_name = SUMPARTS_CLASSES[sp_id]
                print(f"  '{ram_label}' → {sp_name} (ID {sp_id})")

        # Initialize remapped array with ignore class
        remapped = np.full(norm_pred.shape, DEFAULT_IGNORE_CLASS, dtype=np.int32)

        # Map known labels
        for ram_label, sumparts_id in ram_to_sumparts.items():
            mask = (norm_pred == ram_label)
            remapped[mask] = sumparts_id

        # Collect & warn about unmapped labels
        unmapped = [lab for lab in unique_labels if lab not in ram_to_sumparts]
        if unmapped and verbose:
            print(f"\n[WARN] {len(unmapped)} RAM++ labels not in label_mapping.json "
                  f"(using 'unclassified'):")
            for lab in unmapped:
                print(f"  '{lab}'")

        if verbose:
            unique_after = np.unique(remapped)
            print("\n[INFO] Remapping Summary:")
            print(f"  Before: {len(unique_labels)} unique RAM++ labels")
            print(f"  After:  {len(unique_after)} unique SUM-Parts classes")
            print(f"  SUM-Parts classes after remapping: {unique_after.tolist()}")

            print("\n[INFO] Prediction distribution AFTER remapping:")
            total = len(remapped)
            for sp_id in sorted(unique_after):
                count = (remapped == sp_id).sum()
                pct = 100.0 * count / total
                sp_name = SUMPARTS_CLASSES.get(sp_id, f"class_{sp_id}")
                print(f"    {sp_name:20s} (ID {sp_id:2d}): {count:6d} faces ({pct:5.2f}%)")

        return remapped

    # ------------------------------------------------------------------
    # Case B: predictions are integer IDs, use pred_to_ram as before
    # ------------------------------------------------------------------
    if verbose:
        print("\n[REMAPPING]")
        print("[INFO] Step 1: Prediction ID → RAM++ Label")
        for pid in sorted(pred_to_ram.keys())[:20]:  # Show first 20
            print(f"  {pid:3d} → '{pred_to_ram[pid]}'")
        if len(pred_to_ram) > 20:
            print(f"  ... ({len(pred_to_ram)} total)")

    if verbose:
        print("\n[INFO] Step 2: RAM++ Label → SUM-Parts ID")
        for ram_label in sorted(ram_to_sumparts.keys()):
            sp_id = ram_to_sumparts[ram_label]
            sp_name = SUMPARTS_CLASSES[sp_id]
            print(f"  '{ram_label}' → {sp_name} (ID {sp_id})")

    # Build complete mapping: pred_id → sumparts_id
    pred_to_sumparts = {}
    unmapped = []

    for pred_id, ram_label in pred_to_ram.items():
        if ram_label in ram_to_sumparts:
            pred_to_sumparts[pred_id] = ram_to_sumparts[ram_label]
        else:
            pred_to_sumparts[pred_id] = DEFAULT_IGNORE_CLASS
            unmapped.append(f"{pred_id}:'{ram_label}'")

    if verbose:
        print("\n[INFO] Step 3: Final Combined Mapping (Pred ID → SUM-Parts ID)")
        for pid in sorted(pred_to_sumparts.keys())[:20]:
            sp_id = pred_to_sumparts[pid]
            sp_name = SUMPARTS_CLASSES.get(sp_id, "unknown")
            ram_label = pred_to_ram.get(pid, "unknown")
            print(f"  {pid:3d} ('{ram_label}') → {sp_name} (ID {sp_id})")
        if len(pred_to_sumparts) > 20:
            print(f"  ... ({len(pred_to_sumparts)} total)")

    if unmapped and verbose:
        print(f"\n[WARN] {len(unmapped)} prediction IDs not mapped (using 'unclassified'):")
        for item in unmapped[:10]:
            print(f"  {item}")
        if len(unmapped) > 10:
            print(f"  ... ({len(unmapped) - 10} more)")

    # Remap the array (integer predictions)
    remapped = np.full_like(pred_labels, DEFAULT_IGNORE_CLASS)

    for pred_id, sumparts_id in pred_to_sumparts.items():
        mask = (pred_labels == pred_id)
        remapped[mask] = sumparts_id

    if verbose:
        unique_before = np.unique(pred_labels)
        unique_after = np.unique(remapped)
        print("\n[INFO] Remapping Summary:")
        print(f"  Before: {len(unique_before)} unique prediction IDs")
        print(f"  After:  {len(unique_after)} unique SUM-Parts classes")
        print(f"  SUM-Parts classes after remapping: {unique_after.tolist()}")

        # Show distribution BEFORE remapping
        print("\n[INFO] Prediction distribution BEFORE remapping:")
        for pred_id in sorted(unique_before)[:20]:
            count = (pred_labels == pred_id).sum()
            pct = 100.0 * count / len(pred_labels)
            ram_label = pred_to_ram.get(int(pred_id), "unknown")
            sumparts_id = pred_to_sumparts.get(int(pred_id), -1)
            sumparts_name = (
                SUMPARTS_CLASSES.get(sumparts_id, "unmapped")
                if sumparts_id >= 0 else "unmapped"
            )
            print(f"    ID {int(pred_id):3d} ('{ram_label:15s}') → {sumparts_name:20s}: "
                  f"{count:6d} faces ({pct:5.2f}%)")

        # Show distribution AFTER remapping
        print("\n[INFO] Prediction distribution AFTER remapping:")
        for sumparts_id in sorted(unique_after):
            count = (remapped == sumparts_id).sum()
            pct = 100.0 * count / len(remapped)
            sumparts_name = SUMPARTS_CLASSES.get(sumparts_id, f"class_{sumparts_id}")
            print(f"    {sumparts_name:20s} (ID {sumparts_id:2d}): {count:6d} faces ({pct:5.2f}%)")

    return remapped


def compute_metrics(gt_labels: np.ndarray,
                   pred_labels: np.ndarray,
                   ignore_class: int = DEFAULT_IGNORE_CLASS,
                   verbose: bool = True) -> Dict:
    """Compute evaluation metrics."""
    
    if verbose:
        print(f"\n[METRICS COMPUTATION]")
        print(f"[INFO] Filtering ignore class: {ignore_class} ({SUMPARTS_CLASSES[ignore_class]})")
    
    # Filter ignore class
    total_faces = len(gt_labels)                        # NEW: total faces (including ignored)
    if ignore_class >= 0:
        valid_mask = (gt_labels != ignore_class)
        gt = gt_labels[valid_mask]
        pred = pred_labels[valid_mask]
        if verbose:
            print(f"[INFO] Evaluating on {len(gt)}/{len(gt_labels)} faces "
                  f"({100*len(gt)/len(gt_labels):.1f}%)")
    else:
        gt = gt_labels
        pred = pred_labels
    
    num_faces_evaluated = len(gt)                       # NEW
    
    # Confusion matrix
    confusion = np.zeros((NUM_FACE_CLASSES, NUM_FACE_CLASSES), dtype=np.int64)
    
    for i in range(NUM_FACE_CLASSES):
        for j in range(NUM_FACE_CLASSES):
            confusion[i, j] = np.sum((gt == i) & (pred == j))
    
    # Print confusion matrix
    if verbose:
        print(f"\n[INFO] Confusion Matrix:")
        print(f"  (Showing non-empty GT classes)")
        total_correct = 0
        total_faces_considered = 0
        
        # Check for predictions of classes not in GT
        gt_classes_present = set(np.unique(gt))
        pred_classes_present = set(np.unique(pred))
        pred_only_classes = pred_classes_present - gt_classes_present
        
        if pred_only_classes:
            print(f"\n[INFO] Predicted classes NOT in ground truth:")
            for cls_id in sorted(pred_only_classes):
                if cls_id < NUM_FACE_CLASSES:
                    cls_name = SUMPARTS_CLASSES[cls_id]
                    count = np.sum(pred == cls_id)
                    pct = 100 * count / len(pred)
                    print(f"  Class {cls_id:2d} ({cls_name:20s}): {count:6d} faces ({pct:5.1f}%) - ALL WRONG")
        
        print(f"\n[INFO] Ground truth class breakdown:")
        for i in range(NUM_FACE_CLASSES):
            row_sum = confusion[i, :].sum()
            if row_sum == 0:
                continue
            
            gt_name = SUMPARTS_CLASSES[i]
            print(f"\n  GT: {gt_name:20s} (ID {i:2d}) - {row_sum:6d} faces")
            
            for j in range(NUM_FACE_CLASSES):
                count = confusion[i, j]
                if count > 0:
                    pred_name = SUMPARTS_CLASSES[j]
                    status = "✓ CORRECT" if i == j else "✗ WRONG"
                    pct = 100 * count / row_sum
                    print(f"    → Predicted as {pred_name:20s} (ID {j:2d}): {count:6d} ({pct:5.1f}%) {status}")
                    total_faces_considered += count
                    if i == j:
                        total_correct += count
        
        if total_faces_considered > 0:
            acc = 100 * total_correct / total_faces_considered
            print(f"\n  Summary: {total_correct:6d}/{total_faces_considered:6d} faces correct ({acc:.2f}%)")
    
    # Per-class IoU
    per_class_iou = {}
    ious = []
    
    # NEW: per-class face counts (for classes actually evaluated)
    per_class_faces = {}
    weighted_iou_sum = 0.0
    total_evaluated_faces_for_weight = 0
    
    for i in range(NUM_FACE_CLASSES):
        if i == ignore_class:
            continue
        
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        union = tp + fp + fn
        total_for_class = confusion[i, :].sum()
        
        if union > 0:
            iou = tp / union
            per_class_iou[SUMPARTS_CLASSES[i]] = float(iou)
            ious.append(iou)
            
            # Only weight by classes that actually appear in GT
            if total_for_class > 0:
                per_class_faces[SUMPARTS_CLASSES[i]] = int(total_for_class)
                weighted_iou_sum += iou * total_for_class
                total_evaluated_faces_for_weight += total_for_class
    
    mean_iou = np.mean(ious) if ious else 0.0
    
    # NEW: weighted mIoU (true mIoU, weighted by #GT faces per class)
    if total_evaluated_faces_for_weight > 0:
        weighted_mean_iou = weighted_iou_sum / total_evaluated_faces_for_weight
    else:
        weighted_mean_iou = 0.0
    
    # Per-class accuracy
    per_class_acc = {}
    accs = []
    
    for i in range(NUM_FACE_CLASSES):
        if i == ignore_class:
            continue
        
        total = confusion[i, :].sum()
        if total > 0:
            acc = confusion[i, i] / total
            per_class_acc[SUMPARTS_CLASSES[i]] = float(acc)
            accs.append(acc)
    
    mean_class_acc = np.mean(accs) if accs else 0.0
    overall_acc = float(np.mean(gt == pred))
    
    # NEW: fraction of faces evaluated
    fraction_faces_evaluated = num_faces_evaluated / total_faces if total_faces > 0 else 0.0
    
    return {
        "mean_iou": float(mean_iou),
        "weighted_mean_iou": float(weighted_mean_iou),          # NEW
        "mean_class_accuracy": float(mean_class_acc),
        "overall_accuracy": float(overall_acc),
        "per_class_iou": per_class_iou,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion.tolist(),
        "num_faces_evaluated": int(num_faces_evaluated),        # UPDATED meaning
        "num_total_faces": int(total_faces),                    # NEW
        "fraction_faces_evaluated": float(fraction_faces_evaluated),  # NEW
        "faces_per_class": per_class_faces                      # NEW
    }


def evaluate(gt_ply: str,
            pred_npy: str,
            label_map: str,
            masks_dir: str,
            verbose: bool = True) -> Dict:
    """Run complete evaluation pipeline."""
    
    print(f"{'='*70}")
    print(f"SUM-PARTS EVALUATION WITH LABEL REMAPPING")
    print(f"{'='*70}")
    
    # Load ground truth
    print(f"\n[1/5] Loading ground truth")
    print(f"  File: {gt_ply}")
    gt_labels = load_face_labels(gt_ply)
    print(f"  Loaded: {len(gt_labels)} faces")
    print(f"  GT classes: {sorted(np.unique(gt_labels).tolist())}")
    
    # Load predictions
    print(f"\n[2/5] Loading predictions")
    print(f"  File: {pred_npy}")
    pred_labels = np.load(pred_npy, allow_pickle=True)
    print(f"  Loaded: {len(pred_labels)} predictions")
    print(f"  Pred IDs: {sorted(np.unique(pred_labels).tolist())}")
    
    # Validate shapes
    if len(gt_labels) != len(pred_labels):
        raise ValueError(
            f"Shape mismatch: GT has {len(gt_labels)} faces, "
            f"predictions have {len(pred_labels)}"
        )
    
    # Build mappings
    print(f"\n[3/5] Building label mappings")
    print(f"  Masks directory: {masks_dir}")
    print(f"  Label mapping file: {label_map}")
    
    pred_to_ram = build_pred_to_ram_mapping(masks_dir)
    ram_to_sumparts = build_ram_to_sumparts_mapping(label_map)
    
    # Remap predictions
    print(f"\n[4/5] Remapping predictions")
    pred_labels_remapped = remap_predictions(
        pred_labels,
        pred_to_ram,
        ram_to_sumparts,
        verbose=verbose
    )
    
    # Compute metrics
    print(f"\n[5/5] Computing metrics")
    results = compute_metrics(gt_labels, pred_labels_remapped, verbose=verbose)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"\nOverall Metrics:")
    print(f"  Mean IoU (unweighted):    {results['mean_iou']:.4f}")
    print(f"  Mean IoU (weighted):      {results['weighted_mean_iou']:.4f}")  # NEW
    print(f"  Mean Class Accuracy:      {results['mean_class_accuracy']:.4f}")
    print(f"  Overall Accuracy:         {results['overall_accuracy']:.4f}")
    
    # NEW: faces evaluated vs total
    num_eval = results["num_faces_evaluated"]
    num_total = results["num_total_faces"]
    frac_eval = results["fraction_faces_evaluated"]
    print(f"  Faces Evaluated:          {num_eval} / {num_total} "
          f"({frac_eval*100:.2f}%)")
    
    print(f"\n{'='*70}")
    print(f"Per-Class Results:")
    print(f"{'='*70}")
    print(f"{'Class':<25} {'IoU':>10} {'Accuracy':>10} {'#Faces':>10}")  # NEW: add #Faces column
    print(f"{'-'*70}")
    
    sorted_classes = sorted(
        results["per_class_iou"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    faces_per_class = results.get("faces_per_class", {})
    
    for class_name, iou in sorted_classes:
        acc = results["per_class_accuracy"].get(class_name, 0.0)
        n_faces = faces_per_class.get(class_name, 0)
        print(f"{class_name:<25} {iou:>10.4f} {acc:>10.4f} {n_faces:>10d}")
    
    print(f"{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions with automatic RAM++ → SUM-Parts remapping"
    )
    parser.add_argument("--gt_ply", required=True,
                       help="Ground truth PLY file")
    parser.add_argument("--pred_npy", required=True,
                       help="Predicted face labels (.npy)")
    parser.add_argument("--label_map", required=True,
                       help="Label mapping JSON (RAM++ → SUM-Parts)")
    parser.add_argument("--masks_dir", required=True,
                       help="Masks directory with view_XX/label.json files")
    parser.add_argument("--output_json", default=None,
                       help="Save results to JSON")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate(
        gt_ply=args.gt_ply,
        pred_npy=args.pred_npy,
        label_map=args.label_map,
        masks_dir=args.masks_dir,
        verbose=not args.quiet
    )
    
    # Save results
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
