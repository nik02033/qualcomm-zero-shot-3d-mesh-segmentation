# Qualcomm: Zero-shot 3D Mesh Segmentation

End-to-end pipeline for **zero-shot semantic segmentation of large 3D meshes** using multi-view rendering, vision-language grounding, and 2D–3D label fusion.

Main ideas:
- Render the mesh from many views (PyTorch3D)
- Detect & segment objects in each view (GroundingDINO + FastSAM or GroundingDINO 1.5 + SAM2)
- Optionally auto-discover class names (RAM++)
- Fuse 2D labels back onto the 3D mesh (per-face semantic labels)

---

## Repo Layout (Key Scripts)

- `run_pipeline.py` – **main entrypoint**, orchestrates the full pipeline
- `render_qualcomm_pytorch3d.py` – multiview PyTorch3D renderer
- `main_sam.py` – FastSAM mask extraction from renders
- `grounded_sam2_gd15_batch.py` – GroundingDINO 1.5 inference over renders
- `gdino_infer.py` – GroundingDINO v1 inference over renders
- `mask_voting.py` – match FastSAM masks to boxes (voting heuristics)
- `main_fuse_classcolor_pytorch3d.py` – fuse 2D labels back to mesh
- `collect_ram_all_classes.py` – RAM++-based class discovery (writes `custom_classes.txt`)
- `conversion.py` – helper to convert PLY + texture → OBJ

External repos expected next to this project:
- `Grounded-SAM-2/` – for local GroundingDINO v1 + GDINO 1.5 cloud batch script
- `recognize-anything/` – RAM++ (batch_inference + pretrained weights)
- `FastSAM/` – for local FastSAM inference
---

## Pipeline Overview

### 1. (Optional) PLY → OBJ

```bash
python conversion.py \
  --ply       mesh.ply \
  --texture    texture.jpg \
  --output_dir output
```

### 2. Main Pipeline

```bash
python run_pipeline.py \
  --input_mesh mesh.obj \
  --backend gdino1 \            # or gdino15
  --config pipeline_config.json \
  [--sumparts] \
  [--ram] \
  [--dry_run]
```

**Backends**:
- `gdino1`  – local GroundingDINO v1 + FastSAM + mask voting
- `gdino15` – GroundingDINO 1.5 cloud (DDS) + SAM2 (no FastSAM/mask_voting)

**Optional `--ram`**:
- After rendering, runs `collect_ram_all_classes.py` on the render directory
- Uses RAM++ to auto-generate `custom_classes.txt` (union of open-vocab tags)
- `neg.txt` (if present) is used to blacklist tokens

---

## Config (`pipeline_config.json`)

You configure rendering + thresholds in a single JSON file. Example:

```json
{
  "image_size": 1024,
  "elev_deg": 35.0,
  "zoom": 1.0,
  "num_coarse_views": 9,
  "num_fine_views": 3,
  "fine_zoom": 0.7,
  "backoff": 1.2,
  "z_down_frac": 0.10,

  "box_thr": 0.2,
  "text_thr": 0.25,
  "min_score": 0.15,

  "coverage_req": 0.85,
  "area_low": 0.85,
  "area_high": 2.0,
  "hug_single_min": 0.30,
  "hug_single_max": 1.00,
  "inside_tol_frac": 0.05,

  "grounding_model": "GroundingDino-1.5-Pro",
  "box_threshold": 0.2,
  "iou_threshold": 0.8
}
```

---

## Outputs

For a given tile `Tile_...` the pipeline writes:

- `Tile_.../renders/view_*.png` and `camera_params.json`
- `Tile_.../dino_dets/dets_view_*.npy` and annotated JPEGs
- `Tile_.../dino_dets/masks_labeled/masks_view_*.npy` (per-view labeled masks)
- `Tile_.../face_class_strings.npy` (per-face class labels)
- `Tile_.../segmented.obj` (segmented mesh with class colors)
