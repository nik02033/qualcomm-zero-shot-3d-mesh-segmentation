#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def run_cmd(cmd, cwd=None):
    """Pretty-print and execute a shell command."""
    print("\n" + "=" * 80)
    print("Running command:")
    if cwd:
        print(f"  (cwd: {cwd})")
    print("  ", " ".join(map(str, cmd)))
    print("=" * 80)
    subprocess.run(cmd, check=True, cwd=cwd)


def main():
    parser = argparse.ArgumentParser(
        description="Run full Qualcomm 3D mesh pipeline sequentially."
    )
    parser.add_argument(
        "--input_mesh",
        required=True,
        help="Path to input mesh, e.g. converted/Tile_+1990_+2691_L2.obj",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["gdino1", "gdino15"],
        help="Detection backend to use (gdino1 or gdino15).",
    )
    parser.add_argument(
        "--config",
        default="pipeline_config.json",
        help="JSON config file with rendering and detection parameters.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands but do not execute them.",
    )
    parser.add_argument(
        "--sumparts",
        action="store_true",
        help="Use SUM-Parts-specific settings; propagates to fusion/GSAM2 scripts.",
    )
    parser.add_argument(
        "--ram",
        action="store_true",
        help=(
            "After rendering, run collect_ram_all_classes.py to generate "
            "custom_classes.txt via RAM++ over the rendered views."
        ),
    )

    args = parser.parse_args()

    config = load_config(Path(args.config))

    script_root = Path(__file__).resolve().parent

    input_mesh = Path(args.input_mesh)
    if not input_mesh.exists():
        raise FileNotFoundError(f"Input mesh not found: {input_mesh}")

    # Example: converted/Tile_+1990_+2691_L2.obj → Tile_+1990_+2691_L2
    tile_name = input_mesh.stem
    base_dir = Path(tile_name)

    # All pipeline outputs live under this tile folder:
    renders_dir     = base_dir / "renders"
    masks_dir       = base_dir / "masks"
    mask_debug_dir  = base_dir / "mask_debug"
    dino_dets_dir   = base_dir / "dino_dets"
    ann_dir         = dino_dets_dir / "annotated"
    labeled_dir     = dino_dets_dir / "masks_labeled"

    # Make sure directories exist where needed
    for p in [renders_dir, masks_dir, mask_debug_dir, dino_dets_dir, ann_dir, labeled_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # Convenience wrapper to respect --dry_run
    def maybe_run(cmd, cwd=None):
        if args.dry_run:
            print("\n[DRY RUN] Would run:")
            if cwd:
                print(f"  (cwd: {cwd})")
            print("  ", " ".join(map(str, cmd)))
        else:
            run_cmd(cmd, cwd=cwd)

    # ------------------------------------------------------------------
    # 1) Render mesh → images
    # ------------------------------------------------------------------
    render_cmd = [
        "python",
        "render_qualcomm_pytorch3d.py",
        "--input", str(input_mesh),
        "--image_size", str(config["image_size"]),
        "--elev_deg", str(config["elev_deg"]),
        "--zoom", str(config["zoom"]),
        "--num_coarse_views", str(config["num_coarse_views"]),
        "--num_fine_views", str(config["num_fine_views"]),
        "--fine_zoom", str(config["fine_zoom"]),
        "--backoff", str(config["backoff"]),
        "--z_down_frac", str(config["z_down_frac"]),
        "--output_dir", str(renders_dir),
    ]
    maybe_run(render_cmd)

    # ------------------------------------------------------------------
    # 1.5) Optional RAM++ tag collection → custom_classes.txt
    # ------------------------------------------------------------------
    if args.ram:
        # Paths relative to this script’s directory:
        ra_dir       = script_root / "recognize-anything"
        checkpoint   = ra_dir / "pretrained" / "ram_plus_swin_large_14m.pth"
        neg_file     = script_root / "neg.txt"
        out_classes  = script_root / "custom_classes.txt"

        ram_cmd = [
            "python",
            "collect_ram_all_classes.py",
            "--ra-dir", str(ra_dir),
            "--image-dir", str(renders_dir),
            "--checkpoint", str(checkpoint),
            "--out", str(out_classes),
        ]

        # Only pass --neg if the file exists; collect script already handles missing, but this is explicit.
        if neg_file.exists():
            ram_cmd += ["--neg", str(neg_file)]

        maybe_run(ram_cmd)

    # ------------------------------------------------------------------
    # 2) Backend-specific steps
    # ------------------------------------------------------------------

    if args.backend == "gdino1":
        # --------------------------------------------------------------
        # gdino1 path:
        #   1) FastSAM masks
        #   2) GroundingDINO v1 (local)
        #   3) Mask voting
        # --------------------------------------------------------------

        sam_cmd = [
            "python",
            "main_sam.py",
            "--renders_dir", str(renders_dir),
            "--masks_dir", str(masks_dir),
            "--debug_dir", str(mask_debug_dir),
        ]
        maybe_run(sam_cmd)

        # GroundingDINO thresholds from config (with sane defaults)
        box_thr   = config.get("box_thr", 0.2)
        text_thr  = config.get("text_thr", 0.25)
        min_score = config.get("min_score", 0.15)

        gdino_cmd = [
            "python",
            "gdino_infer.py",
            "--backend", "gdino1",
            "--renders_dir", str(renders_dir),
            "--dets_dir", str(dino_dets_dir),
            "--ann_dir", str(ann_dir),
            "--box_thr", str(box_thr),
            "--text_thr", str(text_thr),
            "--min_score", str(min_score),
        ]
        maybe_run(gdino_cmd)

        # Mask voting thresholds from config (all optional; defaults are in mask_voting.py)
        coverage_req    = config.get("coverage_req", None)
        area_low        = config.get("area_low", None)
        area_high       = config.get("area_high", None)
        hug_single_min  = config.get("hug_single_min", None)
        hug_single_max  = config.get("hug_single_max", None)
        inside_tol_frac = config.get("inside_tol_frac", None)

        mask_voting_cmd = [
            "python",
            "mask_voting.py",
            "--renders_dir", str(renders_dir),
            "--dets_dir", str(dino_dets_dir),
            "--ann_dir", str(ann_dir),
            "--labeled_dir", str(labeled_dir),
        ]

        # Only append args that are explicitly set in the config
        if coverage_req is not None:
            mask_voting_cmd += ["--coverage_req", str(coverage_req)]
        if area_low is not None:
            mask_voting_cmd += ["--area_low", str(area_low)]
        if area_high is not None:
            mask_voting_cmd += ["--area_high", str(area_high)]
        if hug_single_min is not None:
            mask_voting_cmd += ["--hug_single_min", str(hug_single_min)]
        if hug_single_max is not None:
            mask_voting_cmd += ["--hug_single_max", str(hug_single_max)]
        if inside_tol_frac is not None:
            mask_voting_cmd += ["--inside_tol_frac", str(inside_tol_frac)]

        maybe_run(mask_voting_cmd)

    elif args.backend == "gdino15":
        # --------------------------------------------------------------
        # gdino15 path:
        #   Run Grounded-SAM-2 batch script with GDINO 1.5 cloud
        # --------------------------------------------------------------
        gsam2_root = Path("Grounded-SAM-2")

        # GDINO 1.5 parameters from config
        grounding_model = config.get("grounding_model", "GroundingDino-1.5-Pro")
        box_threshold   = config.get("box_threshold", 0.2)
        iou_threshold   = config.get("iou_threshold", 0.8)

        gsam2_cmd = [
            "python",
            "grounded_sam2_gd15_batch.py",
            "--images_dir", str(Path("..") / base_dir / "renders"),
            "--annot_dir",  str(Path("..") / base_dir / "dino_dets" / "annotated"),
            "--npy_dir",    str(Path("..") / base_dir / "dino_dets" / "masks_labeled"),
            "--grounding_model", str(grounding_model),
            "--box_threshold",   str(box_threshold),
            "--iou_threshold",   str(iou_threshold),
        ]
        if args.sumparts:
            gsam2_cmd.append("--sumparts")

        maybe_run(gsam2_cmd, cwd=str(gsam2_root))

    # ------------------------------------------------------------------
    # 3) Fusion (runs for BOTH backends)
    # ------------------------------------------------------------------
    out_labels = base_dir / "face_class_strings.npy"
    out_colored_obj = base_dir / f"segmented_{tile_name}.obj"

    fuse_cmd = [
        "python",
        "main_fuse_classcolor_pytorch3d.py",
        "--obj", str(input_mesh),
        "--renders", str(renders_dir),
        "--masks_dir", str(labeled_dir),
        "--out_labels", str(out_labels),
        "--out_colored_obj", str(out_colored_obj),
    ]
    if args.sumparts:
        fuse_cmd.append("--sumparts")

    maybe_run(fuse_cmd)

    print("\nPipeline completed successfully.")
    print(f"Base folder: {base_dir.resolve()}")
    print(f"Labels:      {out_labels.resolve()}")
    print(f"Segmented:   {out_colored_obj.resolve()}")


if __name__ == "__main__":
    main()
