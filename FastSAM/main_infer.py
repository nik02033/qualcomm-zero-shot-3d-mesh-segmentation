# infer.py (place in project root alongside the FastSAM/ folder)
import os
import sys
import argparse
from pathlib import Path
import clip
import numpy as np
import torch


def run_fastsam_on_directory(
    images_dir,
    model_path,
    output_dir,
    device=None,
    retina_masks=True,
    imgsz=1024,
    conf=0.3,
    iou=0.9,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)
    # Import AFTER path + purge so FastSAM's own deps resolve correctly
    from fastsam import FastSAM, FastSAMPrompt  # noqa: E402

    model = FastSAM(model_path)

    results = []
    image_files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    for img_file in image_files:
        if "depth" in img_file.lower():
            print(f"[INFO] Skipping depth image {img_file}")
            continue

        img_path = os.path.join(images_dir, img_file)
        print(f"[INFO] Running FastSAM (text prompt mode) on {img_file}...")

        everything_results = model(
            img_path,
            device=device,
            retina_masks=retina_masks,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
        )
        print(f"[DEBUG] Type of everything_results: {type(everything_results)}")

        prompt_process = FastSAMPrompt(img_path, everything_results, device=device)

        city_classes = [
            "building", "tree", "water", "streetlamp",
            "chimney", "car", "road", "bridge", "park",
        ]
        ann = prompt_process.text_prompt(text=", ".join(city_classes))

        output_path = os.path.join(output_dir, img_file)
        prompt_process.plot(annotations=ann, output_path=output_path)

        # collect raw masks if present
        all_masks = []

        def _extract_masks(res_obj):
            masks_obj = getattr(res_obj, "masks", None)
            if masks_obj is None:
                return
            masks_tensor = masks_obj.masks  # torch.Tensor [N,H,W] or [N,1,H,W]
            masks_np_local = masks_tensor.cpu().numpy()
            for i in range(masks_np_local.shape[0]):
                mask_i = masks_np_local[i]
                if mask_i.ndim == 3 and mask_i.shape[0] == 1:
                    mask_i = np.squeeze(mask_i, axis=0)
                if mask_i.ndim == 2:
                    all_masks.append(mask_i.astype(np.uint8))
                else:
                    print(f"[WARNING] Skipping invalid mask shape: {mask_i.shape}")

        if isinstance(everything_results, list):
            for res in everything_results:
                _extract_masks(res)
        else:
            _extract_masks(everything_results)

        if all_masks:
            masks_np = np.stack(all_masks, axis=0)
            mask_output_path = os.path.join(output_dir, img_file + ".npy")
            np.save(mask_output_path, masks_np)
            print(f"[INFO] Saved masks numpy array to {mask_output_path}")
        else:
            print(f"[WARNING] No valid masks found for {img_file}, skipping saving masks.")

        results.append({"image": img_path, "masks": ann})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Directory of rendered images")
    parser.add_argument("--model_path", default="./weights/FastSAM.pt", help="Path to FastSAM weights")
    parser.add_argument("--output_dir", default="outputs", help="Directory to save masks")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.9)
    parser.add_argument("--no-retina-masks", action="store_true", help="Disable retina masks")
    args = parser.parse_args()

    # Resolve absolute paths BEFORE we chdir
    project_root = Path(__file__).resolve().parent
    images_dir_abs = str((project_root / args.images_dir).resolve())
    output_dir_abs = str((project_root / args.output_dir).resolve())
    model_path_abs = str((project_root / args.model_path).resolve())

    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Loading FastSAM model from {model_path_abs}")
    print(f"[INFO] Running inference on: {images_dir_abs}")
    
    # run
    run_fastsam_on_directory(
        images_dir=images_dir_abs,
        model_path=model_path_abs,
        output_dir=output_dir_abs,
        device=args.device,
        retina_masks=not args.no_retina_masks,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )


if __name__ == "__main__":
    main()
