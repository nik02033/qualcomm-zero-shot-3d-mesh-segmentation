#!/usr/bin/env python3
# render_qualcomm_pytorch3d.py
#
# PyTorch3D multiview renderer (coarse ring + adaptive fine grid)
# - Robust camera placement via look_at_view_transform
# - Per-view znear / zfar based on camera–object distance
# - "Headlight" lighting (light near camera) + strong ambient
# - Double-sided rendering (no back-face culling)
# - Naïve rasterizer (bin_size=0) to avoid bin overflows on huge meshes
# - Fallback white vertex texture if OBJ textures aren’t available
# - Outputs:
#       <output_dir>/view_XX.png
#       <output_dir>/camera_params.json
#   camera_params.json is compatible with the rest of the pipeline

import os, json, math, argparse
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader,
    MeshRenderer,
    PointLights,
    Materials,
    TexturesVertex,
    blending,
    look_at_view_transform,
)

# Flags for global render behavior
ALWAYS_MESHLAB_LIGHTING = True     # If True, use a simple "Meshlab-style" headlight
ALWAYS_DOUBLE_SIDED     = True     # If True, disable back-face culling

# ---------- Camera / render constants ----------

CLEARANCE_FRAC = 0.10      # Keep camera at least 10% of bbox height above "ground" along up-axis
VFOV_DEG = 60.0            # Vertical field-of-view in degrees (used to derive intrinsics in JSON)


def ensure_dir(p: str):
    """
    Create directory 'p' (and parents) if it doesn't exist.
    No-op if it already exists.
    """
    os.makedirs(p, exist_ok=True)


def fov_to_intrinsics(w: int, h: int, fov_deg: float):
    """
    Convert a vertical field-of-view to pinhole intrinsics (fx, fy, cx, cy).

    We assume a standard pinhole model:
        fx = fy = (H / 2) / tan(FOV_v / 2)
        cx = W / 2, cy = H / 2

    Args:
        w: image width in pixels.
        h: image height in pixels.
        fov_deg: vertical FOV in degrees.

    Returns:
        (fx, fy, cx, cy)
    """
    f = (h / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return f, f, w / 2.0, h / 2.0


def compute_bounds_ground_up(mesh_verts: np.ndarray):
    """
    Compute axis-aligned bounds, "up" axis, and a ground value.

    Heuristic:
      - Up-axis is chosen as the axis with the smallest extent
        (often Z for building-like scenes).
      - Ground is approximated by the 5th percentile along the up-axis.

    Args:
        mesh_verts: (V, 3) vertex array.

    Returns:
        center: (3,) bbox center
        extent: (3,) bbox size along each axis
        up_idx: index of up-axis (0:X, 1:Y, 2:Z)
        ground_value: approximate ground height along up-axis
    """
    if mesh_verts.size == 0:
        # Fallback for degenerate inputs
        return np.array([0., 0., 0.]), np.array([1., 1., 1.]), 1, -0.5

    mins = mesh_verts.min(axis=0)
    maxs = mesh_verts.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = (maxs - mins)

    # Up-axis = smallest extent dimension
    up_idx = int(np.argmin(extent))
    # "Ground" = lower-bound percentile along up-axis
    ground_value = float(np.percentile(mesh_verts[:, up_idx], 5.0))

    return center, extent, up_idx, ground_value


def fit_distance_axis(extent: np.ndarray, up_idx: int, vfov_deg: float, aspect: float = 1.0) -> float:
    """
    Compute a camera distance that "fits" the object into the view.

    We consider:
      - vertical fit (along up-axis)
      - horizontal fit (diagonal of the 2D projection in horizontal plane)
    and take the maximum of the two distances, then pad by 2%.

    Args:
        extent: bbox extents (3,)
        up_idx: index of up-axis
        vfov_deg: vertical FOV in degrees
        aspect: image aspect ratio (W/H)

    Returns:
        float distance from object center.
    """
    half_h = float(extent[up_idx]) * 0.5
    other = [i for i in range(3) if i != up_idx]
    half_diag_horiz = float(np.linalg.norm(extent[other]) * 0.5)

    vfov = math.radians(vfov_deg)
    hfov = 2.0 * math.atan(math.tan(vfov / 2.0) * aspect)

    d_v = half_h / max(math.tan(vfov / 2.0), 1e-6)
    d_h = half_diag_horiz / max(math.tan(hfov / 2.0), 1e-6)

    return max(d_v, d_h) * 1.02  # small padding to avoid clipping at edges


def unit_vec_for_axis(idx: int):
    """
    Return a unit vector along the given axis index (0:X, 1:Y, 2:Z).
    """
    v = np.zeros(3, float)
    v[idx] = 1.0
    return v


def grid_block_centers(center: np.ndarray, extent: np.ndarray, up_idx: int):
    """
    Produce the 3x3 grid of block centers in the horizontal plane.

    The horizontal plane is orthogonal to 'up_idx'. For each horizontal axis,
    we split the bbox into 3 segments and place centers in the middle
    of each segment.

    Returns:
        List of 9 centers (np.ndarray shape (3,)).
    """
    horiz = [i for i in range(3) if i != up_idx]
    mins = center - 0.5 * extent
    steps = extent / 3.0

    # Centers along each horizontal dimension
    offsets_0 = [mins[horiz[0]] + (k + 0.5) * steps[horiz[0]] for k in range(3)]
    offsets_1 = [mins[horiz[1]] + (k + 0.5) * steps[horiz[1]] for k in range(3)]

    centers = []
    for i in range(3):
        for j in range(3):
            c = center.copy()
            c[horiz[0]] = offsets_0[i]
            c[horiz[1]] = offsets_1[j]
            centers.append(c)
    return centers


def np_img_to_pil(img_np: np.ndarray) -> Image.Image:
    """
    Convert float image in [0,1] to 8-bit PIL image.

    Args:
        img_np: numpy array (H, W, 3 or 4) with values in [0,1].

    Returns:
        PIL.Image (RGB).
    """
    img_np = np.clip(img_np, 0.0, 1.0)
    img_np = (img_np * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(img_np)


def main():
    """
    Entry point for PyTorch3D renderer.

    Pipeline:
      1. Parse CLI arguments and create output directory.
      2. Load mesh (with textures or fallback white vertex colors).
      3. Compute bbox, up-axis, ground height, and camera distances.
      4. Render:
         - Coarse ring: N orbit views around full scene.
         - Fine grid: 3x3 block-based "close-up" views with adaptive
                      number of azimuths per block.
      5. Save camera_params.json with intrinsics + extrinsics per view.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input OBJ mesh path.")
    ap.add_argument("--image_size", type=int, default=1024, help="Square output resolution (H=W).")
    ap.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Scale the auto-fit camera distance (1.0=fit, <1 closer, >1 farther).",
    )
    ap.add_argument(
        "--elev_deg",
        type=float,
        default=35.0,
        help="Ring elevation in degrees (slightly top-down).",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="renders",
        help="Directory to save rendered images and camera_params.json (default: renders).",
    )

    # Coarse/fine controls
    ap.add_argument(
        "--num_coarse_views",
        type=int,
        default=None,
        help="Number of coarse orbit views (defaults to --num_views or 9).",
    )
    ap.add_argument(
        "--num_views",
        type=int,
        default=None,
        help="(Deprecated) alias for coarse views.",
    )
    ap.add_argument(
        "--num_fine_views",
        type=int,
        default=3,
        help="Nominal number of fine views per block (can be overridden by adaptive logic).",
    )
    ap.add_argument(
        "--fine_zoom",
        type=float,
        default=0.7,
        help="Extra zoom for fine views (<1 closer). "
             "Final fine dist = fit(block) * zoom * fine_zoom.",
    )

    # Extra knobs
    ap.add_argument(
        "--backoff",
        type=float,
        default=1.2,
        help="Retreat multiplier for camera distance (coarse & fine).",
    )
    ap.add_argument(
        "--z_down_frac",
        type=float,
        default=0.10,
        help="Shift cameras downward along global Z by this fraction of bbox Z extent.",
    )

    args = ap.parse_args()

    # Use CLI output directory
    output_dir = args.output_dir
    ensure_dir(output_dir)
    print(f"[INFO] Loading model (with textures): {args.input}")
    print(f"[INFO] Output directory: {output_dir}")

    # ------------------------------------------------------------------
    # 1) Choose device (CUDA if available, else CPU).
    # ------------------------------------------------------------------
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ------------------------------------------------------------------
    # 2) Load mesh and ensure it has textures.
    #    If no textures present, use a white vertex texture so it won't render black.
    # ------------------------------------------------------------------
    meshes = load_objs_as_meshes([args.input], device=device)
    if getattr(meshes, "textures", None) is None or meshes.textures is None:
        V = meshes.verts_packed().shape[0]
        white = torch.ones((1, V, 3), device=device)
        meshes.textures = TexturesVertex(verts_features=white)

    # ------------------------------------------------------------------
    # 3) Compute bounds, "up" axis, and ground height.
    # ------------------------------------------------------------------
    verts_cpu = meshes.verts_packed().detach().cpu().numpy()
    center, extent, up_idx, ground_val = compute_bounds_ground_up(verts_cpu)
    up_vec = unit_vec_for_axis(up_idx)
    horiz_axes = [i for i in range(3) if i != up_idx]
    bbox_h = float(extent[up_idx])

    # Minimum allowed camera height along up-axis (to avoid going underground).
    min_eye_up = ground_val + CLEARANCE_FRAC * max(bbox_h, 1e-6)

    # ------------------------------------------------------------------
    # 4) Renderer configuration (rasterizer + materials).
    # ------------------------------------------------------------------
    W = H = int(args.image_size)
    cull = False if ALWAYS_DOUBLE_SIDED else True

    # Naïve rasterizer (bin_size=0) to avoid bin overflows on giant meshes.
    try:
        rast_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=cull,
            bin_size=0,
            max_faces_per_bin=0,
        )
    except TypeError:
        # Fallback for older PyTorch3D versions without max_faces_per_bin.
        rast_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=cull,
            bin_size=0,
        )

    # Simple materials (Phong shading).
    materials = Materials(
        device=device,
        specular_color=((0.15, 0.15, 0.15),),
        shininess=(32.0,),
    )

    def make_camera(R_t: torch.Tensor, T_t: torch.Tensor, znear: float, zfar: float):
        """
        Construct a FoVPerspectiveCameras instance with fixed vertical FOV
        and supplied near/far clipping distances.
        """
        return FoVPerspectiveCameras(
            device=device,
            R=R_t,
            T=T_t,
            fov=VFOV_DEG,  # vertical FoV
            znear=znear,
            zfar=zfar,
        )

    def make_renderer(cameras, lights):
        """
        Construct a MeshRenderer with SoftPhong shader and a given camera + lights.
        """
        shader = SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials,
            blend_params=blending.BlendParams(background_color=(0.0, 0.0, 0.0)),
        )
        return MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=rast_settings),
            shader=shader,
        )

    # ------------------------------------------------------------------
    # 5) Global camera distance setup for coarse ring.
    # ------------------------------------------------------------------
    num_coarse = (
        args.num_coarse_views
        if args.num_coarse_views is not None
        else (args.num_views if args.num_views is not None else 9)
    )
    num_coarse = max(int(num_coarse), 0)

    base_dist = fit_distance_axis(extent, up_idx=up_idx, vfov_deg=VFOV_DEG, aspect=1.0)
    coarse_dist = max(base_dist * float(args.zoom), 1e-6)
    elev_deg = float(args.elev_deg)

    # Storage for extrinsics / metadata to dump into camera_params.json
    Rs: List[np.ndarray] = []
    Ts: List[np.ndarray] = []
    eyes: List[Tuple[float, float, float]] = []
    used_azims: List[float] = []
    used_elevs: List[float] = []

    def make_eye_for(target: np.ndarray, az_deg: float, dist: float, elev_deg_local: float):
        """
        Build a camera position on a ring around 'target' with specified azimuth
        and elevation (in degrees), then apply global zoom/backoff/downshift.

        Camera is placed in "world" coordinates, respecting the chosen up-axis.
        """
        el = math.radians(elev_deg_local)
        az = math.radians(az_deg)

        # Apply global backoff multiplier to distance
        dist_eff = dist * max(args.backoff, 1e-6)

        # Decompose into horizontal + vertical components
        r_h = dist_eff * math.cos(el)
        up_off = dist_eff * math.sin(el)

        eye = target.copy()
        eye[horiz_axes[0]] += r_h * math.cos(az)
        eye[horiz_axes[1]] += r_h * math.sin(az)
        eye[up_idx] += up_off

        # Additional downward shift along global Z (for a "slightly lower camera" effect)
        eye[2] -= float(args.z_down_frac) * float(extent[2])

        return eye

    def place_render(i_idx: int, eye: np.ndarray, center_tgt: np.ndarray, is_fine: bool):
        """
        Place a camera at 'eye' looking at 'center_tgt', fix clearance if needed,
        set up lights, render an image, save PNG and log extrinsics.

        Args:
            i_idx: sequential view index (for view_XX naming).
            eye: camera position in world coordinates.
            center_tgt: look-at target (typically global bbox center or block center).
            is_fine: whether this is a "fine" view (slightly different lighting).
        """
        # Enforce minimum height above ground by adjusting elevation if needed.
        if eye[up_idx] < min_eye_up:
            v = eye - center_tgt
            dist = float(np.linalg.norm(v))
            if dist < 1e-9:
                # Degenerate case: if dist ~ 0, choose arbitrary direction
                v = unit_vec_for_axis((up_idx + 1) % 3)
                dist = 1.0

            # Constrain vertical ratio to avoid invalid asin
            needed = (min_eye_up - center_tgt[up_idx]) / max(dist, 1e-6)
            needed = max(min(needed, 0.99), -0.99)

            # Derive elevation from required "up" fraction and keep radius
            el = math.asin(needed)
            r_h = dist * math.cos(el)

            horiz_axes_local = [a for a in range(3) if a != up_idx]
            vh = v.copy()
            vh[up_idx] = 0.0
            az = math.atan2(vh[horiz_axes_local[1]], vh[horiz_axes_local[0]])

            eye = center_tgt.copy()
            eye[horiz_axes_local[0]] += r_h * math.cos(az)
            eye[horiz_axes_local[1]] += r_h * math.sin(az)
            eye[up_idx] += dist * math.sin(el)

            elev_deg_actual = math.degrees(el)
            az_deg_actual = math.degrees(az) % 360.0
        else:
            # No clearance fix needed; infer (az, elev) from the vector
            v = eye - center_tgt
            dist = float(np.linalg.norm(v))
            horiz_axes_local = [a for a in range(3) if a != up_idx]
            az = math.atan2(v[horiz_axes_local[1]], v[horiz_axes_local[0]])
            el = math.asin(np.clip(v[up_idx] / max(dist, 1e-6), -1.0, 1.0))

            elev_deg_actual = math.degrees(el)
            az_deg_actual = math.degrees(az) % 360.0

        # Build camera with PyTorch3D look_at and per-view near/far clipping.
        eye_t = torch.tensor([eye], dtype=torch.float32, device=device)
        at_t = torch.tensor([center_tgt], dtype=torch.float32, device=device)
        up_t = torch.tensor([up_vec], dtype=torch.float32, device=device)
        R_t, T_t = look_at_view_transform(eye=eye_t, at=at_t, up=up_t, device=device)

        # Heuristic near/far based on camera distance
        cam_dist = float(np.linalg.norm(eye - center_tgt))
        znear = max(1e-2, cam_dist * 0.01)
        zfar = max(znear * 1000.0, cam_dist * 10.0)

        cameras = make_camera(R_t, T_t, znear=znear, zfar=zfar)

        # Lighting: strong ambient + "headlight" point light slightly in front of camera.
        vdir = (center_tgt - eye)
        vdir /= (np.linalg.norm(vdir) + 1e-12)
        light_pos = eye - 0.02 * cam_dist * vdir
        intensity = (1.0 if not is_fine else 1.3) * 0.6
        lights = PointLights(
            device=device,
            location=torch.from_numpy(light_pos[None, :]).float().to(device),
            ambient_color=((0.6 * intensity, 0.6 * intensity, 0.6 * intensity),),
            diffuse_color=((0.8 * intensity, 0.8 * intensity, 0.8 * intensity),),
            specular_color=((0.9 * intensity, 0.9 * intensity, 0.9 * intensity),),
        )

        renderer = make_renderer(cameras, lights)

        # Render RGBA, extract RGB, and save to PNG.
        with torch.no_grad():
            images = renderer(meshes)  # (1, H, W, 4)

        img = images[0, ..., :3].detach().cpu().numpy()
        out = os.path.join(output_dir, f"view_{i_idx:02d}.png")
        np_img_to_pil(img).save(out, quality=95)
        print(
            f"[INFO] Saved {os.path.basename(out)}  "
            f"(az={az_deg_actual:.1f}°, elev≈{elev_deg_actual:.1f}°)"
        )

        # Store extrinsics and diagnostic info for JSON.
        Rs.append(R_t[0].detach().cpu().numpy().astype(float))
        Ts.append(T_t[0].detach().cpu().numpy().astype(float))
        eyes.append(tuple(map(float, eye)))
        used_azims.append(float(az_deg_actual))
        used_elevs.append(float(elev_deg_actual))

    print(
        f"[INFO] Up-axis detected: {['X','Y','Z'][up_idx]}; "
        f"elevation {args.elev_deg:.1f}°, zoom {args.zoom}"
    )

    # ------------------------------------------------------------------
    # 6) Coarse ring: orbit around the entire scene.
    # ------------------------------------------------------------------
    idx = 0
    if (
        args.num_coarse_views
        if args.num_coarse_views is not None
        else (args.num_views if args.num_views is not None else 9)
    ) > 0:
        num_coarse = (
            args.num_coarse_views
            if args.num_coarse_views is not None
            else (args.num_views if args.num_views is not None else 9)
        )
        num_coarse = max(int(num_coarse), 0)

        base_dist = fit_distance_axis(extent, up_idx=up_idx, vfov_deg=VFOV_DEG, aspect=1.0)
        coarse_dist = max(base_dist * float(args.zoom), 1e-6)

        # Uniformly spaced azimuths around full 360°
        azims = np.linspace(0.0, 360.0, num=num_coarse, endpoint=False).tolist()
        for az in azims:
            eye = make_eye_for(center, az_deg=az, dist=coarse_dist, elev_deg_local=float(args.elev_deg))
            place_render(idx, eye, center, is_fine=False)
            idx += 1

    # ------------------------------------------------------------------
    # 7) Fine grid: 3x3 block-based close-up views with adaptive coverage.
    # ------------------------------------------------------------------
    num_fine_default = max(int(args.num_fine_views), 0)
    if num_fine_default > 0:
        # Shrink bbox to block size for estimating fine distances.
        fine_extent = extent.copy()
        fine_extent[horiz_axes[0]] /= 3.0
        fine_extent[horiz_axes[1]] /= 3.0

        fine_base_dist = fit_distance_axis(fine_extent, up_idx=up_idx, vfov_deg=VFOV_DEG, aspect=1.0)
        fine_dist = max(fine_base_dist * float(args.zoom) * float(args.fine_zoom), 1e-6)

        # Height statistics along up-axis for adaptive selection.
        verts_up = verts_cpu[:, up_idx] if verts_cpu.size else np.array([center[up_idx]])
        p8 = float(np.percentile(verts_up, 8)) if verts_up.size > 1 else float(verts_up[0])
        p30 = float(np.percentile(verts_up, 30)) if verts_up.size > 1 else float(verts_up[0])
        p85 = float(np.percentile(verts_up, 85)) if verts_up.size > 1 else float(verts_up[0])

        # Precompute 3x3 block centers in horizontal plane.
        block_centers = grid_block_centers(center, extent, up_idx)
        mins = center - 0.5 * extent
        steps = extent / 3.0

        def block_bounds(i, j):
            """
            Get [min,max) bounds in each horizontal axis for block (i,j).
            """
            x0a = mins[horiz_axes[0]] + i * steps[horiz_axes[0]]
            x1a = mins[horiz_axes[0]] + (i + 1) * steps[horiz_axes[0]]
            x0b = mins[horiz_axes[1]] + j * steps[horiz_axes[1]]
            x1b = mins[horiz_axes[1]] + (j + 1) * steps[horiz_axes[1]]
            return (x0a, x1a), (x0b, x1b)

        def azims_for_n(n):
            """
            Return uniform azimuth angles (degrees) for n views (n=1 -> [0]).
            """
            if n <= 1:
                return [0.0]
            return np.linspace(0.0, 360.0, num=n, endpoint=False).tolist()

        # Iterate over 3x3 blocks.
        b = 0
        for i in range(3):
            for j in range(3):
                b_center = block_centers[b]
                b += 1

                # Estimate average height in block to decide how many views it deserves.
                if verts_cpu.size:
                    (a0, a1), (b0, b1) = block_bounds(i, j)
                    a_axis, b_axis = horiz_axes[0], horiz_axes[1]
                    ax = verts_cpu[:, a_axis]
                    bx = verts_cpu[:, b_axis]
                    mask = (ax >= a0) & (ax < a1) & (bx >= b0) & (bx < b1)
                    if np.any(mask):
                        block_avg_h = float(np.mean(verts_cpu[mask, up_idx]))
                    else:
                        block_avg_h = float(ground_val)
                else:
                    block_avg_h = float(center[up_idx])

                # Adaptive number of views based on block height percentile.
                if block_avg_h <= p8:
                    n_views = 1
                elif block_avg_h <= p30:
                    n_views = 2
                elif block_avg_h > p85:
                    n_views = 4
                else:
                    n_views = num_fine_default

                if n_views == 1:
                    # "Top-down" single view: position camera above block center.
                    vfov = math.radians(VFOV_DEG)
                    hfov = 2.0 * math.atan(math.tan(vfov / 2.0) * 1.0)  # aspect=1

                    half_x = 0.5 * fine_extent[horiz_axes[0]]
                    half_y = 0.5 * fine_extent[horiz_axes[1]]
                    d_x = half_x / max(math.tan(hfov / 2.0), 1e-6)
                    d_y = half_y / max(math.tan(vfov / 2.0), 1e-6)
                    td_dist = max(d_x, d_y) * 1.05

                    # Apply backoff for safety.
                    td_dist *= max(args.backoff, 1e-6)

                    # Place camera directly above block center along up-axis.
                    eye = b_center.copy()
                    eye[up_idx] += td_dist

                    # Slight horizontal offset to avoid degenerate look-at (eye = target).
                    eps = 1e-3 * max(
                        fine_extent[horiz_axes[0]],
                        fine_extent[horiz_axes[1]],
                        1e-6,
                    )
                    eye[horiz_axes[0]] += eps

                    place_render(idx, eye, b_center, is_fine=True)
                    idx += 1
                else:
                    # Multiple azimuth views around this block.
                    for az in azims_for_n(n_views):
                        eye = make_eye_for(
                            b_center,
                            az_deg=az,
                            dist=fine_dist,
                            elev_deg_local=float(args.elev_deg),
                        )
                        place_render(idx, eye, b_center, is_fine=True)
                        idx += 1

    # ----------------------------------------------------------------------
    # 8) Save camera parameters JSON next to rendered images.
    # ----------------------------------------------------------------------
    fx, fy, cx, cy = fov_to_intrinsics(W, H, VFOV_DEG)
    cam_params = {
        "image_size": [H, W],
        "fov_deg_vertical": VFOV_DEG,
        "intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "near": None,
            "far": None,
        },
        "R": [R.tolist() for R in Rs],
        "T": [T.tolist() for T in Ts],
        "eyes": eyes,
        "azimuth_deg": used_azims,
        "elevation_deg_actual": used_elevs,
        "target": center.astype(float).tolist(),
        "up_axis": ["X", "Y", "Z"][up_idx],
        "ground_value_along_up": float(ground_val),
        "clearance_frac": CLEARANCE_FRAC,
        "meshlab_lighting": ALWAYS_MESHLAB_LIGHTING,
        "double_sided": ALWAYS_DOUBLE_SIDED,
        "zoom": float(args.zoom),
        "elev_deg_requested": float(args.elev_deg),
        "source": {
            "used_triangle_model": True,
            "path": os.path.abspath(args.input),
        },
    }
    with open(os.path.join(output_dir, "camera_params.json"), "w") as f:
        json.dump(cam_params, f, indent=2)

    print(f"[INFO] Done. Outputs saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
