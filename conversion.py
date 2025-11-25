import os
import trimesh
from pathlib import Path

def convert_ply_to_obj_with_texture(ply_path, texture_path, output_dir):
    ply_path = Path(ply_path)
    texture_path = Path(texture_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PLY using Trimesh
    mesh = trimesh.load_mesh(ply_path, process=False)

    # Ensure texture is applied
    material_name = ply_path.stem
    obj_filename = output_dir / f"{material_name}.obj"
    mtl_filename = output_dir / f"{material_name}.mtl"
    texture_filename = texture_path.name  # Just the basename

    # Write OBJ with mtllib reference manually
    with open(obj_filename, 'w') as obj_file:
        # Write mtllib reference
        obj_file.write(f"mtllib {mtl_filename.name}\n")

        # Write vertex positions
        for v in mesh.vertices:
            obj_file.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Write texture coordinates
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            for uv in mesh.visual.uv:
                obj_file.write(f"vt {uv[0]} {uv[1]}\n")

        # Write normals (optional)
        if mesh.vertex_normals is not None:
            for n in mesh.vertex_normals:
                obj_file.write(f"vn {n[0]} {n[1]} {n[2]}\n")

        # Write faces
        for face in mesh.faces:
            face_line = "f"
            for idx in face:
                tex_idx = idx + 1
                face_line += f" {tex_idx}/{tex_idx}/{tex_idx}"
            obj_file.write(face_line + "\n")

    # Write MTL file
    with open(mtl_filename, 'w') as mtl_file:
        mtl_file.write(f"newmtl {material_name}\n")
        mtl_file.write("Ka 1.000 1.000 1.000\n")
        mtl_file.write("Kd 1.000 1.000 1.000\n")
        mtl_file.write("Ks 0.000 0.000 0.000\n")
        mtl_file.write("d 1.0\n")
        mtl_file.write("illum 2\n")
        mtl_file.write(f"map_Kd {texture_filename}\n")

    # Copy texture to output_dir if needed
    output_texture_path = output_dir / texture_filename
    if not output_texture_path.exists():
        import shutil
        shutil.copy(texture_path, output_texture_path)

    print(f"[✓] Saved: {obj_filename} and {mtl_filename}")


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True, help="Input .ply file path")
    parser.add_argument("--texture", required=True, help="Input .jpg texture path")
    parser.add_argument("--output_dir", required=True, help="Directory to save .obj and .mtl")
    args = parser.parse_args()

    convert_ply_to_obj_with_texture(args.ply, args.texture, args.output_dir)
