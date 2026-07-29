from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes


ROOT = Path(__file__).resolve().parent


def genus(mesh):
    components = mesh.split(only_watertight=False)
    if not mesh.is_watertight or len(components) != 1:
        return None
    return int((2 - mesh.euler_number) // 2)


def implicit_double_torus(resolution=160):
    axis = np.linspace(-1.35, 1.35, resolution, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    major_radius = 0.58
    minor_radius = 0.22
    separation = 0.38

    def torus_sdf(center_x):
        radial = np.sqrt((x - center_x) ** 2 + y**2) - major_radius
        return np.sqrt(radial**2 + z**2) - minor_radius

    field = np.minimum(torus_sdf(-separation), torus_sdf(separation))
    vertices, faces, _, _ = marching_cubes(field, level=0, spacing=(axis[1] - axis[0],) * 3)
    vertices += axis[0]
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale(1.7 / np.ptp(mesh.vertices, axis=0).max())
    return mesh


models = {
    "sphere.obj": trimesh.creation.icosphere(subdivisions=4, radius=0.65),
    "cylinder.obj": trimesh.creation.cylinder(radius=0.5, height=1.2, sections=256),
    "torus.obj": trimesh.creation.torus(
        major_radius=0.58,
        minor_radius=0.20,
        major_sections=160,
        minor_sections=64,
    ),
    "double_torus.obj": implicit_double_torus(),
}

expected_genus = {
    "sphere.obj": 0,
    "cylinder.obj": 0,
    "torus.obj": 1,
    "double_torus.obj": 2,
}

for name, mesh in models.items():
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    actual_genus = genus(mesh)
    if actual_genus != expected_genus[name]:
        raise ValueError(f"{name}: expected genus {expected_genus[name]}, found {actual_genus}")
    mesh.export(ROOT / name)
    print(name, len(mesh.vertices), len(mesh.faces), mesh.euler_number, actual_genus)
