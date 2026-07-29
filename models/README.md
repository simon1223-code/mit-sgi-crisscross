# Benchmark models

| Model | Genus | Purpose |
|---|---:|---|
| `sphere.obj` | 0 | Analytic 3D SDF and circular slice reference |
| `cylinder.obj` | 0 | Analytic capped-cylinder SDF and circular slice reference |
| `torus.obj` | 1 | Analytic torus SDF and annular slice with a hole |
| `double_torus.obj` | 2 | Multiple-handle topology stress test |
| `bunny.obj` | 0 after repair | Scanned organic benchmark |

The analytic meshes are generated deterministically by `generate_models.py`. The script verifies watertightness, connectedness, Euler characteristic, and genus before exporting them.

The supplied bunny is a reduced Stanford Bunny mesh. Its open boundary loops are capped by `01_data_preparation.ipynb` before SDF computation. The original Stanford dataset and research-use terms are documented by the [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/).
