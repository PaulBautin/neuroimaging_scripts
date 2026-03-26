import numpy as np
import trimesh
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt


# argparse



def build_vertex_adjacency(mesh: trimesh.Trimesh) -> list[list[int]]:
    """
    Build vertex adjacency list from triangular faces.

    Args:
        mesh: Trimesh surface.

    Returns:
        List of neighbor indices per vertex.
    """
    faces = mesh.faces
    n_vertices = mesh.vertices.shape[0]

    rows = np.hstack([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.hstack([faces[:, 1], faces[:, 2], faces[:, 0]])

    data = np.ones(len(rows), dtype=np.int8)
    adj = coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    adj = adj + adj.T  # make symmetric

    adjacency = [[] for _ in range(n_vertices)]
    for i, j in zip(*adj.tocoo().nonzero()):
        if i != j:
            adjacency[i].append(j)

    return adjacency


def compute_tangent_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute orthonormal tangent basis from normal.

    Args:
        normal: (3,) unit vector.

    Returns:
        t1, t2: orthonormal tangent vectors.
    """
    normal = normal / np.linalg.norm(normal)

    # pick stable auxiliary vector
    aux = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(aux, normal)) > 0.9:
        aux = np.array([0.0, 1.0, 0.0])

    t1 = np.cross(normal, aux)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)

    return t1, t2


def compute_surface_structure_tensors(
    ply_file: str,
    sigma: float | None = None
) -> dict[str, np.ndarray]:
    """
    Compute per-vertex 2x2 structure tensor in tangent plane.

    Args:
        ply_file: Path to .ply surface file.
        sigma: Optional Gaussian weighting scale in Euclidean distance.

    Returns:
        Dictionary containing:
            - tensors: (N, 2, 2)
            - principal_dir_3d: (N, 3)
            - anisotropy: (N,)
            - eigenvalues: (N, 2)
    """
    mesh = trimesh.load(ply_file, process=False)
    vertices = mesh.vertices
    normals = mesh.vertex_normals
    adjacency = build_vertex_adjacency(mesh)

    n_vertices = vertices.shape[0]

    tensors = np.zeros((n_vertices, 2, 2))
    principal_dir_3d = np.zeros((n_vertices, 3))
    eigenvalues = np.zeros((n_vertices, 2))
    anisotropy = np.zeros(n_vertices)

    for i in range(n_vertices):
        neighbors = adjacency[i]
        if len(neighbors) < 3:
            continue

        vi = vertices[i]
        ni = normals[i]
        t1, t2 = compute_tangent_frame(ni)

        diffs = vertices[neighbors] - vi

        # project into tangent plane
        x = diffs @ t1
        y = diffs @ t2
        coords = np.stack([x, y], axis=1)

        if sigma is not None:
            d2 = np.sum(diffs ** 2, axis=1)
            weights = np.exp(-d2 / (2 * sigma ** 2))
        else:
            weights = np.ones(len(neighbors))

        coords_centered = coords - np.average(coords, axis=0, weights=weights)

        C = np.zeros((2, 2))
        for p, w in zip(coords_centered, weights):
            C += w * np.outer(p, p)

        C /= np.sum(weights)

        tensors[i] = C

        # eigen decomposition
        vals, vecs = np.linalg.eigh(C)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        eigenvalues[i] = vals

        if vals[0] > 0:
            anisotropy[i] = (vals[0] - vals[1]) / (vals[0] + 1e-12)

        # map principal 2D direction back to 3D
        principal_2d = vecs[:, 0]
        principal_dir_3d[i] = principal_2d[0] * t1 + principal_2d[1] * t2

    return {
        "tensors": tensors,
        "principal_dir_3d": principal_dir_3d,
        "anisotropy": anisotropy,
        "eigenvalues": eigenvalues,
    }


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Compute surface structure tensors.")
    parser.add_argument("ply_file", type=str, help="Path to input .ply file.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save output .npy files.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Gaussian weighting scale in Euclidean distance.",
    )

    args = parser.parse_args()
    # args.sigma = 0.001
    # results = compute_surface_structure_tensors(args.ply_file, sigma=args.sigma)

    # os.makedirs(args.output_dir, exist_ok=True)
    # np.save(os.path.join(args.output_dir, "tensors.npy"), results["tensors"])
    # np.save(os.path.join(args.output_dir, "principal_dir_3d.npy"), results["principal_dir_3d"])
    # np.save(os.path.join(args.output_dir, "anisotropy.npy"), results["anisotropy"])
    # np.save(os.path.join(args.output_dir, "eigenvalues.npy"), results["eigenvalues"])
    from brainspace.mesh.mesh_io import read_surface, write_surface
    mesh = read_surface('/data/mica/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0/sub-HC062/ses-01/surf/sub-HC062_ses-01_hemi-L_space-nativepro_surf-fsLR-32k_label-white.surf.gii')
    write_surface(mesh, "/local_raid/data/pbautin/downloads/spc-04_hem-R_epo-all_dissection/bradipho_data/spc-04/dissection/spc-04_hem-R_epo-09_dissection.vtp")
    mesh = trimesh.load(args.ply_file, file_type='ply', process=False)
    mesh.export("/local_raid/data/pbautin/downloads/spc-04_hem-R_epo-all_dissection/bradipho_data/spc-04/dissection/spc-04_hem-R_epo-09_dissection.stl")
    from PIL import Image
    mesh.visual.material.image = Image.open('/local_raid/data/pbautin/downloads/spc-04_hem-R_epo-all_dissection/bradipho_data/spc-04/dissection/spc-04_hem-R_epo-09_dissection.jpg')
    mesh.visual = mesh.visual.to_color()

    

    principal_dir_3d = np.load(os.path.join(args.output_dir, "principal_dir_3d.npy"))
    anisotropy = np.load(os.path.join(args.output_dir, "anisotropy.npy"))


    lines = []
    step = 0.0015
    n_steps = 2

    # visualize principal directions as line segments colored by direction
    nb_vertices = mesh.vertices.shape[0]
    for vid in range(0, nb_vertices, 10):
        start = mesh.vertices[vid]
        direction = principal_dir_3d[vid]
        anisotropy_value = anisotropy[vid]
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            continue
        direction = direction / norm

        points = np.array([start + direction * i * step * anisotropy_value for i in range(n_steps)])
        # vertices are local to this line, so indices go 0..n_steps-1
        # color line by rgb direction
        color = (np.abs(direction) * 255).astype(np.uint8)
        line = trimesh.path.entities.Line(points=np.arange(n_steps), color=color)
        path = trimesh.path.Path3D(entities=[line], vertices=points)
        lines.append(path)

    # build a single scene with mesh and all line paths
    scene = trimesh.Scene([mesh, *lines])
    scene.show()

    
    
    # Direction-encoded coloring (like DTI RGB maps)
    # |X| -> R, |Y| -> G, |Z| -> B, weighted by anisotropy
    abs_dir = np.abs(principal_dir_3d)  # shape (N, 3)
    # Normalize each direction vector to unit length
    norms = np.linalg.norm(abs_dir, axis=-1, keepdims=True)
    abs_dir = abs_dir / (norms + 1e-12)

    # Weight by anisotropy (FA-like scalar, 0=isotropic, 1=highly anisotropic)
    anisotropy_norm = anisotropy / (anisotropy.max() + 1e-12)
    rgb = abs_dir * anisotropy_norm[:, np.newaxis]  # scale brightness by anisotropy

    # Convert to uint8 RGBA
    colors = np.concatenate([
        (rgb * 255).astype(np.uint8),
        np.full((len(rgb), 1), 255, dtype=np.uint8)  # alpha channel
    ], axis=-1)

    mesh.visual.vertex_colors = colors
    mesh.show()


if __name__ == "__main__":
    main()