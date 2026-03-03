import numpy as np
from pxr import Usd, UsdGeom, Gf
import omni.usd

def apply_gf_matrix4d(points, gf_mat4):
    # points: (N,3)
    N = points.shape[0]
    homog = np.ones((N, 4), dtype=np.float64)
    homog[:, :3] = points.astype(np.float64)
    M = np.array(gf_mat4, dtype=np.float64)          # (4,4)
    out = homog @ M.T
    return out[:, :3].astype(np.float32)

def sample_points_on_triangles(V, F, n):
    """
    V: (Nv, 3) vertices
    F: (Nf, 3) triangle indices
    n: number of points
    returns: (n, 3) sampled points in the same frame as V
    """
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    # triangle areas
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    area = np.maximum(area, 1e-12)

    # choose triangles proportional to area
    p = area / area.sum()
    tri_idx = np.random.choice(len(F), size=n, p=p)

    a = v0[tri_idx]
    b = v1[tri_idx]
    c = v2[tri_idx]

    # barycentric sampling (uniform on triangle)
    u = np.random.rand(n, 1)
    v = np.random.rand(n, 1)
    # fold to keep u+v <= 1
    mask = (u + v) > 1.0
    u[mask] = 1.0 - u[mask]
    v[mask] = 1.0 - v[mask]

    pts = a + u * (b - a) + v * (c - a)
    return pts

def _triangulate(faceVertexCounts, faceVertexIndices):
    """Triangulate polygon faces into triangles (fan triangulation)."""
    tris = []
    idx = 0
    for c in faceVertexCounts:
        face = faceVertexIndices[idx: idx + c]
        idx += c
        if c < 3:
            continue
        # fan: (0, i, i+1)
        for i in range(1, c - 1):
            tris.append([face[0], face[i], face[i + 1]])
    return np.asarray(tris, dtype=np.int64)

def quat_wxyz_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)

def transform_points(pts_local, pos_w, quat_wxyz):
    R = quat_wxyz_to_rotmat(quat_wxyz)
    return (pts_local @ R.T) + pos_w[None, :]

def sample_mesh_points_global(mesh_prim_path: str, n: int, pos: np.ndarray, quat: np.ndarray, time_code=None) -> np.ndarray:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(mesh_prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path: {mesh_prim_path}")

    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        raise TypeError(f"Prim at {mesh_prim_path} is not a UsdGeom.Mesh")

    tc = Usd.TimeCode.Default() if time_code is None else time_code

    V = np.asarray(mesh.GetPointsAttr().Get(tc), dtype=np.float64)
    counts = mesh.GetFaceVertexCountsAttr().Get(tc)
    indices = mesh.GetFaceVertexIndicesAttr().Get(tc)
    F = _triangulate(counts, indices)          # uses your triangulate helper
    pts_local = sample_points_on_triangles(V, F, n) # uses your barycentric sampler

    # transform mesh-local -> world using USD (includes scale/offset)
    prim = stage.GetPrimAtPath(mesh_prim_path)
    xform = UsdGeom.Xformable(prim)
    T_l2w = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pts_local = apply_gf_matrix4d(pts_local, T_l2w)
    pts_w = transform_points(pts_local, pos, quat)
    return pts_w

def save_pointclouds(pointcloud_list, filename="datasets/pointclouds.npz"):
    """
    pointcloud_list: list of (N,3) numpy arrays
    """
    # Convert torch tensors if needed
    pcs = []
    for pc in pointcloud_list:
        if hasattr(pc, "detach"):  # torch tensor
            pc = pc.detach().cpu().numpy()
        pcs.append(pc)

    # Save as numbered arrays
    np.savez_compressed(filename, *pcs)
    print(f"Saved {len(pcs)} point clouds to {filename}")