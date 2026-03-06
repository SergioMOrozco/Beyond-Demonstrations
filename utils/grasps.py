import numpy as np

def normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n

def quat_from_R(R):
    """Convert 3x3 rotation matrix to (x,y,z,w) quaternion."""
    # Robust conversion
    m = R
    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2,1] - m[1,2]) / s
        qy = (m[0,2] - m[2,0]) / s
        qz = (m[1,0] - m[0,1]) / s
    elif (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
        s = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
        qw = (m[2,1] - m[1,2]) / s
        qx = 0.25 * s
        qy = (m[0,1] + m[1,0]) / s
        qz = (m[0,2] + m[2,0]) / s
    elif m[1,1] > m[2,2]:
        s = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
        qw = (m[0,2] - m[2,0]) / s
        qx = (m[0,1] + m[1,0]) / s
        qy = 0.25 * s
        qz = (m[1,2] + m[2,1]) / s
    else:
        s = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
        qw = (m[1,0] - m[0,1]) / s
        qx = (m[0,2] + m[2,0]) / s
        qy = (m[1,2] + m[2,1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q = q / np.linalg.norm(q)
    return q

def estimate_table_z(pc):
    """Rough: take low percentile as table height; works in your sim scenes."""
    return float(np.percentile(pc[:,2], 2.0))

def topdown_antipodal_grasps(
    pc,
    n_candidates=200,
    sample_pairs=4000,
    min_width=0.01,
    max_width=0.08,
    approach_clearance=0.06,
    finger_depth=0.05,
    table_clearance=0.004,
    seed=0,
):
    """
    Generate simple top-down (approach = -Z_world) antipodal grasps.
    Returns list of dicts: {pos, quat(xyzw), width, score}
    """
    rng = np.random.default_rng(seed)
    pc = np.asarray(pc, dtype=np.float64)
    if pc.shape[0] < 50:
        return []

    # Object centroid and rough table height
    c = pc.mean(axis=0)
    z_table = estimate_table_z(pc)

    # Only keep points above table a bit (helps if plate points exist)
    keep = pc[:,2] > (z_table + table_clearance)
    pc2 = pc[keep]
    if pc2.shape[0] < 50:
        pc2 = pc  # fallback

    # Precompute for fast checks
    # We'll use XY pairs for top-down pinch; treat closing axis in XY plane.
    idx = rng.integers(0, pc2.shape[0], size=(sample_pairs, 2))
    p1 = pc2[idx[:,0]]
    p2 = pc2[idx[:,1]]
    d = p2 - p1
    dist = np.linalg.norm(d[:, :2], axis=1)  # width in XY (since top-down)
    valid = (dist >= min_width) & (dist <= max_width)

    p1 = p1[valid]
    p2 = p2[valid]
    d = d[valid]
    dist = dist[valid]
    if p1.shape[0] == 0:
        return []

    # Closing axis = along (p2 - p1) in XY plane
    y_axis = np.stack([d[:,0], d[:,1], np.zeros_like(d[:,0])], axis=1)
    y_norm = np.linalg.norm(y_axis, axis=1)
    y_axis = y_axis / (y_norm[:,None] + 1e-9)

    # Approach axis = -Z_world (top-down)
    x_axis = np.tile(np.array([0.0, 0.0, -1.0]), (y_axis.shape[0], 1))

    # z_axis = x cross y (right-handed)
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis, axis=1)
    ok = z_norm > 1e-6
    x_axis, y_axis, z_axis, p1, p2, dist = x_axis[ok], y_axis[ok], z_axis[ok], p1[ok], p2[ok], dist[ok]
    z_axis = z_axis / (np.linalg.norm(z_axis, axis=1)[:,None] + 1e-9)

    centers = 0.5 * (p1 + p2)

    # Simple collision-ish filters:
    # 1) Don't put grasp below table
    ok = centers[:,2] > (z_table + table_clearance)
    x_axis, y_axis, z_axis, centers, dist = x_axis[ok], y_axis[ok], z_axis[ok], centers[ok], dist[ok]

    # 2) Finger depth / approach clearance: require free space along approach above the grasp
    # Approx: ensure there are not many points above the center within a small XY radius (gripper body zone).
    if centers.shape[0] == 0:
        return []

    # Score: closeness to centroid in XY + larger clearance to table
    xy_center_dist = np.linalg.norm((centers - c)[:, :2], axis=1)
    score = -xy_center_dist + 0.2*(centers[:,2] - z_table) - 0.1*np.abs(dist - 0.5*(min_width+max_width))

    # Keep top N before expensive checks
    topk = min(n_candidates * 5, centers.shape[0])
    sel = np.argpartition(-score, topk-1)[:topk]
    centers, x_axis, y_axis, z_axis, dist, score = centers[sel], x_axis[sel], y_axis[sel], z_axis[sel], dist[sel], score[sel]

    # Build poses + final simple "body clearance" check
    grasps = []
    for i in range(centers.shape[0]):
        p = centers[i].copy()

        # Pregrasp is above along +Z_world (since approach is -Z)
        pre = p + np.array([0, 0, approach_clearance], dtype=np.float64)

        # Quick check: gripper body column between pre and p should be mostly empty near (x,y)
        # Cylinder radius ~ 1.5cm
        r = 0.015
        # Points within radius in XY
        dx = pc2[:,0] - p[0]
        dy = pc2[:,1] - p[1]
        in_r = (dx*dx + dy*dy) < (r*r)
        # Points between p.z and pre.z
        in_z = (pc2[:,2] > p[2]) & (pc2[:,2] < pre[2])
        if np.count_nonzero(in_r & in_z) > 15:
            continue

        # Rotation matrix columns = [x_axis, y_axis, z_axis]
        R = np.stack([x_axis[i], y_axis[i], z_axis[i]], axis=1)
        q = quat_from_R(R)

        grasps.append(
            dict(
                pos=p,
                quat=q,  # (x,y,z,w)
                width=float(dist[i]),
                score=float(score[i]),
                pregrasp=pre,
            )
        )

    grasps.sort(key=lambda g: g["score"], reverse=True)
    return grasps[:n_candidates]