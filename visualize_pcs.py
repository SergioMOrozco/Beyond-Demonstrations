import numpy as np
import open3d as o3d
import time

def global_bbox(pcs, max_frames=200):
    mins = []
    maxs = []
    step = max(1, len(pcs)//max_frames)
    for pc in pcs[::step]:
        pts = np.asarray(pc, dtype=np.float64)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] == 0:
            continue
        mins.append(pts.min(0))
        maxs.append(pts.max(0))
    if not mins:
        raise ValueError("No finite points in any frames.")
    mn = np.min(np.stack(mins), axis=0)
    mx = np.max(np.stack(maxs), axis=0)
    return mn, mx


def load_pointclouds(filename="pointclouds.npz"):
    data = np.load(filename)
    pcs = [data[key] for key in data.files]
    print(f"Loaded {len(pcs)} point clouds")
    return pcs

def visualize_pointcloud(pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    # Optional: give random color
    colors = np.random.rand(len(pc), 3)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

def play_pointcloud(pcs, fps=30, point_size=4.0, voxel_size=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PointCloud Video", width=1280, height=720)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.show_coordinate_frame = False
    opt.background_color = np.asarray([0, 0, 0], dtype=np.float32)
    opt.point_size = float(point_size)

    def prep(pc):
        pts = np.asarray(pc, dtype=np.float64)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] == 0:
            return pts
        if voxel_size is not None:
            tmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(voxel_size)
            pts = np.asarray(tmp.points)
        return pts

    # --- first frame: set points and auto-fit camera ONCE ---
    i0 = 0
    while i0 < len(pcs):
        pts0 = prep(pcs[i0])
        if pts0.shape[0] > 0:
            break
        i0 += 1
    if i0 == len(pcs):
        raise ValueError("All point clouds are empty or non-finite.")

    pcd.points = o3d.utility.Vector3dVector(pts0)


    mn, mx = global_bbox(pcs)
    bbox = o3d.geometry.AxisAlignedBoundingBox(mn, mx)
    vis.reset_view_point(True)          # safe to call once after something exists
    vis.get_view_control()              # ensure view control exists
    # easiest way to fit to bbox using old API: temporarily set pcd to bbox corners
    pcd.points = o3d.utility.Vector3dVector(np.array([
        mn, mx,
        [mn[0], mn[1], mx[2]],
        [mn[0], mx[1], mn[2]],
        [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]],
        [mx[0], mn[1], mx[2]],
        [mn[0], mx[1], mx[2]],
    ], dtype=np.float64))

    pcd.paint_uniform_color([1, 1, 1])
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    vis.reset_view_point(True)   # <-- auto-fit once

    dt = 1.0 / fps

    # --- playback: update geometry, but DO NOT touch camera ---
    for pc in pcs[i0+1:]:
        pts = prep(pc)
        if pts.shape[0] == 0:
            print("broken")
            continue

        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color([1, 1, 1])
        vis.update_geometry(pcd)

        vis.poll_events()     # <-- allows your camera control
        vis.update_renderer()
        time.sleep(dt)

    vis.destroy_window()


if __name__ == "__main__":
    pcs = load_pointclouds("datasets/pointclouds.npz")

    play_pointcloud(pcs, fps=30, voxel_size=0.002)